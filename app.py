# app.py
import os
import json
from datetime import datetime
import pytz

import streamlit as st
from dotenv import load_dotenv

# LangChain / community
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain

# Hybrid retriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# -------------------------
# 設定
# -------------------------
SPREADSHEET_ID = "1xeuewRd2GvnLDpDYFT5IJ5u19PUhBOuffTfCyWmQIzA"
DATA_JSONL = "rag_data_cleaned.jsonl"  # JSONL ファイル名
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

# -------------------------
# Streamlit UI 基本
# -------------------------
st.set_page_config(page_title="ナカオさんの函館歴史探訪（ハイブリッド検索版）", layout="wide")
st.title("🎓 ナカオさんの函館歴史探訪（ハイブリッド検索 + GPT-5）")

# -------------------------
# OpenAI APIキー
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI APIキーが見つかりません。.env または Streamlit Secrets に設定してください。")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -------------------------
# 日本語トークナイザ準備（fugashi / MeCab） - BM25 用
# -------------------------
try:
    from fugashi import Tagger
    tagger = Tagger()
    def japanese_tokenize(text: str) -> str:
        # MeCab でトークン化してスペース区切りの形に変換（BM25用）
        tokens = [word.surface for word in tagger(text)]
        return " ".join(tokens)
    st.write("🔎 fugashi (MeCab) tokenizer: OK")
except Exception as e:
    # フォールバック（単純分割） — 精度は落ちる
    tagger = None
    def japanese_tokenize(text: str) -> str:
        # 簡易: 句点・空白で分割してスペースを入れる（精度落ち）
        s = text.replace("\n", " ")
        # insert spaces between Kanji/Hiragana/Katakana and ASCII sequences crudely
        return " ".join([t for t in s.split() if t])
    st.warning("⚠ fugashi (MeCab) が利用できません。BM25 の精度が落ちる可能性があります。")

# -------------------------
# データ読み込み
# -------------------------
@st.cache_data
def load_raw_data(path=DATA_JSONL):
    all_data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        # skip invalid lines
                        continue
    except FileNotFoundError:
        return []
    return all_data

raw_data = load_raw_data()

# -------------------------
# ベクトルストア + BM25 構築（メタデータをチャンクに明示的にコピー）
# -------------------------
@st.cache_resource
def build_retrievers(_raw_data):
    if not _raw_data:
        return None

    # 1) Document オブジェクトを作成（メタデータ含む）
    documents = []
    for item in _raw_data:
        text = item.get("text", "").strip()
        if not text:
            continue
        doc = Document(
            page_content=text,
            metadata={
                "source_video": item.get("source_video", "メタデータ未登録"),
                "url": item.get("url", "#")
            }
        )
        documents.append(doc)

    if not documents:
        return None

    # 2) チャンク分割（元メタデータをコピー）
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunked_docs = []
    for doc in documents:
        chunk_texts = splitter.split_text(doc.page_content)
        for t in chunk_texts:
            chunked_docs.append(Document(page_content=t, metadata=doc.metadata.copy()))

    # 3) FAISS (意味検索)
    embeddings = OpenAIEmbeddings()
    faiss_db = FAISS.from_documents(chunked_docs, embedding=embeddings)
    faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 4})

    # 4) BM25 (キーワード検索) — 日本語を事前トークン化して与える
    # BM25 は単語境界が重要なので、全テキストをトークン化しておく
    bm25_documents_for_index = []
    for d in chunked_docs:
        tokenized = japanese_tokenize(d.page_content)
        # BM25Retriever.from_documents expects Documents; we feed tokenized text as page_content
        bm25_documents_for_index.append(Document(page_content=tokenized, metadata=d.metadata.copy()))

    bm25_retriever = BM25Retriever.from_documents(bm25_documents_for_index)
    bm25_retriever.k = 3  # BM25 から上位3件を取る

    # 5) Ensemble (重み: BM25 0.3, FAISS 0.7)
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7]
    )

    return {
        "faiss_db": faiss_db,
        "faiss_retriever": faiss_retriever,
        "bm25_retriever": bm25_retriever,
        "ensemble_retriever": ensemble
    }

retrievers = build_retrievers(raw_data)
if not retrievers:
    st.error("知識源データが読み込めませんでした。rag_data_cleaned.jsonl を確認してください。")
    st.stop()

ensemble_retriever = retrievers["ensemble_retriever"]

# -------------------------
# プロンプト（誤字訂正・参考情報優先のルールを明確に）
# -------------------------
template = """
あなたは、函館の歴史を案内するベテランガイドのAさんです。
以下のルールに従って、親しみやすく深みのある口調で答えてください。

--- 重要ルール ---
1) 参考情報（context）を最優先にして回答してください。参考情報に矛盾がある場合は、参考情報を根拠にして説明してください。
2) ユーザーの質問に誤字・略称・表記ゆれが含まれている場合、参考情報とあなたの知識に基づいて**正しい名称へ訂正して回答**してください。
   訂正は丁寧に行い、「正しくは〜です」の形で伝えてください。
3) 参考情報の中で明らかに関係ないものは無視して、最も関連性の高い情報のみを使用してください。

--- 話し方の特徴 ---
語尾には「〜ですな」「〜というわけです」「〜なんですよ」などを使い、柔らかく断定的に話してください。

--- 参考情報 ---
{context}

--- 会話の履歴 ---
{chat_history}

--- ユーザーの質問 ---
{question}
"""
prompt_template = PromptTemplate.from_template(template)

# -------------------------
# LLM とチェーン構築（GPT-5 を指定）
# -------------------------
llm = ChatOpenAI(model_name="gpt-5-turbo", temperature=0.2)

combine_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt_template
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=ensemble_retriever,
    combine_docs_chain=combine_chain,
    return_source_documents=True
)

# -------------------------
# Google スプレッドシート接続（ログ用）
# -------------------------
@st.cache_resource
def connect_to_gsheet():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict)
        scoped_creds = creds.with_scopes(["https://www.googleapis.com/auth/spreadsheets"])
        client = gspread.authorize(scoped_creds)
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet("log")
        return worksheet
    except Exception as e:
        st.warning("Googleスプレッドシート接続に失敗しました（ログは無効になります）。")
        return None

worksheet = connect_to_gsheet()

def append_log_to_gsheet(worksheet, username, query, response):
    if worksheet is None:
        return
    try:
        jst = pytz.timezone('Asia/Tokyo')
        timestamp = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')
        worksheet.append_row([timestamp, username, query, response])
    except Exception:
        pass

# -------------------------
# チャット UI
# -------------------------
if "username" not in st.session_state:
    st.session_state.username = ""

if st.session_state.username == "":
    st.session_state.username = st.text_input("ニックネームを入力してEnterを押してください", key="username_input")
    if st.session_state.username:
        st.rerun()
else:
    st.write(f"こんにちは、{st.session_state.username}さん！")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("source_documents"):
                with st.expander("🔍 回答の根拠となったテキスト"):
                    for doc in message["source_documents"]:
                        src = doc.metadata.get("source_video", "メタデータ未登録")
                        url = doc.metadata.get("url", "#")
                        st.markdown(f"**参照元:** [{src}]({url})")
                        st.write(f"> {doc.page_content}")

    # 入力
    if query := st.chat_input("💬 函館の街歩きに基づいて質問してみてください"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                # 会話履歴の整形（(user, assistant) ペア）
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append((msg["content"], ""))
                    elif msg["role"] == "assistant" and chat_history:
                        last_q, _ = chat_history[-1]
                        chat_history[-1] = (last_q, msg["content"])

                # 実行（EnsembleRetriever を利用）
                result = qa_chain({"question": query, "chat_history": chat_history})
                response = result.get("answer", "")

                # レスポンス表示
                st.markdown(response)

                # 参考文献表示（フィルタ済のソースを表示）
                with st.expander("🔍 回答の根拠となったテキスト"):
                    src_docs = result.get("source_documents", [])
                    # src_docs は ensemble のドキュメント（BM25はトークン化を与えているので page_content がトークン列のものも混ざる）
                    # 表示用には、もし metadata に元のテキストを含めていればそれを使う設計が望ましい。
                    for doc in src_docs:
                        # もし BM25 側の tokenized text が渡ってきたら、短すぎる場合は metadata のオリジナルを使う工夫をしましょう
                        text_to_show = doc.page_content
                        if len(text_to_show) < 50 and doc.metadata.get("url"):
                            # 可能であれば、raw_data から元のテキストを探して表示（簡易処理）
                            text_to_show = doc.metadata.get("orig_text", doc.page_content)
                        st.write(text_to_show)
                        st.markdown(f"**参照元:** {doc.metadata.get('source_video','不明なソース')}  |  {doc.metadata.get('url','#')}")

                # ログ保存
                append_log_to_gsheet(worksheet, st.session_state.username, query, response)

                # 会話履歴に追加（assistant）
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "source_documents": result.get("source_documents", [])
                })
