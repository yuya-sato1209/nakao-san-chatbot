import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import pytz
import json

# ▼▼▼ ハイブリッド検索 & 日本語処理に必要なライブラリ ▼▼▼
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- 定数定義 ---
SPREADSHEET_ID = "1xeuewRd2GvnLDpDYFT5IJ5u19PUhBOuffTfCyWmQIzA" 

# --- Streamlit UI設定 ---
st.set_page_config(page_title="ナカオさんの函館歴史探訪", layout="wide")
st.title("🎓 ナカオさんの函館歴史探訪")

# --- APIキーの読み込み ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI APIキーが見つかりません。.envファイルまたはStreamlitのSecretsに設定してください。")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# --- ▼▼▼ 日本語トークナイザー（形態素解析）の準備 ▼▼▼ ---
# BM25で日本語を正しく単語区切りにするための関数です
def get_japanese_tokenizer():
    try:
        from fugashi import Tagger
        # -Owakati は分かち書き（スペース区切り）を出力するオプション
        tagger = Tagger('-Owakati')
        
        def tokenize(text):
            # テキストを単語リストに変換して返す
            return tagger.parse(text).split()
            
        return tokenize
    except ImportError:
        # ライブラリがない場合のフォールバック（文字単位分割）
        st.warning("⚠️ 'fugashi' ライブラリが見つかりません。精度が落ちる可能性があります。")
        return lambda text: list(text)

# トークナイザーを初期化
japanese_tokenizer = get_japanese_tokenizer()

# --- データ読み込み関数 ---
@st.cache_data
def load_raw_data():
    all_data = []
    try:
        with open("rag_data_cleaned.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        return []
    return all_data

# --- ハイブリッド検索の構築 ---
@st.cache_resource
def setup_retrievers(_raw_data):
    if not _raw_data:
        return None

    # 1. ドキュメントの作成
    documents = []
    for data in _raw_data:
        if data.get("text") and data.get("text").strip():
            doc = Document(
                page_content=data["text"],
                metadata={
                    "source_video": data.get("source_video", "不明なソース"),
                    "url": data.get("url", "#")
                }
            )
            documents.append(doc)

    # 2. テキスト分割
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    
    if not split_docs:
        return None

    # 3. ベクトル検索機 (FAISS) の作成 - 「意味」で探す
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embedding=embedding)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

    # 4. キーワード検索機 (BM25) の作成 - 「単語」で探す
    # ▼▼▼ ここで日本語トークナイザーを適用 ▼▼▼
    bm25_retriever = BM25Retriever.from_documents(
        split_docs,
        preprocess_func=japanese_tokenizer # 日本語対応を追加
    )
    bm25_retriever.k = 2

    # 5. アンサンブル検索機 (Hybrid) の作成
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7]
    )
    
    return ensemble_retriever

# --- プロンプトテンプレート ---
template = """
あなたは、函館の歴史を案内するベテランガイドのAさんです。


--- 重要ルール：ユーザーの質問内容に誤字・略称・曖昧性がある場合 ---
参考情報またはあなたの知識をもとに「正しい名称へ訂正して回答」してください。
訂正は丁寧に行い、「正しくは〜です」という形で伝えてください。
複数の候補がある場合、質問に関連するもののみを使って回答してください。
明らかに関係ない参考情報は無視してください。

--- 回答の方針 ---
1. あなたの回答は、参考情報を基に作成してください。
2. 過去の「会話の履歴」も踏まえて、自然な会話になるようにしてください。

--- 話し方の特徴 ---
- 街歩きに参加した人たちからの質問に、まるでその場で語りかけるように参考情報の文脈で話してください。


--- 参考情報 ---
{context}
--- 会話の履歴 ---
{chat_history}
--- ユーザーの質問 ---
{question}
"""
prompt_template = PromptTemplate.from_template(template)

# --- LLM + 検索チェーンの準備 ---
# ▼▼▼ モデル名を正しい "gpt-4o" に修正 ▼▼▼
llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.3) 
raw_data = load_raw_data()

# 検索機のセットアップ
retriever = setup_retrievers(raw_data)

if retriever:
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever, # ハイブリッド検索機を使用
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )
else:
    st.error("知識源データが読み込めませんでした。rag_data_cleaned.jsonlを確認してください。")
    st.stop()

# --- Googleスプレッドシート連携 ---
@st.cache_resource
def connect_to_gsheet():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict)
        scoped_creds = creds.with_scopes([
            "https://www.googleapis.com/auth/spreadsheets"
        ])
        client = gspread.authorize(scoped_creds)
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet("log")
        return worksheet
    except Exception as e:
        st.error("Googleスプレッドシートへの接続に失敗しました。Secretsと共有設定を再確認してください。")
        return None

def append_log_to_gsheet(worksheet, username, query, response):
    if worksheet is not None:
        try:
            jst = pytz.timezone('Asia/Tokyo')
            timestamp = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')
            worksheet.append_row([timestamp, username, query, response])
        except Exception:
            pass

worksheet = connect_to_gsheet()

# --- チャット機能 ---
if "username" not in st.session_state:
    st.session_state.username = ""

if st.session_state.username == "":
    st.session_state.username = st.text_input("ニックネームを入力して、Enterキーを押してください", key="username_input")
    if st.session_state.username:
        st.rerun()
else:
    st.write(f"こんにちは、{st.session_state.username}さん！")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if "source_documents" in message and message["source_documents"]:
                    with st.expander("🔍 回答の根拠となったテキスト"):
                        for doc in message["source_documents"]:
                            video_title = doc.metadata.get("source_video", "不明なソース")
                            video_url = doc.metadata.get("url", "#")
                            st.write(f"**参照元:** [{video_title}]({video_url})")
                            st.write(f"> {doc.page_content}")

    if query := st.chat_input("💬 函館の街歩きに基づいて質問してみてください"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append((msg["content"], ""))
                    elif msg["role"] == "assistant":
                        if chat_history:
                            last_q, _ = chat_history[-1]
                            chat_history[-1] = (last_q, msg["content"])

                result = qa({"question": query, "chat_history": chat_history})
                response = result["answer"]
                
                st.markdown(response)
                
                append_log_to_gsheet(worksheet, st.session_state.username, query, response)
                
                with st.expander("🔍 回答の根拠となったテキスト"):
                    for doc in result["source_documents"]:
                        video_title = doc.metadata.get("source_video", "不明なソース")
                        video_url = doc.metadata.get("url", "#")
                        st.write(doc.page_content)
                        st.write(f"**参照元:** [{video_title}]({video_url})")

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "source_documents": result["source_documents"]
                })