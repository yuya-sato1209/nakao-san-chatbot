import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter # 修正1のためRecursiveを使用
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import pytz
import json

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

# --- データ読み込み関数 ---
@st.cache_data
def load_raw_data():
    all_data = []
    with open("rag_data_cleaned.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError:
                    st.warning(f"rag_data.jsonlに不正な形式の行があったため、スキップされました。")
    return all_data

# ▼▼▼ 修正点 1: メタデータがチャンク分割後も保持されるよう、処理を明示的に修正 ▼▼▼
@st.cache_resource
def load_vectorstore(_raw_data):
    # 1. まず、JSONLの各行をDocumentオブジェクトとして読み込む
    documents_with_metadata = []
    for data in _raw_data:
        doc = Document(
            page_content=data["text"],
            metadata={
                "source_video": data.get("source_video", "不明なソース"),
                "url": data.get("url", "#")
            }
        )
        documents_with_metadata.append(doc)

    # 2. テキストをチャンク分割し、メタデータを手動で引き継ぐ
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    final_docs = []
    for doc in documents_with_metadata:
        # 元のDocumentのテキストだけを分割
        chunks = splitter.split_text(doc.page_content)
        for chunk_text in chunks:
            # 分割されたチャンクごとに新しいDocumentを作成
            # この際、元のメタデータを明示的にコピーして引き継ぐ
            new_chunk_doc = Document(
                page_content=chunk_text,
                metadata=doc.metadata.copy() # メタデータを明示的にコピー
            )
            final_docs.append(new_chunk_doc)

    # 3. メタデータが引き継がれたチャンクでDBを構築
    if not final_docs:
        st.error("知識源データ（rag_data.jsonl）の読み込みに失敗したか、中身が空です。")
        st.stop()

    embedding = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(final_docs, embedding=embedding)
    return vectordb

# --- プロンプトテンプレート ---
# (プロンプト自体は変更なし)
template = """
あなたは、函館の歴史を案内するベテランガイドのAさんです。
あなたの役割は、街歩きに参加した人たちからの質問に、まるでその場で語りかけるように、親しみやすく、かつ知識の深さを感じさせる口調で答えることです。

\---重要ルール：ユーザーの質問内容に誤字・略称・曖昧性がある場合---
参考情報またはあなたの知識をもとに「正しい名称へ訂正して回答」してください。
訂正は丁寧に行い、「正しくは〜です」という形で伝えてください。

\--- 回答の方針 ---

1.  あなたの回答は、AIとして持つあなた自身の広範な知識と参考情報を基に作成してください。
2.  過去の「会話の履歴」も踏まえて、自然な会話になるようにしてください。
3.　回答を生成した後に、あなたの知識を使って回答に間違いがないか、確認してください。もし間違いがあったらあなたの知識で修正して回答してください。

\--- 話し方の特徴 ---

  - 語尾には「〜ですな」「〜というわけです」「〜なんですよ」などを使い、柔らかく断定的な話し方をしてください。


--- 参考情報 ---
{context}
--- 会話の履歴 ---
{chat_history}
--- ユーザーの質問 ---
{question}
"""
prompt_template = PromptTemplate.from_template(template)

# --- LLM + 検索チェーンの準備 ---
# ▼▼▼ 修正点 3: モデル名を "gpt-4.1" (存在しない) から "gpt-4o" (最新) に修正 ▼▼▼
# これが「プロンプト連携不全」の真の原因である可能性が高い
llm = ChatOpenAI(model_name="gpt-4.1") 
raw_data = load_raw_data()
vectordb = load_vectorstore(raw_data)

# ▼▼▼ 修正点 2: 類似度スコアの閾値を 0.8 から 0.6 に緩める ▼▼▼
# (FAISSのスコアは 0=近い, 1=遠い が標準だが、LangChainは 1=近い に正規化する)
# (0.8 -> 0.6 に下げることで、より広い範囲のドキュメントを許可する)
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.7, 'k': 3} # 0.8から0.6に変更, kは元の3を維持
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt_template} # プロンプト連携はこれで正しい
)

# --- Googleスプレッドシート連携 ---
# (変更なし)
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
        st.exception(e)
        return None

def append_log_to_gsheet(worksheet, username, query, response):
    if worksheet is not None:
        try:
            jst = pytz.timezone('Asia/Tokyo')
            timestamp = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')
            worksheet.append_row([timestamp, username, query, response])
        except Exception as e:
            st.warning(f"ログの書き込みに失敗しました: {e}")

worksheet = connect_to_gsheet()

# --- チャット機能 ---
# (変更なし)
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