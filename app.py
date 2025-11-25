import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import pytz
import json

# ▼▼▼ Hybrid Search ▼▼▼
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- Google sheet settings ---
SPREADSHEET_ID = "1xeuewRd2GvnLDpDYFT5IJ5u19PUhBOuffTfCyWmQIzA" 

# --- UI ---
st.set_page_config(page_title="ナカオさんの函館歴史探訪", layout="wide")
st.title("🎓 ナカオさんの函館歴史探訪")

# --- Load API ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI APIキーがありません。.envかStreamlit Secretsに設定してください。")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# --- Load data ---
@st.cache_data
def load_raw_data():
    rows = []
    try:
        with open("rag_data_cleaned.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except FileNotFoundError:
        st.error("rag_data_cleaned.jsonl が見つかりません。")
        return []
    return rows


# --- Create Hybrid Retriever ---
@st.cache_resource
def create_retriever(_raw):

    docs = []
    for data in _raw:
        if "text" in data and data["text"].strip():
            docs.append(
                Document(
                    page_content=data["text"],
                    metadata={
                        "source_video": data.get("source_video", "不明"),
                        "url": data.get("url", "#")
                    }
                )
            )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    faiss = vectorstore.as_retriever(search_kwargs={'k': 2})

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 2

    return EnsembleRetriever(retrievers=[faiss, bm25], weights=[0.55, 0.45])


# --- Prompt (修正版: 誤称訂正ルール強化) ---
prompt = PromptTemplate.from_template("""
あなたは函館の歴史を案内するベテラン語り部「ナカオさん」です。

📝 **重要ルール**
- ユーザーが名前を誤って入力した場合は、参考情報または知識をもとに **正しい表記へ直して回答**してください。
- 訂正はやさしく「正しくは〜です」と前置きしてから説明してください。

🎤 **話し方**
- 「〜ですな」「〜というわけなんです」「〜なんですよ」などの語尾で話してください。
- 文章は硬すぎず、温かい語り口。

📚 **参考情報**
{context}

💬 **会話履歴**
{chat_history}

❓ **質問**
{question}
""")


# --- Build RAG chain ---
@st.cache_resource
def build_qa_chain(retriever):
    llm = ChatOpenAI(model_name="gpt-5", temperature=0.3)

    combine_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    return RetrievalQA(
        retriever=retriever,
        combine_documents_chain=combine_chain,
        return_source_documents=True
    )


raw_data = load_raw_data()
retriever = create_retriever(raw_data)
qa_chain = build_qa_chain(retriever)


# --- Google Sheets logging ---
def connect_sheet():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict)
        client = gspread.authorize(creds.with_scopes(["https://www.googleapis.com/auth/spreadsheets"]))
        return client.open_by_key(SPREADSHEET_ID).worksheet("log")
    except:
        st.warning("📄 Google Sheetsに接続できませんでした。ログ保存はスキップします。")
        return None

sheet = connect_sheet()


def log_message(user, input_text, output):
    if sheet:
        timestamp = datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M")
        sheet.append_row([timestamp, user, input_text, output])


# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

user = st.sidebar.text_input("ニックネーム：", value="ゲスト")

if user:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if q := st.chat_input("💬 函館について聞いてください"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            with st.spinner("考えています…"):
                result = qa_chain({"query": q})
                answer = result["result"]

                st.markdown(answer)
                log_message(user, q, answer)

                if result["source_documents"]:
                    with st.expander("🔍 参考にした資料"):
                        for doc in result["source_documents"]:
                            st.write(f"📌 **{doc.metadata['source_video']}**")
                            st.write(doc.page_content)
                            st.markdown(f"[▶ 資料を見る]({doc.metadata['url']})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
