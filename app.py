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

# ▼▼▼ ★ Reranker用ライブラリ ▼▼▼
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# ▲▲▲ 追加ここまで ▲▲▲
from langchain_community.retrievers import SentenceTransformerRerank
SPREADSHEET_ID = "1xeuewRd2GvnLDpDYFT5IJ5u19PUhBOuffTfCyWmQIzA"

st.set_page_config(page_title="ナカオさんの函館歴史探訪", layout="wide")
st.title("🎓 ナカオさんの函館歴史探訪")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI APIキーが見つかりません。.envまたはSecretsを確認してください。")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

@st.cache_data
def load_raw_data():
    all_data = []
    with open("rag_data_cleaned.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError:
                    st.warning("データ形式エラーのため一部スキップされました")
    return all_data

@st.cache_resource
def load_vectorstore(_raw_data):
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    final_docs = []
    for doc in documents_with_metadata:
        chunks = splitter.split_text(doc.page_content)
        for chunk_text in chunks:
            new_chunk_doc = Document(page_content=chunk_text, metadata=doc.metadata.copy())
            final_docs.append(new_chunk_doc)

    embedding = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(final_docs, embedding=embedding)
    return vectordb


# --- プロンプト（変更なし） ---
template = """
（※省略、元のまま）
"""
prompt_template = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)  # ←モデル修正推奨

raw_data = load_raw_data()
vectordb = load_vectorstore(raw_data)

# ---------------------------------------
# 🔍 Reranker付き Retriever に変更
# ---------------------------------------

# ★ まずFAISS通常検索
base_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# ★ BGE-rerankerを追加
reranker = SentenceTransformerRerank(
    model_name="BAAI/bge-reranker-v2-m3",
    top_k=3
)

retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    compressor=reranker
)

# ---------------------------------------

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,  # ←ここだけ変更
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

# --- Google Sheets連携（変更なし） ---
@st.cache_resource
def connect_to_gsheet():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict)
        scoped_creds = creds.with_scopes(
            ["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(scoped_creds)
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet("log")
        return worksheet
    except Exception as e:
        st.error("Googleスプレッドシートに接続できません。")
        st.exception(e)
        return None

def append_log_to_gsheet(worksheet, username, query, response):
    if worksheet:
        try:
            jst = pytz.timezone('Asia/Tokyo')
            timestamp = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')
            worksheet.append_row([timestamp, username, query, response])
        except Exception as e:
            st.warning(f"ログ保存エラー: {e}")

worksheet = connect_to_gsheet()

# --- チャットUI（変更なし） ---
if "username" not in st.session_state:
    st.session_state.username = ""

if st.session_state.username == "":
    st.session_state.username = st.text_input("ニックネーム入力 → Enter", key="username_input")
    if st.session_state.username:
        st.rerun()
else:
    st.write(f"こんにちは、{st.session_state.username}さん！")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "source_documents" in message:
                with st.expander("🔍 回答根拠"):
                    for doc in message["source_documents"]:
                        st.write(f"**参照元:** [{doc.metadata.get('source_video','?')}]({doc.metadata.get('url','#')})")
                        st.write(f"> {doc.page_content}")

    if query := st.chat_input("💬 質問してみてください"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append((msg["content"], ""))
                    else:
                        last_q, _ = chat_history[-1]
                        chat_history[-1] = (last_q, msg["content"])

                result = qa({"question": query, "chat_history": chat_history})
                response = result["answer"]
                st.markdown(response)

                append_log_to_gsheet(worksheet, st.session_state.username, query, response)

                with st.expander("🔍 回答根拠となった情報"):
                    for doc in result["source_documents"]:
                        st.write(f"📌 {doc.page_content}")
                        st.write(f"➡ {doc.metadata.get('source_video')} / {doc.metadata.get('url')}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "source_documents": result["source_documents"]
                })
