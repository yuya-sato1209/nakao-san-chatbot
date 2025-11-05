import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
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
    with open("rag_data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError:
                    st.warning("rag_data.jsonl に不正な行がありました（スキップされました）。")
    return all_data

@st.cache_resource
def load_vectorstore(_raw_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = []

    for data in _raw_data:
        base_doc = Document(
            page_content=data["text"],
            metadata={
                "source_video": data.get("source_video", "メタデータ未登録"),
                "url": data.get("url", "#")
            }
        )
        chunks = splitter.split_documents([base_doc])
        for c in chunks:
            c.metadata = base_doc.metadata  # メタデータを明示的に引き継ぐ
            split_docs.append(c)

    embedding = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(split_docs, embedding=embedding)
    return vectordb

# --- プロンプトテンプレート ---
template = """
あなたは、函館の歴史を案内するベテランガイドのAさんです。
あなたの役割は、街歩きに参加した人たちからの質問に、まるでその場で語りかけるように、
親しみやすく、かつ知識の深さを感じさせる口調で答えることです。

--- 回答生成の手順 ---
1. 以下の「参考情報」を読み、文字起こし特有の誤字・冗長表現を自然な日本語に頭の中で修正してください。
2. 参考情報と会話履歴を踏まえ、ユーザーの質問に答えてください。
3. 固有名詞は参考情報の通り正確に使用し、推測で補完しないでください。

--- 参考情報 ---
{context}
--- 会話の履歴 ---
{chat_history}
--- ユーザーの質問 ---
{question}
"""
prompt_template = PromptTemplate.from_template(template)

# --- LLM + 検索チェーンの準備 ---
llm = ChatOpenAI(model_name="gpt-5-turbo")


raw_data = load_raw_data()
vectordb = load_vectorstore(raw_data)
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 5}  # より多くの関連文を拾う
)

combine_chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=prompt_template)

qa = ConversationalRetrievalChain(
    retriever=retriever,
    combine_docs_chain=combine_chain,
    return_source_documents=True
)

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
        st.error("Googleスプレッドシートへの接続に失敗しました。")
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "source_documents" in message:
                with st.expander("🔍 回答の根拠となったテキスト"):
                    for doc in message["source_documents"]:
                        src = doc.metadata.get("source_video", "メタデータ未登録")
                        url = doc.metadata.get("url", "#")
                        st.markdown(f"**参照元:** [{src}]({url})")
                        st.write(f"> {doc.page_content}")

    if query := st.chat_input("💬 函館の街歩きに関する質問をどうぞ！"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append((msg["content"], ""))
                    elif msg["role"] == "assistant" and chat_history:
                        last_q, _ = chat_history[-1]
                        chat_history[-1] = (last_q, msg["content"])

                result = qa({"question": query, "chat_history": chat_history})
                response = result["answer"]

                st.markdown(response)
                append_log_to_gsheet(worksheet, st.session_state.username, query, response)

                with st.expander("🔍 回答の根拠となったテキスト"):
                    for doc in result["source_documents"]:
                        src = doc.metadata.get("source_video", "メタデータ未登録")
                        url = doc.metadata.get("url", "#")
                        st.markdown(f"**参照元:** [{src}]({url})")
                        st.write(f"> {doc.page_content}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "source_documents": result["source_documents"]
                })
