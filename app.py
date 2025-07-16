import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.documents import Document # Documentを直接作成するために追加
import os
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import pytz
import json # JSONLファイルを読み込むために追加

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

# --- RAG用ベクトルDBの構築 ---
@st.cache_resource
def load_vectorstore():
    documents_with_metadata = []
    # TextLoaderの代わりにjsonlファイルを一行ずつ読み込む
    with open("rag_data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # 読み込んだデータからDocumentオブジェクトを作成
            doc = Document(
                page_content=data["text"],
                metadata={
                    "source_video": data.get("source_video", "不明なソース"),
                    "url": data.get("url", "#"),
                    "start_time": data.get("start_time", 0)
                }
            )
            documents_with_metadata.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents_with_metadata)
    embedding = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embedding=embedding)
    return vectordb


# --- ▼▼▼ プロンプトテンプレートを大幅に強化 ▼▼▼ ---
template = """
あなたは、函館の歴史を案内するベテランガイドのAさんです。
あなたの役割は、街歩きに参加した人たちからの質問に、まるでその場で語りかけるように、親しみやすく、かつ知識の深さを感じさせる口調で答えることです。




--- 参考情報 ---
{context}
--- 参考情報ここまで ---

--- 会話の履歴 ---
{chat_history}
--- 会話の履歴ここまで ---

上記の情報をすべて踏まえた上で、以下の「ユーザーの質問」にAさんとして答えてください。

--- ユーザーの質問 ---
{question}
--- ユーザーの質問ここまで ---
"""
prompt_template = PromptTemplate.from_template(template)

# --- LLM + 検索チェーンの準備 ---
llm = ChatOpenAI(model_name="gpt-4.1")
vectordb = load_vectorstore()
retriever = vectordb.as_retriever()

# ▼▼▼ 会話チェーンに、強化したプロンプトを正しく設定 ▼▼▼
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt_template} # この行が重要！
)

# --- Googleスプレッドシート連携 ---
# (この部分は変更なし)
@st.cache_resource
def connect_to_gsheet():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict)
        scoped_creds = creds.with_scopes([
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ])
        client = gspread.authorize(scoped_creds)
        spreadsheet = client.open("中尾さんChatBot会話ログ")
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

# ユーザー名入力フォーム
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

    # 過去の会話履歴を表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "source_documents" in message:
                with st.expander("🔍 参考に使われたテキスト"):
                    for doc in message["source_documents"]:
                        video_title = doc.metadata.get("source_video", "不明なソース")
                        base_url = doc.metadata.get("url", "#")
                        start_time = doc.metadata.get("start_time", 0)
                        timestamped_url = f"{base_url}&t={start_time}s"
                        st.write(f"**動画:** [{video_title}（{start_time}秒〜）]({timestamped_url})")
                        st.write(f"> {doc.page_content}")



    # チャット入力
    if query := st.chat_input("💬 函館の街歩きに基づいて質問してみてください"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                chat_history = []
                for message in st.session_state.messages[:-1]:
                    if message["role"] == "user":
                        chat_history.append((message["content"], ""))
                    elif message["role"] == "assistant":
                        if chat_history:
                            last_question, _ = chat_history[-1]
                            chat_history[-1] = (last_question, message["content"])

                result = qa({"question": query, "chat_history": chat_history})
                response = result["answer"]
                st.markdown(response)
                
                append_log_to_gsheet(worksheet, st.session_state.username, query, response)
                
                with st.expander("🔍 参考に使われたテキスト"):
                    for doc in result["source_documents"]:
                        video_title = doc.metadata.get("source_video", "不明なソース")
                        base_url = doc.metadata.get("url", "#")
                        start_time = doc.metadata.get("start_time", 0)
                        # YouTubeのタイムスタンプ付きURLを生成
                        timestamped_url = f"{base_url}&t={start_time}s"
                        st.write(f"**動画:** [{video_title}（{start_time}秒〜）]({timestamped_url})")
                        st.write(f"> {doc.page_content}")

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "source_documents": result["source_documents"]
                })
