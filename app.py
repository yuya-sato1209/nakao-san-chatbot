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

# --- 定数定義 ---
# ▼▼▼ ここにあなたのスプレッドシートIDを設定してください ▼▼▼
SPREADSHEET_ID = "1z5kbcz-84A7--ziiicVy3TgcToi4qOmvisEbqad0daM" 

# --- Streamlit UI設定 ---
st.set_page_config(page_title="ナカオさんの函館歴史探訪", layout="wide")
st.title("� ナカオさんの函館歴史探訪")

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
                    st.warning(f"rag_data.jsonlに不正な形式の行があったため、スキップされました。")
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
    docs = splitter.split_documents(documents_with_metadata)
    embedding = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embedding=embedding)
    return vectordb

# --- プロンプトテンプレート ---
template = """
あなたは、函館の歴史を案内するベテランガイドのAさんです。
あなたの役割は、街歩きに参加した人たちからの質問に、まるでその場で語りかけるように、親しみやすく、かつ知識の深さを感じさせる口調で答えることです。


--- 参考情報 ---
{context}
--- 会話の履歴 ---
{chat_history}
--- ユーザーの質問 ---
{question}
"""
prompt_template = PromptTemplate.from_template(template)

# --- LLM + 検索チェーンの準備 ---
llm = ChatOpenAI(model_name="gpt-5")
raw_data = load_raw_data()
vectordb = load_vectorstore(raw_data)
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.7, 'k': 2}
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

# --- Googleスプレッドシート連携 ---
@st.cache_resource
def connect_to_gsheet():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict)
        # ▼▼▼ スコープからDriveを削除 ▼▼▼
        scoped_creds = creds.with_scopes([
            "https://www.googleapis.com/auth/spreadsheets"
        ])
        client = gspread.authorize(scoped_creds)
        # ▼▼▼ IDで直接スプレッドシートを開くように変更 ▼▼▼
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

# --- ヘルパー関数 ---
def extract_keywords(text):
    prompt = f"以下の文章から、函館の歴史に関連する重要なキーワード（地名、人名、出来事など）を最大3つまで抽出し、カンマ区切りでリストアップしてください。\n\n文章:\n{text}\n\nキーワード:"
    try:
        keyword_llm = ChatOpenAI(model_name="gpt-5", temperature=0)
        response = keyword_llm.invoke(prompt)
        keywords = [kw.strip() for kw in response.content.split(',') if kw.strip()]
        return keywords
    except Exception:
        return []

def find_videos_by_keywords(keywords, all_data):
    found_videos = {}
    if not keywords:
        return []
    for keyword in keywords:
        for item in all_data:
            if keyword.lower() in item["text"].lower():
                if item["url"] not in found_videos:
                    found_videos[item["url"]] = {
                        "title": item.get("source_video", "不明なソース"),
                        "url": item.get("url", "#")
                    }
    return list(found_videos.values())

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
                            st.write(f"**動画:** [{video_title}]({video_url})")
                            st.write(f"> {doc.page_content}")
                if "related_videos" in message and message["related_videos"]:
                    with st.expander("🎬 関連動画"):
                        for video in message["related_videos"]:
                            video_url = video['url']
                            st.write(f"**動画:** [{video['title']}]({video_url})")

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
                keywords = extract_keywords(response)
                related_videos = find_videos_by_keywords(keywords, raw_data)
                
                st.markdown(response)
                
                append_log_to_gsheet(worksheet, st.session_state.username, query, response)
                
                with st.expander("🔍 回答の根拠となったテキスト"):
                    for doc in result["source_documents"]:
                        video_title = doc.metadata.get("source_video", "不明なソース")
                        video_url = doc.metadata.get("url", "#")
                        st.write(f"**動画:** [{video_title}]({video_url})")
                        st.write(f"> {doc.page_content}")

                if related_videos:
                    with st.expander("🎬 関連動画"):
                        for video in related_videos:
                            video_url = video['url']
                            st.write(f"**動画:** [{video['title']}]({video_url})")

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "source_documents": result["source_documents"],
                    "related_videos": related_videos
                })
