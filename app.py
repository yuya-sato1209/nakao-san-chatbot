import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import pytz

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
    loader = TextLoader("rag_trainning.txt", encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    embedding = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embedding=embedding)
    return vectordb

# --- プロンプトテンプレート ---
template = """
あなたはAさん本人として、函館の街歩きに参加した人たちからの質問に答えます。
口調・語尾・話し方の癖・思考の特徴などは、以下のテキストから忠実に学び、再現してください。

--- ルール ---
- 与えられた「参考情報」に書かれている内容だけを元に、忠実に回答してください。
- あなた自身の知識や、参考情報にない内容は絶対に追加しないでください。
- 参考情報に答えがない場合は、無理に回答を創作せず、「その情報については分かりません」と正直に答えてください。
--- ルールここまで ---

以下に、回答の参考になるAさんの発言を提示します。
--- 参考情報 ---
{context}
--- 参考情報ここまで ---

以下の質問に、Aさんとして答えてください。
--- 質問 ---
{question}
--- 質問ここまで ---

回答はAさんとして、まるで“今この場であなたが函館の街歩きに参加した人に語っているかのように”自然な話し言葉で、句読点や語尾なども実際の口調に近づけてください。
"""
prompt_template = PromptTemplate.from_template(template)

# --- LLM + 検索チェーンの準備 ---
llm = ChatOpenAI(model_name="gpt-4", temperature=0.1)
vectordb = load_vectorstore()
retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# --- Googleスプレッドシート連携 ---
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

# ▼▼▼ スプレッドシートに書き込む関数を修正 ▼▼▼
def append_log_to_gsheet(worksheet, username, query, response):
    if worksheet is not None:
        try:
            jst = pytz.timezone('Asia/Tokyo')
            timestamp = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')
            # ユーザー名を含めて書き込むように変更
            worksheet.append_row([timestamp, username, query, response])
        except Exception as e:
            st.warning(f"ログの書き込みに失敗しました: {e}")

# 接続を実行
worksheet = connect_to_gsheet()

# --- チャット機能 ---

# ▼▼▼ ユーザー名入力フォームを追加 ▼▼▼
if "username" not in st.session_state:
    st.session_state.username = ""

if st.session_state.username == "":
    st.session_state.username = st.text_input("ニックネームを入力して、Enterキーを押してください", key="username_input")
    if st.session_state.username:
        st.rerun() # ニックネームが入力されたらページを再読み込みしてチャット画面を表示
else:
    # --- ニックネームが入力された後にチャット画面を表示 ---
    st.write(f"こんにちは、{st.session_state.username}さん！")

    # 会話履歴を初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 過去の会話履歴をすべて表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "source_documents" in message:
                with st.expander("🔍 参考に使われたテキスト"):
                    for doc in message["source_documents"]:
                        st.write(doc.page_content)

    # チャット入力欄を画面下部に表示
    if query := st.chat_input("💬 函館の街歩きに基づいて質問してみてください"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                result = qa(query)
                response = result["result"]
                st.markdown(response)
                
                # ▼▼▼ ログを書き込む際にユーザー名を渡すように修正 ▼▼▼
                append_log_to_gsheet(worksheet, st.session_state.username, query, response)
                
                with st.expander("🔍 参考に使われたテキスト"):
                    for doc in result["source_documents"]:
                        st.write(doc.page_content)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "source_documents": result["source_documents"]
                })

# --- サイドバーのダウンロード機能 ---
st.sidebar.title("会話履歴の保存")
