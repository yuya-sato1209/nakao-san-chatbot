import streamlit as st # type: ignore
from langchain_community.chat_models import ChatOpenAI # type: ignore
from langchain_community.document_loaders import TextLoader # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_community.embeddings import OpenAIEmbeddings # type: ignore
from langchain.chains import RetrievalQA # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain.prompts import PromptTemplate # type: ignore
from dotenv import load_dotenv # type: ignore
import os
import gspread # type: ignore
from google.oauth2.service_account import Credentials # type: ignore
from datetime import datetime
import pytz # type: ignore

# --- Streamlit UI設定 ---
st.set_page_config(page_title="中尾さんなりきりChatBot", layout="wide")
st.title("🎓 中尾さんなりきりChatBot")

# --- APIキーの読み込み ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    # ローカルの.envファイルから読み込めない場合、StreamlitのSecretsを試す
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("OpenAI APIキーが見つかりません。.envファイルまたはStreamlitのSecretsに設定してください。")
        st.stop()
else:
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
口調・語尾・話し方の癖・思考の特徴などは、以下の講館テキストから忠実に学び、再現してください。

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
llm = ChatOpenAI(model_name="gpt-4")
vectordb = load_vectorstore()
retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# --- Googleスプレッドシート連携 ---

# Googleスプレッドシートへの接続（初回実行時のみキャッシュを利用）
@st.cache_resource
def connect_to_gsheet():
    try:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"])
        scoped_creds = creds.with_scopes([
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ])
        client = gspread.authorize(scoped_creds)
        spreadsheet = client.open("中尾さんChatBot会話ログ")
        worksheet = spreadsheet.worksheet("log")
        return worksheet
    except Exception as e:
        st.error(f"Googleスプレッドシートへの接続に失敗しました: {e}")
        return None

# スプレッドシートに会話ログを追記する関数
def append_log_to_gsheet(worksheet, query, response):
    if worksheet is not None:
        try:
            jst = pytz.timezone('Asia/Tokyo')
            timestamp = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')
            worksheet.append_row([timestamp, query, response])
        except Exception as e:
            st.warning(f"ログの書き込みに失敗しました: {e}")

# 接続を実行
worksheet = connect_to_gsheet()

# --- チャット機能 ---

# 会話履歴をセッション状態で初期化
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
    # ユーザーの質問を会話履歴に追加し、表示
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # AIの回答を生成し、表示
    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            result = qa(query)
            response = result["result"]
            st.markdown(response)
            
            # ログをスプレッドシートに書き込む
            append_log_to_gsheet(worksheet, query, response)
            
            # 参考テキストを折りたたみで表示
            with st.expander("🔍 参考に使われたテキスト"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content)

            # AIの回答と参考テキストを会話履歴に追加
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "source_documents": result["source_documents"]
            })

# --- サイドバーのダウンロード機能 ---

st.sidebar.title("会話履歴の保存")

def format_log(messages):
    log_data = ""
    for message in messages:
        role = "あなた" if message["role"] == "user" else "中尾さん"
        log_data += f"[{role}]\n{message['content']}\n\n"
    return log_data.encode('utf-8')

if st.session_state.messages:
    log_bytes = format_log(st.session_state.messages)
    st.sidebar.download_button(
        label="会話履歴をダウンロード",
        data=log_bytes,
        file_name="nakao_san_chat_log.txt",
        mime="text/plain"
    )
else:
    st.sidebar.info("まだ会話がありません。")