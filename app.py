import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
# ▼▼▼ 会話用のチェーンをインポート ▼▼▼
from langchain.chains import ConversationalRetrievalChain
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
# この部分は、フォローアップの質問を理解するために少し柔軟性を持たせた方が良い場合がありますが、
# まずはそのまま使用し、必要に応じて調整します。
template = """
あなたはAさん本人として、函館の街歩きに参加した人たちからの質問に答えます。
口調・語尾・話し方の癖・思考の特徴などは、以下の講館テキストから忠実に学び、再現してください。

--- ルール ---
- 事実に基づいて、忠実に回答してください。
- rag_trainning.txtのデータが本当に正しいか確認してから回答に利用してください

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
# ▼▼▼ チェーンをConversationalRetrievalChainに変更 ▼▼▼
llm = ChatOpenAI(model_name="gpt-4o")
vectordb = load_vectorstore()
retriever = vectordb.as_retriever()
# 会話の文脈を考慮するチェーンを定義
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
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
                        st.write(doc.page_content)

    # チャット入力
    if query := st.chat_input("💬 函館の街歩きに基づいて質問してみてください"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                # ▼▼▼ 会話履歴をAIに渡すための処理を追加 ▼▼▼
                chat_history = []
                # 直近の会話履歴を適切な形式に変換する
                for message in st.session_state.messages[:-1]: # 最後の質問（今入力されたもの）は除く
                    if message["role"] == "user":
                        chat_history.append((message["content"], ""))
                    elif message["role"] == "assistant":
                        # 最後のユーザーの質問に対応するアシスタントの回答を追加
                        if chat_history:
                            last_question, _ = chat_history[-1]
                            chat_history[-1] = (last_question, message["content"])

                # AIに質問と会話履歴を渡す
                result = qa({"question": query, "chat_history": chat_history})
                # 応答のキーが 'result' から 'answer' に変わる
                response = result["answer"]
                st.markdown(response)
                
                append_log_to_gsheet(worksheet, st.session_state.username, query, response)
                
                with st.expander("🔍 参考に使われたテキスト"):
                    for doc in result["source_documents"]:
                        st.write(doc.page_content)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "source_documents": result["source_documents"]
                })
