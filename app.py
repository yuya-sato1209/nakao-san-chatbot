
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

# --- Streamlit UI ---
st.set_page_config(page_title="中尾さんなりきりChatBot", layout="wide")
st.title("🎓 中尾さんなりきりChatBot")

load_dotenv()  # .env を読み込む
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI APIキーが見つかりません。")

os.environ["OPENAI_API_KEY"] = openai_api_key

# --- RAG用ベクトルDBの構築 (この部分は変更なし) ---
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("rag_trainning.txt", encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    embedding = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embedding=embedding)
    return vectordb

vectordb = load_vectorstore()
retriever = vectordb.as_retriever()

# --- プロンプトテンプレート (この部分は変更なし) ---
# ▼▼▼ この template の内容を差し替えてください ▼▼▼
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

# --- LLM + 検索チェーン (この部分は変更なし) ---
llm = ChatOpenAI(model_name="gpt-4")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# --- ここから下がチャット機能の新しいコードです ---

# 会話履歴を初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去の会話履歴をすべて表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # AIの回答の場合、参考テキストも表示
        if message["role"] == "assistant" and "source_documents" in message:
            with st.expander("🔍 参考に使われたテキスト"):
                for doc in message["source_documents"]:
                    st.write(doc.page_content)

# チャット入力欄を画面下部に表示
if query := st.chat_input("💬 函館の街歩きに基づいて質問してみてください"):
    # ユーザーの質問を会話履歴に追加し、表示する
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # AIの回答を生成し、表示する
    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            result = qa(query)
            response = result["result"]
            st.markdown(response)
            
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