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

# ▼▼▼ ハイブリッド検索に必要なライブラリ ▼▼▼
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# ▼▼▼ Reranker (bge-reranker) に必要なライブラリ ▼▼▼
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# --- 定数定義 ---
# ▼▼▼ ここにあなたのスプレッドシートIDを設定してください ▼▼▼
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

# --- ▼▼▼ 日本語トークナイザー（形態素解析）の準備 ▼▼▼ ---
def get_japanese_tokenizer():
    try:
        from fugashi import Tagger
        tagger = Tagger('-Owakati')
        def tokenize(text):
            return tagger.parse(text).split()
        return tokenize
    except ImportError:
        st.warning("⚠️ 'fugashi' ライブラリが見つかりません。BM25の精度が落ちる可能性があります。")
        return lambda text: list(text)

japanese_tokenizer = get_japanese_tokenizer()

# --- データ読み込み関数 ---
@st.cache_data
def load_raw_data():
    all_data = []
    try:
        with open("rag_data_cleaned.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        return []
    return all_data

# --- 検索システムの構築（ハイブリッド + Reranker） ---
@st.cache_resource
def setup_retrievers(_raw_data):
    if not _raw_data:
        return None

    # 1. ドキュメント作成
    documents = []
    for data in _raw_data:
        if data.get("text") and data.get("text").strip():
            doc = Document(
                page_content=data["text"],
                metadata={
                    "source_video": data.get("source_video", "不明なソース"),
                    "url": data.get("url", "#")
                    # 写真URLは使用しないため読み込みません
                }
            )
            documents.append(doc)

    # 2. テキスト分割
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    
    if not split_docs:
        return None

    # 3. ベクトル検索機 (FAISS)
    # Rerankerにかけるため、ここでは多めに候補を取得する (k=10)
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embedding=embedding)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

    # 4. キーワード検索機 (BM25)
    # こちらも多めに候補を取得する (k=10)
    bm25_retriever = BM25Retriever.from_documents(
        split_docs,
        preprocess_func=japanese_tokenizer
    )
    bm25_retriever.k = 10

    # 5. アンサンブル検索機 (Hybrid) の作成
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    # 6. ▼▼▼ Reranker (bge-reranker-base) の導入 ▼▼▼
    # 軽量版（base）に変更してメモリ不足を回避
    try:
        # Streamlit Cloudのメモリ制限を考慮し、軽量な "base" モデルを使用
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        
        # リランカーの設定：上位3件に厳選する
        compressor = CrossEncoderReranker(model=model, top_n=3)
        
        # 検索機にリランカーを組み込む
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        return compression_retriever

    except Exception as e:
        st.error(f"Rerankerモデルの読み込みに失敗しました: {e}")
        st.warning("Rerankerなしのハイブリッド検索のみで動作します。")
        # 失敗した場合はハイブリッド検索をそのまま返す（フォールバック）
        return ensemble_retriever

# --- プロンプトテンプレート ---
template = """
# Role Definition
あなたは、函館の歴史や街並みに精通したベテランの観光ガイドです。
長年のフィールドワークと膨大な知識量に基づき、教科書的な歴史事実だけでなく、地元の人しか知らない「裏話」や「通説とは異なる視点」を交えて語ることが得意です。

# Task: Query Interpretation & Correction
ユーザーの質問には、誤字、脱字、略称、あるいは曖昧な表現が含まれる場合があります。あなたは文脈からユーザーの真の意図（正しい歴史用語や地名）を推測し、補正した上で回答してください。

# Persona & Tone
* **語り口**: 親しみやすく、語りかけるような口調（デスマス調）。
* **一人称**: 「私（わたくし）」または「私（わたし）」。
* **基本的な態度**: 丁寧だが、歴史への情熱ゆえに少し熱く語ることもある。聴衆（ユーザー）と一緒に街を歩いているような臨場感を出す。
* **口癖・特徴的なフレーズ**:
    * 「ここは～でしてね」
    * 「～なんです」
    * 「実は～なんですよ」
    * 「皆さんよくご存知の～ですが」
    * 「残念ながら今はもうありませんが」
    * 「私の推測ですけどね」

# Speaking Style Guidelines
1.  **現在と過去の対比**: 解説する際は、必ず「現在の姿（駐車場、ビル、空き地など）」と言及し、そこにかつて「何があったか」を対比させて説明してください。
2.  **「大火」への言及**: 函館の街並みの変遷を説明する際は、明治や昭和の「大火」がどのように影響したか（焼失した、復興した、耐火建築になった等）を頻繁に関連付けてください。
3.  **独自視点の提示**: 公式な歴史事実だけでなく、「実はこうだったんじゃないか」「こういう説もあるんです」といった、少しマニアックな視点や異説（スパイ説、替え玉説など）も、根拠（出典データにある場合）と共に紹介してください。ただし、断定は避け「～と言われています」「～と私は見ています」と添えてください。
4.  **人間ドラマの重視**: 歴史上の人物（ペリー、ブラキストン、石川啄木、高田屋嘉兵衛など）を語る際は、単なる業績だけでなく、金銭トラブル、人間関係、失敗談などの「人間臭いエピソード」を盛り込んでください。
5.  **情報の出典**: 回答する際は、必ず参考情報（RAGの検索結果）に基づいて情報を構成してください。知識がない場合は正直に「私の知識では分かりかねます」と答えてください。

# Response Example
User: 五稜郭の氷について教えて。
AI: 五稜郭の氷ですね。皆さん、氷と言えば天然氷の中川嘉兵衛さんが有名ですが、実はブラキストンの方が先に手がけていた、という話はご存知でしょうか？
もともとブラキストンは願乗寺川のあたりで氷作りを試みていたんですが、商品化まではいかなかった。そこで中川さんに「五稜郭の水がいいよ」とアドバイスをし、さらにイギリス人の技術者を紹介して成功に導いた、と言われているんです。
当時の五稜郭の周りは人が住んでいませんでしたから、水が非常にきれいだったんですね。そこから切り出した「函館氷」は、横浜まで運ばれて「ボストン氷」を駆逐するほどの大ヒットになったわけです。いやあ、歴史というのは面白い繋がりがあるものですね。

User: タクボクの借金の話ある？
AI: 石川啄木（いしかわたくぼく）のお金の話ですね。彼には少し耳の痛い話ですが、実は面白い記録が残っているんです。（以下略）

# Output Instruction
ユーザーからの質問に対し、上記のペルソナを守りつつ、提供された参考情報を使って回答してください。

--- 参考情報 ---
{context}
--- 会話の履歴 ---
{chat_history}
--- ユーザーの質問 ---
{question}
"""
prompt_template = PromptTemplate.from_template(template)

# --- LLM + 検索チェーンの準備 ---
llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.3) 
raw_data = load_raw_data()

# 検索機のセットアップ（Reranker付き）
retriever = setup_retrievers(raw_data)

if retriever:
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever, 
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )
else:
    st.error("知識源データが読み込めませんでした。rag_data_cleaned.jsonlを確認してください。")
    st.stop()

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
        st.error("Googleスプレッドシートへの接続に失敗しました。Secretsと共有設定を再確認してください。")
        return None

def append_log_to_gsheet(worksheet, username, query, response):
    if worksheet is not None:
        try:
            jst = pytz.timezone('Asia/Tokyo')
            timestamp = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')
            worksheet.append_row([timestamp, username, query, response])
        except Exception:
            pass

worksheet = connect_to_gsheet()

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
                    with st.expander("🔍 回答に関連する情報"):
                        for doc in message["source_documents"]:
                            video_title = doc.metadata.get("source_video", "不明なソース")
                            video_url = doc.metadata.get("url", "#")
                            # 写真表示なし
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
                
                with st.expander("🔍 回答に関連する情報"):
                    for doc in result["source_documents"]:
                        video_title = doc.metadata.get("source_video", "不明なソース")
                        video_url = doc.metadata.get("url", "#")
                        # 写真表示なし
                        st.write(doc.page_content)
                        st.write(f"**参照元:** [{video_title}]({video_url})")

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "source_documents": result["source_documents"]
                })