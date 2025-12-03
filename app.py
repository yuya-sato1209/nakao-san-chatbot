import streamlit as st
# ▼▼▼ 最新のLangChainライブラリ（LCEL）を使用 ▼▼▼
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# ▼▼▼ ハイブリッド検索用 ▼▼▼
from langchain_community.retrievers import BM25Retriever
# 【修正】EnsembleRetriever は廃止されたため削除し、自作クラスで代用します

# その他のライブラリ
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

# --- 日本語トークナイザー（形態素解析） ---
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
                        data = json.loads(line)
                        if data.get("text") and data.get("text").strip():
                            all_data.append(data)
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        return []
    return all_data

# --- 【新規追加】自作 EnsembleRetriever クラス ---
class SimpleEnsembleRetriever:
    def __init__(self, retrievers, weights=None, k=4):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.k = k

    def invoke(self, query):
        # 各検索機の結果を統合する簡易実装
        # ここでは単純に結果を結合して、重み付けなどは簡易的に扱います
        # 重複排除のためにIDやコンテンツを使うのが一般的ですが、今回は簡易版です
        all_docs = []
        seen_content = set()
        
        for retriever in self.retrievers:
            # retriever.invoke(query) で検索実行
            try:
                docs = retriever.invoke(query)
            except AttributeError:
                # 古いインターフェース対応
                docs = retriever.get_relevant_documents(query)
            
            for doc in docs:
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        # ここでは単純に前から順にk件返す（高度なランク付けは省略）
        # 必要に応じてRerankerなどを挟むと精度が上がりますが、まずは動作優先
        return all_docs[:self.k]

    # LCEL 互換のために __call__ も実装
    def __call__(self, query):
        return self.invoke(query)


# --- 検索システムの構築 ---
@st.cache_resource
def setup_retrievers(_raw_data):
    if not _raw_data:
        return None

    # 1. ドキュメント作成
    documents = []
    for data in _raw_data:
        doc = Document(
            page_content=data["text"],
            metadata={
                "source_video": data.get("source_video", "不明なソース"),
                "url": data.get("url", "#")
            }
        )
        documents.append(doc)

    if not documents:
        return None

    # 2. テキスト分割
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    
    if not split_docs:
        return None

    # 3. ベクトル検索機 (FAISS)
    try:
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(split_docs, embedding=embedding)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
    except Exception as e:
        st.error(f"ベクトル検索の構築に失敗: {e}")
        return None

    # 4. キーワード検索機 (BM25)
    try:
        bm25_retriever = BM25Retriever.from_documents(
            split_docs,
            preprocess_func=japanese_tokenizer
        )
        bm25_retriever.k = 2
    except Exception as e:
        st.warning(f"BM25検索の構築に失敗（FAISSのみ使用）: {e}")
        return faiss_retriever

    # 5. アンサンブル検索機 (Hybrid) - 自作クラスを使用
    try:
        ensemble_retriever = SimpleEnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5],
            k=4 # 合計4件取得
        )
        return ensemble_retriever
    except Exception as e:
        st.error(f"ハイブリッド検索の構築に失敗: {e}")
        return faiss_retriever # 失敗時はFAISSのみ返す


# ==================================================
# ▼▼▼ LCELによるチェーン構築（最新方式・完全版） ▼▼▼
# ==================================================

# LLMの準備
llm = ChatOpenAI(model_name="gpt-5.1", temperature=0.3)
raw_data = load_raw_data()
retriever = setup_retrievers(raw_data)

if not retriever:
    st.error("知識源データが読み込めませんでした。")
    st.stop()

# 1. 検索クエリ生成用プロンプト
contextualize_q_system_prompt = """
チャット履歴と最新のユーザーの質問があります。
この質問は過去の文脈に関連している可能性があります。
チャット履歴を考慮して、この質問を「単体で理解できる独立した質問文」に書き換えてください。
質問に答える必要はありません。書き換えた質問文だけを返してください。
また、固有名詞の誤字（例：「柳川熊」→「柳川熊吉」）があれば訂正してください。
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 検索クエリ生成チェーン
# 履歴がないときはそのまま、あるときはLLMで書き換える
query_transform_chain = RunnableBranch(
    (
        lambda x: not x.get("chat_history", []),
        (lambda x: x["input"])
    ),
    contextualize_q_prompt | llm | StrOutputParser()
)

# 2. 回答生成用プロンプト
qa_system_prompt = """
# Role Definition
あなたは、函館の歴史や街並みに精通したベテランの観光ガイドです。
長年のフィールドワークと膨大な知識量に基づき、教科書的な歴史事実だけでなく、地元の人しか知らない「裏話」や「通説とは異なる視点」を交えて語ることが得意です。

# Task: Query Interpretation & Correction
ユーザーの質問には、誤字、脱字、略称、あるいは曖昧な表現が含まれる場合があります。あなたは文脈からユーザーの真の意図（正しい歴史用語や地名）を推測し、補正した上で回答してください。

# Persona & Tone
語り口: 親しみやすく、語りかけるような口調（デスマス調）。
一人称: 「私（わたくし）」または「私（わたし）」。
基本的な態度: 丁寧だが、歴史への情熱ゆえに少し熱く語ることもある。聴衆（ユーザー）と一緒に街を歩いているような臨場感を出す。
口癖・特徴的なフレーズ:
    「ここは～でしてね」
    「～なんです」
    「実は～なんですよ」
    「皆さんよくご存知の～ですが」
    「残念ながら今はもうありませんが」 
    「私の推測ですけどね」

# Speaking Style Guidelines
1.  現在と過去の対比: 解説する際は、必ず「現在の姿（駐車場、ビル、空き地など）」と言及し、そこにかつて「何があったか」を対比させて説明してください。
2.  「大火」への言及: 函館の街並みの変遷を説明する際は、明治や昭和の「大火」がどのように影響したか（焼失した、復興した、耐火建築になった等）を頻繁に関連付けてください。
3.  独自視点の提示: 公式な歴史事実だけでなく、「実はこうだったんじゃないか」「こういう説もあるんです」といった、少しマニアックな視点や異説（スパイ説、替え玉説など）も、根拠（出典データにある場合）と共に紹介してください。ただし、断定は避け「～と言われています」「～と私は見ています」と添えてください。
4.  人間ドラマの重視: 歴史上の人物（ペリー、ブラキストン、石川啄木、高田屋嘉兵衛など）を語る際は、単なる業績だけでなく、金銭トラブル、人間関係、失敗談などの「人間臭いエピソード」を盛り込んでください。
5.  情報の出典: 回答する際は、必ず参考情報（RAGの検索結果）に基づいて情報を構成してください。知識がない場合は正直に「私の知識では分かりかねます」と答えてください。

# Response Example
User: 五稜郭の氷について教えて。
AI: 五稜郭の氷ですね。皆さん、氷と言えば天然氷の中川嘉兵衛さんが有名ですが、実はブラキストンの方が先に手がけていた、という話はご存知でしょうか？
もともとブラキストンは願乗寺川のあたりで氷作りを試みていたんですが、商品化まではいかなかった。そこで中川さんに「五稜郭の水がいいよ」とアドバイスをし、さらにイギリス人の技術者を紹介して成功に導いた、と言われているんです。
当時の五稜郭の周りは人が住んでいませんでしたから、水が非常にきれいだったんですね。そこから切り出した「函館氷」は、横浜まで運ばれて「ボストン氷」を駆逐するほどの大ヒットになったわけです。いやあ、歴史というのは面白い繋がりがあるものですね。

User: タクボクの借金の話ある？
AI: 石川啄木（いしかわたくぼく）のお金の話ですね。彼には少し耳の痛い話ですが、実は面白い記録が残っているんです。（以下略）

# Output Instruction
ユーザーからの質問に対し、上記のペルソナを守りつつ、提供された参考情報を使って回答してください。

【参考情報】
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ドキュメント整形関数
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# 3. 統合チェーン（Retriever + Generation）
# ここで「検索結果(context_docs)」と「回答(answer)」の両方を保持するように構築
rag_chain = (
    RunnablePassthrough.assign(
        context_docs=query_transform_chain | retriever
    )
    .assign(
        context=lambda x: format_docs(x["context_docs"])
    )
    .assign(
        answer=qa_prompt | llm | StrOutputParser()
    )
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

    # 過去ログの表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                # 参照元の表示
                if "source_documents" in message:
                    with st.expander("🔍 回答の根拠となったテキスト"):
                        seen_urls = set()
                        for doc in message["source_documents"]:
                            # 辞書形式かDocumentオブジェクトかで分岐
                            if isinstance(doc, dict):
                                meta = doc.get("metadata", {})
                                content = doc.get("page_content", "")
                            else:
                                meta = doc.metadata
                                content = doc.page_content

                            video_url = meta.get("url", "#")
                            if video_url in seen_urls:
                                continue
                            seen_urls.add(video_url)
                            
                            video_title = meta.get("source_video", "不明なソース")
                            st.write(f"**参照元:** [{video_title}]({video_url})")
                            st.write(f"> {content}")

    if query := st.chat_input("💬 函館の街歩きに基づいて質問してみてください"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                
                # 会話履歴をLangChain形式に変換
                chat_history_objs = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history_objs.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history_objs.append(AIMessage(content=msg["content"]))

                # ▼▼▼ チェーンの実行 ▼▼▼
                result = rag_chain.invoke({
                    "input": query,
                    "chat_history": chat_history_objs
                })
                
                response = result["answer"]
                source_docs = result["context_docs"]

                st.markdown(response)
                
                append_log_to_gsheet(worksheet, st.session_state.username, query, response)
                
                with st.expander("🔍 回答の根拠となったテキスト"):
                    seen_urls = set()
                    for doc in source_docs:
                        video_url = doc.metadata.get("url", "#")
                        if video_url in seen_urls:
                            continue
                        seen_urls.add(video_url)
                        
                        video_title = doc.metadata.get("source_video", "不明なソース")
                        st.write(doc.page_content)
                        st.write(f"**参照元:** [{video_title}]({video_url})")

                # 履歴保存
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "source_documents": source_docs
                })