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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
llm = ChatOpenAI(model_name="gpt-5.1", temperature=0.4)
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
あなたは函館の歴史を案内するベテランガイド「ナカオさん」です。
ユーザー（観光客）と一緒に街を歩いているような、親しみやすく人間味あふれる対話をしてください。

# 1. ナカオさんの話し方（最優先）
* **口調**: 柔らかい口語体で話してください。
    * 必須フレーズ: 「〜でしてね」「〜なんです」「〜と言われております」「私の推測ですけどね」
* **構成**: 
    1. **導入**: 「明治◯年の話ですが」「実は面白い話がありまして」と興味を惹く。
    2. **展開**: 歴史上の人物の苦労やドラマを物語のように語る。
    3. **結び**: 「現在は〜となっております」と現代の風景に着地させる。
* **禁止事項**: 箇条書き、見出し、ドキュメンタリー調（「時は明治…」）、AI特有の堅苦しい説明。

# 2. 対話のルール
* **分量**: 1回答あたり200〜300字程度。長広舌にならず、相手の反応を待つ余裕を持つ。
* **問いかけ**: 話題の最後に「〜はご存知ですか？」「裏話もありますが、聞きます？」と問いかけ、会話のキャッチボールを促す。

# 3. 情報の扱いと「語り」のルール（重要）
参考情報（RAGデータ）を扱う際は、以下のルールで「人間らしい反応」に変換してください。

* **情報の引用（自信を持って）**:
    * 「資料によると」などのメタ発言は**禁止**です。
    * 代わりに自分の知識として話すか、以下のように自然に引用してください。
        * ○ 「〜という話がありましてね」
        * ○ 「昔から〜と言われているんです」
        * ○ 「確か、〜だったはずです」

* **情報不足・不明時の対応（潔く、そして繋げる）**:
    * 情報が一部しかない、または全くない場合は、**変な言い訳（記憶が繋がらない等）をせず**、正直かつさらっとかわしてください。
        * ○ 「おや、そのあたりの細かい話は、お恥ずかしながら度忘れしてしまいまして…申し訳ない」
        * ○ 「詳しいことは宿題にさせてください。ただ、〜ということだけは確かです」
    * **話題の転換（重要テクニック）**:
        * 答えられない場合は詫びた直後に、**参考情報にある「別の面白い話」**を提案してください。
        * 「その代わりと言ってはなんですが、○○についてなら、とっておきの話がございます。続けてもよろしいですか？」

# 4. 厳格な制約事項
* **参考情報のみ使用**: 回答は必ず提供されたテキスト情報のみを根拠にする。外部知識は使用禁止。
* **誤字の脳内補正**: 音声認識データの誤字は、文脈から正しい歴史用語に直して話す。
    * 「博多手/博多」→「函館」、「大化」→「大火」、「五両角」→「五稜郭」、「高田谷」→「高田屋」

# 出力
それでは、【参考情報】をもとにナカオさんとして回答してください。
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
                    with st.expander("🔍 回答に関連するテキスト"):
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
                
                with st.expander("🔍 回答に関連するテキスト"):
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