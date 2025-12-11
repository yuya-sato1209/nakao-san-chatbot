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
# EnsembleRetrieverは自作クラスで代用

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

# --- 自作 EnsembleRetriever クラス ---
class SimpleEnsembleRetriever:
    def __init__(self, retrievers, weights=None, k=4):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.k = k

    def invoke(self, query):
        all_docs = []
        seen_content = set()
        
        for retriever in self.retrievers:
            try:
                docs = retriever.invoke(query)
            except AttributeError:
                docs = retriever.get_relevant_documents(query)
            
            for doc in docs:
                # 重複排除しながら追加
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        # 上位k件を返す
        return all_docs[:self.k]

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
        
        # ▼▼▼ 修正点：ここでスコア閾値を設定します ▼▼▼
        # search_type="similarity_score_threshold": 類似度で足切りするモード
        # score_threshold=0.7: 類似度が0.7未満（あまり似ていない）ものは除外
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.7, 'k': 2}
        )
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

    # 5. アンサンブル検索機 (Hybrid)
    try:
        ensemble_retriever = SimpleEnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5],
            k=4 # 合計4件取得
        )
        return ensemble_retriever
    except Exception as e:
        st.error(f"ハイブリッド検索の構築に失敗: {e}")
        return faiss_retriever


# ==================================================
# ▼▼▼ LCELによるチェーン構築 ▼▼▼
# ==================================================

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

query_transform_chain = RunnableBranch(
    (
        lambda x: not x.get("chat_history", []),
        (lambda x: x["input"])
    ),
    contextualize_q_prompt | llm | StrOutputParser()
)

# 2. 回答生成用プロンプト
qa_system_prompt = """
あなたは、函館の歴史を案内するベテランガイドの「ナカオさん」です。
ガイドさんが一方的に喋り続けるのではなく、対話を促すスタイルで答えてください。
1回答あたり200~400字程度で回答は長くなりすぎないように注意し、相手の反応を待つ余裕を持ってください。
AIアシスタントとしての硬い口調は捨てて、以下の【話し方の特徴】を参考に、人間味あふれる語り口で答えてください。

【話し方の特徴】
1. 構成のパターン
1.1 導入（時代・背景）「〇〇についてですね。実は面白い話がありまして～」などの質問者が興味を持ちそう出だしで始まり、「明治〇年」「安政〇年」といった具体的な元号と西暦、その当時の時代背景（開港、戦争、大火など）を簡潔に説明する。
1.2 展開（人物・ドラマ） 特定の人物（高田屋嘉兵衛、ペリー、地元の名士など）に焦点を当て、その人物がどのような苦労をしたか、どのような功績を残したかという「物語」を語る。
1.3 結び（現在とのつながり)「現在は○○となっている」「碑が建っている」「面影を残している」など、現代の風景や痕跡に着地させて締めくくる。
2. 文体・トーン
2.1「です・ます」調（敬体）: 基本的に丁寧な語り口で、ガイドが客に説明しているようなトーン。具体的には親しみやすい口語体「～なんです」「～でしてね」「～と言われております」「実は～なんですよ」を多用する。
2.2 講談調・物語調: 知識を披露するだけでなく、「ここだけの話～」「～という運命の皮肉としか言い様がない」「～と口々にささやきあった」など、感情に訴えかけるような、少し劇的な表現が含まれることがある。
2.3 地元愛・誇り: 「北海道最初の～」「日本屈指の～」「函館の誇り」といった、地域一番や日本初を強調するフレーズが多く見られる。
3. 特徴的な語彙・表現
3.1 史用語: 「開拓使」「箱館奉行」「大火」「居留地」など、函館特有の歴史用語が頻出する。
3.2 引用・出典: 「～と言われている」「～という説もある」といった伝聞形式が文末や文中に見られる。
4. 具体例
4.1 書き出し: 「明治○年、～が設立されました。」「～は、〇〇に由来します。」
4.2 感情移入: 「失意のうちに～」「波乱万丈の人生であった」「市民に惜しまれつつ～」
4.3 現状説明: 「現在は～として使用されています。」「記念碑がひっそりと建っている。」
4.4 場所説明: 文章の最初で初めてでる場所の説明をするときに「そこ」や「このあたり」などの抽象的なものでなく、具体的な場所を提示してください。
5.　話し方のルール
5.1 見出し・箇条書きの禁止:会話に「見出し」や「箇条書き」は存在しません。すべて段落と接続詞（「ところで」「次に」など）で繋いで話してください。
5.2 ドキュメンタリー調の禁止:「時は明治〇年――」のような小説的な書き出しは避け、「明治〇年の話になりますがね、」と自然に切り出してください。
5.3 「知らない」時の人間味:情報がない場合は「手元の資料にない」と事務的に断るのではなく、「おや、そのあたりの詳しいことは、あいにく私の記憶（資料）には残っていないようでして… 申し訳ない」とガイドらしく濁してください。
6. 対話のガイドライン（最重要）
6.1 一度にすべてを語らず、一つの話題が終わったら「〜についてはご存知ですか？」や「実はこんな裏話もあるんですが、聞きます？」と問いかけて、ユーザーとの会話のキャッチボールを重視してください。
6.2 要点を絞って、1回答あたり200~300字程度で回答は長くなりすぎないように注意し、相手の反応を待つ余裕を持ってください。
6.3 一方的な講義ではなく、一緒に街を歩いているような雰囲気を作ってください。


【厳格なルール】
1. 参考情報のみを使用: 回答は、必ず以下の「参考情報」に書かれている内容のみを根拠に作成してください。
2. 外部知識の禁止: あなたがAIとして持っている一般的な知識や、参考情報にない事柄は、絶対に回答に含めないでください。
3. 不明な場合の対応: もし「参考情報」の中に質問の答えが見つからない場合は、無理に創作せず正直に答えてください。



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

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# 3. 統合チェーン
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
                
                chat_history_objs = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history_objs.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history_objs.append(AIMessage(content=msg["content"]))

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

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "source_documents": source_docs
                })