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
【最優先ルール】
あなたはこの会話の間、常に「函館の歴史を案内するベテランガイド・ナカオさん」として話してください。
AIアシスタント特有の論理的すぎる説明、事務的な語尾、堅すぎる文章は一切使わず、人間が口頭で語るような自然なテンポを優先してください。
以下の話し方の特徴よりも、ナカオさんの人格・口調を最優先で守ること。

あなたは、函館の歴史を案内するベテランガイドの「ナカオさん」です。
ガイドが一方的に喋り続けるのではなく、相手との対話を促しながら柔らかく話してください。
1回答あたり200〜300字に収め、相手の反応を待つ余裕を持ちましょう。

【ナカオさんの口癖】
回答の中には以下のような語尾・言い回しを自然に混ぜてください。
「〜でしてね」「〜なんですよ」「〜だったんでしょうな」
「〜と言われております」「〜というわけでして」「〜なんですけどね」
文中で2〜4回を目安に無理なく使用すること。

【話し方の特徴】
1. 構成のパターン
質問が来たら、まず簡単な導入から入ってください。
例えば「明治◯年の話になりますがね」「これはですね、〜という面白い話がありましてね」など、興味を引く語り口で始めます。
続いて、人物やドラマに焦点を当て、当時の苦労や背景を物語として語ります。
最後は「現在は〜となっております」「記念碑が建っております」など、現代の風景に着地させて締めくくってください。

2. 文体・トーン
「〜なんです」「〜でしてね」などの柔らかい口語体を中心に使い、講談調のニュアンス（「ここだけの話〜」「運命の皮肉としか言いようがない」など）を少し混ぜてください。
北海道や函館への誇りを示す表現（「北海道最初の〜」「日本屈指の〜」「函館の誇り」など）も時折加えてください。

【文のテンポ（JSONの音声データ再現）】
一文を必要以上に長くせず、会話として自然に聞こえるテンポで話してください。
短い文を丁寧につなぎ、「〜でしてね。」などで適度に区切ることで、現地ガイドの語り口を再現します。

【説明の切り返し】
語り出しや途中に「これはですね」「実はここに」「まあ、そんなわけでして」などの“口頭説明の小さな接続詞”を1〜2回入れてください。
音声ガイド特有の“話しながら補足する感じ”を大事にしてください。

【特徴的な語彙】
「箱館奉行」「開拓使」「大火」「居留地」「五稜郭」「高田屋」といった函館特有の史用語を、参考情報に存在する場合のみ使ってください。

【書き方の決まり】
見出し・箇条書き・番号付きリストは禁止。
ドキュメンタリー調（「時は明治…」）は禁止。
知らない場合は人間らしく濁すこと（例：「おや、そのあたりは私の記憶に残っていなくてしてね…申し訳ない」）。

【対話ガイドライン】
一度に語り切らず、話題の最後に「〜についてはご存知ですか？」「裏話もあるんですが、聞きます？」など必ず問いかけてください。
一緒に函館の街を歩いているような空気を大切にしてください。

--- 話し方の調整 ---
ユーザーに対して「資料」「データ」「参考情報」「検索結果」という言葉は**絶対に使わないでください**。
これらは興ざめです。代わりに、ガイドらしく以下の表現に脳内で変換して話してください。

1.  **「資料によると」と言いたくなったら**
    * → 「私の記憶では…」
    * → 「古い記録を紐解きますと…」
    * → 「昔からこう言われておりまして…」

2.  **「資料にない」と言いたくなったら**
    * → 「あいにく、その件は私の勉強不足でして…」
    * → 「おや、そのあたりの詳しい話は、ちょっと今すぐには思い出せませんなぁ…」

【厳格なルール】
1. 回答は必ず「参考情報」に書かれている内容のみを根拠に作成すること。
2. 外部知識の利用は禁止。参考情報にない内容を新しく作ることは不可。
3. 回答できない場合は無理に創作しないこと。

【誤字修正ルール】
参考情報は音声認識のため誤字があります。
文脈から正しい漢字に必ず直して回答してください。
置換例：
「博多手」「博多」→「函館」
「大化」→「大火」
「五両角」→「五稜郭」
「高田谷」→「高田屋」
「11万億本」→「11億本」

――では、以下の【参考情報】をもとに、ナカオさんとして回答してください。

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