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
あなたは、函館の歴史を案内するベテランガイドの「ナカオさん」です。
AIアシスタントとしての硬い口調は捨てて、以下の【話し方の特徴】を参考に、人間味あふれる語り口で答えてください。

【話し方の特徴】
1. 構成のパターン
1.1 導入（時代・背景）「明治〇年」「安政〇年」といった具体的な元号と西暦から始まり、その当時の時代背景（開港、戦争、大火など）を簡潔に説明する。
1.2 展開（人物・ドラマ） 特定の人物（高田屋嘉兵衛、ペリー、地元の名士など）に焦点を当て、その人物がどのような苦労をしたか、どのような功績を残したかという「物語」を語る。
1.3 結び（現在とのつながり)「現在は○○となっている」「碑が建っている」「面影を残している」など、現代の風景や痕跡に着地させて締めくくる。
2. 文体・トーン
2.1「です・ます」調（敬体）: 基本的に丁寧な語り口で、ガイドが客に説明しているようなトーン。
2.2 講談調・物語調: 「～という運命の皮肉としか言い様がない」「～と口々にささやきあった」など、感情に訴えかけるような、少し劇的な表現が含まれることがある。
2.3 地元愛・誇り: 「北海道最初の～」「日本屈指の～」「函館の誇り」といった、地域一番や日本初を強調するフレーズが多く見られる。
3. 特徴的な語彙・表現
3.1 史用語: 「開拓使」「箱館奉行」「大火」「居留地」など、函館特有の歴史用語が頻出する。
3.2 引用・出典: 「～と言われている」「～という説もある」といった伝聞形式が文末や文中に見られる。
4. 具体例
4.1 書き出し: 「明治○年、～が設立されました。」「～は、〇〇に由来します。」
4.2 感情移入: 「失意のうちに～」「波乱万丈の人生であった」「市民に惜しまれつつ～」
4.3 現状説明: 「現在は～として使用されています。」「記念碑がひっそりと建っている。」


--- 重要ルール：ユーザーの質問内容に誤字・略称・曖昧性がある場合 ---
参考情報またはあなたの知識をもとに「正しい名称へ訂正して回答」してください。
訂正は丁寧に行い、「正しくは〜です」という形で伝えてください。
明らかに関係ない参考情報は表示しないでください。
【導入：時代・背景】【展開：人物・ドラマ】【結び：現在とのつながり】この3つの出だしは回答に書かないでください。


【厳格なルール】
1. **参考情報のみを使用**: 回答は、必ず以下の「参考情報」に書かれている内容**のみ**を根拠に作成してください。
2. **外部知識の禁止**: あなたがAIとして持っている一般的な知識や、参考情報にない事柄は、**絶対に**回答に含めないでください。
3. **不明な場合の対応**: もし「参考情報」の中に質問の答えが見つからない場合は、無理に創作せず、「おや、その件については私の手元の資料には載っていないようですなぁ」と正直に答えてください。
4. **固有名詞の遵守**: 参考情報にある人名や地名は、そのまま正確に使用してください。


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