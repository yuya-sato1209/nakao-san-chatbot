import os
import json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm # 進捗状況を表示するためにtqdmライブラリを使用します

# --- AIへの指示（プロンプト）を定義 ---
# ここでAIに「校正役」としての役割を厳密に与えます。
CLEANING_PROMPT_TEMPLATE = """
あなたは、プロの編集者です。あなたの仕事は、以下に提供される「文字起こしテキスト」を、自然で文法的に正しい、読みやすい「校正済みテキスト」に変換することです。

【厳格なルール】
1.  **意味を変えない**: 元のテキストの事実や固有のニュアンス、話し手の意図を絶対に改変してはいけません。
2.  **ノイズの除去**: 「えーと」「あのー」のようなフィラー（言い淀み）や、意味のない単語の繰り返し（例：「観光、函館観光、函館」）は除去してください。
3.  **誤字の修正**: 明らかな音声認識の誤字（例：「明治電脳」→「明治天皇」、「最盤書」→「裁判所」）は、文脈に基づいて修正してください。
4.  **文章の整形**: 不自然な句読点や改行を修正し、文脈に応じて自然な段落分けを行ってください。
5.  **情報の追加禁止**: 元のテキストにない情報を絶対に追加してはいけません。
6.  **出力形式**: 校正後のテキストのみを出力してください。それ以外の前置きや解説は不要です。

---
文字起こしテキスト:
{text}
"""

def clean_text_with_ai(llm_chain, dirty_text):
    """
    AIチェーンを呼び出して、単一のテキストをクリーニングする関数
    """
    try:
        # AIに校正を依頼
        response = llm_chain.invoke({"text": dirty_text})
        return response
    except Exception as e:
        print(f"AIの呼び出し中にエラーが発生しました: {e}")
        return dirty_text # エラー時は元のテキストをそのまま返す

def main():
    """
    メインの処理を実行する関数
    """
    # --- 1. セットアップ ---
    print("AIクリーニングプログラムを開始します。")
    load_dotenv()
    
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEYが.envファイルに設定されていません。")
        return

    # AIモデルとプロンプト、出力パーサーをチェーン（連結）する
    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.0)
    prompt = ChatPromptTemplate.from_template(CLEANING_PROMPT_TEMPLATE)
    output_parser = StrOutputParser()
    llm_chain = prompt | llm | output_parser

    input_file = "rag_data.jsonl"
    output_file = "rag_data_cleaned.jsonl"

    # --- 2. データの読み込み ---
    print(f"'{input_file}' から元のデータを読み込んでいます...")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"エラー: '{input_file}' が見つかりません。")
        return
    
    print(f"合計 {len(lines)} 行のデータを読み込みました。")

    # --- 3. データのクリーニング処理 ---
    print(f"AIによるクリーニングを開始します。'{output_file}' に保存します...")
    
    cleaned_data = []
    
    # tqdmを使って進捗バーを表示
    for line in tqdm(lines, desc="AIがデータを校正中"):
        if not line.strip():
            continue # 空行はスキップ
        
        try:
            data = json.loads(line)
            dirty_text = data.get("text")

            if not dirty_text or not dirty_text.strip():
                # テキストが空の場合は、そのまま（メタデータだけ）保存
                cleaned_data.append(data)
                continue
            
            # AIを呼び出してテキストを校正
            clean_text = clean_text_with_ai(llm_chain, dirty_text)
            
            # 元のデータのテキストだけを、校正済みのものに入れ替える
            data["text"] = clean_text
            cleaned_data.append(data)
            
        except json.JSONDecodeError:
            print(f"警告: JSONの解析に失敗した行をスキップしました: {line[:50]}...")
        except Exception as e:
            print(f"警告: 不明なエラーで行をスキップしました: {e}")

    # --- 4. データの保存 ---
    print(f"クリーニングが完了しました。'{output_file}' にデータを書き込みます...")
    with open(output_file, "w", encoding="utf-8") as f:
        for data in cleaned_data:
            # ensure_ascii=False で日本語をそのまま保存
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    print("すべての処理が完了しました。")

if __name__ == "__main__":
    main()