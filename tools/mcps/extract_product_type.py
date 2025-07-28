"""
➡️ RFP 내용 기반 필요 제품군 추출 기능을 테스트하기 위한 코드입니다.
"""
import json
import ast 

from typing import List, Any
from string import Template

from google import genai
from google.genai.types import Part

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import pandas as pd
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv("/workspace/Dhq_chatbot/pdf2DB_Agent/.env")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Exatract_product_type_tools")

persist_dir = "/workspace/data/chromaDB"
ENCODER = "dragonkue/BGE-m3-ko"

embedding_model = HuggingFaceEmbeddings(model_name=ENCODER)

vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embedding_model,
)

db_client = vectorstore._client
client = OpenAI()

# RFP 요약문을 가져오는 함수
@mcp.tool()
def get_rfp_summary(rfp_text: str) -> str:
    """
    streamlit session에 임시로 저장된 RFP 요약문(=제품 규격서)가져오기
    """
    try:
        with open("/tmp/rfp_summary.txt", "r", encoding="utf-8") as file:
            rfp_summary = file.read().strip()
            return rfp_summary
    except Exception as e:
        return f"Error reading RFP summary: {str(e)}"

# collection 선택하는 함수
@mcp.tool()
def select_collection_list(query: str) -> List[str]:
    """
    주어진 query를 읽고 검색이 필요한 제품군을 확인하여 검색을 실행할 Collection List를 반환
    """
    # DB에 존재하는 collection list를 가져오기
    all_collections = str(db_client.list_collections())

    # 간단한 작업이므로, Gemini 2.5 flash lite 모델 사용
    llm = genai.Client()

    prompt_t = Template("""당신은 제품 요구 사항을 읽고 어떤 DB에서 검색을 해야할지 결정하는 역할을 해야 합니다.
주어진 DB의 collection list와 제품의 요구 사항을 읽고 어떤 collection에서 제품 검색을 수행해야 할지 collection 이름을 나열하세요.

답변은 반드시 쉼표(,)로 구분된 collection 이름의 리스트 형태이어야 합니다. (예시: ['collection1', 'collection3'])                 
[DB의 collection list]
$all_collections
                      
[제품의 요구 사항]
$query
                      
[제품을 검색할 DB 목록]
답변: """)
    
    prompt = prompt_t.substitute(all_collections=all_collections, query=query)

    response = llm.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt]
    )

    selected_collections_str = response.text.strip()

    try:
        # 문자열을 안전하게 List[str]로 변환
        selected_collections = ast.literal_eval(selected_collections_str)
        if not isinstance(selected_collections, list):
            raise ValueError("LLM 응답이 리스트가 아닙니다.")
        return selected_collections
    except Exception as e:
        raise ValueError(f"LLM 응답 파싱 오류: {e}\n원본 응답: {selected_collections_str}")

# GPT 4o API를 통해 제품 추천 여부에 대해 YES / NO를 반환하는 함수
def call_gpt4o_judge(product_info: dict, requirements: dict) -> str:
    """GPT-4o를 통해 해당 제품이 요구 조건에 부합하는지 평가"""

    system_prompt = "당신은 IT 프로덕트 전문가로서, 제품이 주어진 요구 조건을 만족하는지 평가하는 역할을 수행합니다."
    user_prompt_t = Template("""
다음은 제품 요구 조건과 제품 정보입니다. 요구 조건에 따라 제품 정보가 유사한 점 혹은 충분한 점이 있는지 확인하세요.
요구 조건의 내용을 완벽히 충족하지 않더라도, 제품이 요구 조건을 조금이라도 만족할 수 있다면 'YES'라고 답변하세요.

요구 조건:
$product_info

제품 정보:
$requirements

이 제품이 요구 조건을 조금이라도 만족할 수 있는 제품이라면 'YES'만 출력하고, 그렇지 않으면 'NO'만 출력해주세요. 이유는 출력하지 마세요.
""")
    
    user_prompt = user_prompt_t.substitute(product_info=product_info, requirements=requirements)

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    answer = response.choices[0].message.content.strip().upper()
    return answer

def merge_doc_and_metadata(doc: str, metadatas: dict) -> str:
    lines = []

    for key, value in metadatas.items():
        lines.append(f"{key}: {value}")

    lines.append("제품 설명: " + doc)
    
    return "\n".join(lines)

# collection 내 제품 별 추천 여부 결정 함수
@mcp.tool()
def evaluate_products_from_vectorstore(collection_name:str, requirements:str) -> list:
    """요구 조건에 따라 VectorDB에서 top_k 제품 검색 후 LLM 판단을 통해 추천 목록 생성"""
    db = Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )
    
    all_products = db.get()

    docs = all_products["documents"]
    metadatas = all_products["metadatas"]

    recommended_products = []

    for i, (doc, metas) in enumerate(zip(docs, metadatas)):
        product_info = merge_doc_and_metadata(doc, metas)

        decision_for_recommend = call_gpt4o_judge(product_info, requirements)

        if decision_for_recommend == "YES":
            print(f"Product {doc} is recommended.")
            recommended_products.append(product_info)
        elif decision_for_recommend == "NO":
            print(f"Product {doc} is not recommended.")
        else:
            print("Wrong response from GPT-4o, expected 'YES' or 'NO'.")

    if not recommended_products:
        return "No products meet the requirements."
    else:
        return recommended_products

# 추천 후보 제품에 대한 보고서 작성 함수
# @mcp.tool()
def final_recommendation_report(requirements:str, recommended_products: list) -> str:
    """추천된 제품에 대한 보고서 작성"""
    joined_products = "\n\n".join(recommended_products)

    print(f"Recommended Product List: {recommended_products}\n")
    print(f"Product List: {joined_products}\n")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 IT 제품 전문가로서, 주어진 제품 정보를 바탕으로 추천 보고서를 작성하는 역할을 수행해야 합니다."},
            {"role": "user", "content": f"""다음의 제품 요구 사항과 추천된 제품 정보 리스트를 읽고 제품 추천 보고서를 작성해주세요.
보고서를 작성할 때는 어떤 제품을 왜 추천하는지 제품 별로 간단한 이유를 들어 설명해야 합니다.
             
[제품 요구 사항]
{requirements}


[추천된 제품 정보]
{joined_products}"""}
        ]
    )

    report = response.choices[0].message.content.strip()

    return report

@mcp.tool()
def load_product_name_from_vectordb(collection_name: str) -> List[str]:
    """
    collection name을 입력 받아 그 안에 저장된 모든 제품 명을 가져오는 함수
    """
    db_client = Chroma(persist_directory=persist_dir, collection_name=collection_name)

    result = db_client._collection.get(include=["documents", "metadatas"])

    titles = []
    for meta in result["metadatas"]:
        title = meta.get("product_name" or meta.get("id") or None)
        if title:
            titles.append(title)
    return titles

if __name__ == "__main__":
    mcp.run(transport="stdio")
