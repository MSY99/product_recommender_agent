"""
➡️ Naive RAG의 검색 결과를 테스트하기 위한 코드입니다.
"""
from typing import List, Any
from string import Template

from google import genai
from google.genai.types import Part

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import pandas as pd

from dotenv import load_dotenv

load_dotenv("/workspace/Dhq_chatbot/pdf2DB_Agent/.env")

persist_dir = "/workspace/data/chromaDB"
ENCODER = "dragonkue/BGE-m3-ko"

embedding_model = HuggingFaceEmbeddings(model_name=ENCODER)

vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embedding_model,
)

db_client = vectorstore._client

# collection 선택하는 함수
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
답변은 반드시 쉼표(,)로 구분된 collection 이름의 나열이어야 합니다. (예시: collection1, collection3)                 
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

    selected_collections = response.text.strip()

    return selected_collections

# collection 내에서 검색하는 함수
def rag_at_selected_collection(query: str, collection_name: str) -> List[Any]:
    """
    선택된 colleciton에 대해 RAG 검색을 수행
    """
    vs = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )
    # retriever = vs.as_retriever(search_kwargs={"k": 3})
    # retrieved_docs = retriever.invoke(query)

    retrieved_docs = vs.similarity_search_with_score(query, k=5)
    filtered_docs = [(doc, score) for doc, score in retrieved_docs if score >= 0.3]

    # 🔍 None 제거
    valid_docs = [(doc, score) for doc, score in filtered_docs if doc.page_content]

    if not valid_docs:
        return "❗ 관련 문서를 찾을 수 없습니다."
    
    result = "" \
    
    for idx, (doc, score) in enumerate(valid_docs, 1):
        df = pd.DataFrame([doc.metadata])
        result += f"검색된 문서 {idx}: similarity {score:.3f} \n"
        result += f"{doc.page_content}\n"
        #result += f"제품 상세 스펙: \n{df.to_markdown(index=False)}\n\n"
    
    return result

# 검색된 결과에 대해 필터링 + 랭킹을 매기는 함수


if __name__ == "__main__":
    query = """제공된 RFP에 따르면, 구매자는 Dual Active/Active 컨트롤러 구성(컨트롤러당 1.9GHz 6-core 프로세서, 64GB Cache Memory, 정전 시 캐시 보호 기능), Usable 70TiB (Physical 103.2TB) 이상의 Distributed RAID6 구성의 스토리지를 요구하며, 2.4TB 10K 2.5 Inch SAS 디스크 43개를 기본 제공하고 최대 1,008개까지 확장 가능해야 합니다. 인터페이스는 FC, 25G iSCSI(iWARP, RoCE)를 지원하며, Host Interface로 10Gb UTP 이더넷 4포트, 16Gb FC 8포트를 제공해야 합니다. 또한, Web 기반 운영 소프트웨어, RAID 0, 1, 10 및 Distributed RAID 5, 6 지원, 씬 프로비저닝, UNMAP, 압축 및 중복제거, 로컬/원격 복제, 단방향 마이그레이션, 티어링, 암호화, 실시간 스토리지 이중화 솔루션 등의 기능을 지원해야 하고, 주요 구성 요소(Power, Fan, Controller)는 이중화 구성되어야 합니다. SAN 스위치 2식은 각 포트 별 4/8/16/32Gbps 자동인식 전이중 통신방식지원, F_Port, E_Port, M_Port, D_Port 지원, NPIV, Frame-based Trunking 지원, 펌웨어 업그레이드 및 로그 저장 가능한 USB 포트를 제공해야하며, 16포트이상 제공(16Gbps SFP포함), 최대 24포트 이상 확장가능해야 합니다. 마지막으로, 3년 무상 A/S(24시간 1시간 이내 방문 서비스), 제조사 물품공급 및 기술지원확약서, 기존 시스템과의 호환성, 기술 지원 및 교육 제공이 필요합니다."""

    collection_names = select_collection_list(query)
    result = rag_at_selected_collection(query, 'dell_storage')
    print(query)
    print("\n")
    print(f"Selected Collections: {collection_names}\n\n")
    print(result)
