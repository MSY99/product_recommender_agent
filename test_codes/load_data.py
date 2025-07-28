from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  # 또는 사용 중인 embedding
import json
import os

# 설정
persist_dir = "/workspace/data/chromaDB"
ENCODER = "dragonkue/BGE-m3-ko"

embedding_model = HuggingFaceEmbeddings(model_name=ENCODER)

vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embedding_model,
)

collection_names = [col.name for col in vectorstore._client.list_collections()]

db_client = vectorstore._client

# 전체 루프
for collection_name in collection_names:
    # Vector DB 연결
    vs = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=collection_name
    )

    # 모든 문서 가져오기
    docs = vs.get()
    documents = docs["documents"]
    metadatas = docs["metadatas"]
    ids = docs["ids"]

    # JSON 형태로 변환
    result = []
    for doc, meta, doc_id in zip(documents, metadatas, ids):
        result.append({
            "id": doc_id,
            "content": doc,
            "metadata": meta
        })

    # 저장
    output_path = f"/workspace/data/tmp2_folder/{collection_name}_documents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ {collection_name} → {output_path} 저장 완료")
