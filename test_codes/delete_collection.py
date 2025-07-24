from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # 또는 다른 embedding 사용
import os

# 🔧 설정값
CHROMA_PERSIST_DIR = "/workspace/data/chromaDB"  # ChromaDB 디렉토리 경로
ENCODER = "dragonkue/BGE-m3-ko"

embedding_model = HuggingFaceEmbeddings(model_name=ENCODER) # 또는 HuggingFaceEmbeddings() 등 사용

# Chroma 클라이언트 초기화
vectorstore = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embedding_model
)

# Chroma 내부 클라이언트 접근
client = vectorstore._client

# 모든 컬렉션 목록 가져오기
all_collections = client.list_collections()

# 빈 컬렉션만 찾아서 삭제
for col in all_collections:
    collection_name = col.name
    collection = client.get_collection(name=collection_name)
    if collection.count() == 0:
        print(f"🗑️ Deleting empty collection: {collection_name}")
        client.delete_collection(name=collection_name)
