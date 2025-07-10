"""
➡️ Naive RAG의 검색 결과를 테스트하기 위한 코드입니다.
"""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

persist_dir = "/workspace/data/chromaDB"
ENCODER = "dragonkue/BGE-m3-ko"

embedding_model = HuggingFaceEmbeddings(model_name=ENCODER)
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

def rag_to_context(query: str) -> str:
    retrieved_docs = retriever.invoke(query)

    # 🔍 None 제거
    valid_docs = [doc for doc in retrieved_docs if doc.page_content]

    if not valid_docs:
        return "❗ 관련 문서를 찾을 수 없습니다."

    result = "\n\n".join(
        f"Title: {doc.metadata.get('title', 'No Title')}\n"
        f"Category: {doc.metadata.get('category', 'N/A')}\n"
        f"{doc.page_content}"
        for doc in valid_docs
    )
    return result

if __name__ == "__main__":
    query = "소셜 미디어 플랫폼 회사에서 IBM 서버를 도입한 사례를 알려줘."

    result = rag_to_context(query)
    print(result)
