import streamlit as st
import asyncio
import nest_asyncio
import os
import io
import platform
import shutil
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from rfp_extracter import extract_text_from_pdf, extract_text_from_docx, convert_rfp_to_RAG

# Windows 환경 비동기 설정
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()

# Streamlit 전역 event loop 생성
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(BASE_DIR, "uploaded_pdfs")
os.makedirs(SAVE_FOLDER, exist_ok=True)

if "pdf_upload_key" not in st.session_state:
    st.session_state["pdf_upload_key"] = 0

# 대화 세션 초기화
if "history" not in st.session_state:
    st.session_state.history = []

# GPT-4o 모델 고정
MODEL_NAME = "gpt-4o"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# GPT-4o Langchain 객체 준비
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=16000,
    api_key=OPENAI_API_KEY
)

# PDF 업로드 사이드바
with st.sidebar:
    if st.button("대화 초기화", key="init_chat"):
        st.session_state.chat_history = []
        st.session_state["rag_term"] = ""
        if "uploaded_file_path" in st.session_state and st.session_state["uploaded_file_path"]:
            try:
                os.remove(st.session_state["uploaded_file_path"])
                st.success(f"임시 파일 삭제 완료: `{os.path.basename(st.session_state['uploaded_file_path'])}`")
            except Exception as e:
                st.warning(f"임시 파일 삭제 실패: {e}")
            st.session_state["uploaded_file_path"] = None

    st.subheader("📄 파일 업로드")
    uploaded_file = st.file_uploader(
        "PDF 또는 DOCX 파일을 업로드하세요",
        type=["pdf", "docx"],
        key="file_uploader"
    )

    if uploaded_file is not None:
        # 파일 임시 저장
        file_bytes = uploaded_file.read()
        tmp_path = os.path.join("/tmp", uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        st.session_state["uploaded_file_path"] = tmp_path
        st.success(f"업로드 완료: `{uploaded_file.name}`")

        file_type = None
        if uploaded_file.name.lower().endswith(".pdf"):
            file_type = "pdf"
        elif uploaded_file.name.lower().endswith(".docx"):
            file_type = "docx"

        # 파일 buffer
        file_buffer = io.BytesIO(file_bytes)
        file_buffer.seek(0)

        try:
            if file_type == "pdf":
                st.info("PDF 파일로 인식됨. 텍스트를 추출합니다...")
                txt_path = extract_text_from_pdf(file_buffer)      # 임시 txt 경로(str)
            elif file_type == "docx":
                st.info("DOCX 파일로 인식됨. 텍스트를 추출합니다...")
                txt_path = extract_text_from_docx(file_buffer)     # 임시 txt 경로(str)
            else:
                st.warning("지원하지 않는 파일 형식입니다.")
                txt_path = None

            if txt_path:
                # (화면에 미리보기로 띄우는 용도)
                with open(txt_path, encoding="utf-8") as tf:
                    extracted_text = tf.read()
                st.success("텍스트 추출 완료!")
                st.text_area(f"{file_type.upper()} 텍스트", extracted_text, height=300)
                
                # [수정] 임시파일 경로(txt_path)를 convert_rfp_to_RAG에 넘겨야 안전!
                rag_term = convert_rfp_to_RAG(txt_path)
                st.session_state["rag_term"] = rag_term
        except Exception as e:
            st.error(f"{file_type.upper()} 처리 중 오류: {e}")
    else:
        st.session_state["uploaded_file_path"] = None
        st.info("PDF 또는 DOCX 파일을 선택해 주세요.")
        st.session_state["rag_term"] = ""
    st.divider()

# 메인 페이지
st.title("🖥️ Agent Demo")
st.markdown("✨ Ask questions to the GPT-4o agent about your PDF document.")

# 대화 기록 출력 함수
def print_message():
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant", avatar="🤖").markdown(message["content"])

print_message()

# 질문 처리 함수 (비동기)
async def process_query(query):
    try:
        # PDF 텍스트 추출된 폴더가 있으면 내용 결합하여 context로 사용
        context = ""
        if "preprocessed_pdf_dir" in st.session_state:
            output_dir = st.session_state["preprocessed_pdf_dir"]
            txt_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".txt"))
            for file in txt_files:
                with open(os.path.join(output_dir, file), encoding="utf-8") as f:
                    context += f.read() + "\n"
        else:
            context = ""

        prompt = f"""You are a helpful assistant. Answer the user's question based on the following context (if any).
        If there is no context, just answer as best as you can.
        ---
        Context:
        {context}
        ---
        Question: {query}
        Answer:"""

        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        return resp.content
    except Exception as e:
        return f"❌ Error: {e}"

# 사용자 입력 처리
user_query = st.chat_input("💬 Enter your question")
if user_query:
    st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Generating answer..."):
            answer = st.session_state.event_loop.run_until_complete(process_query(user_query))
            st.markdown(answer)
    st.session_state.history.append({"role": "user", "content": user_query})
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()

# 리셋 버튼(사이드바)
with st.sidebar:
    st.subheader("🔄 Actions")
    if st.button("Reset Conversation", use_container_width=True, type="primary"):
        st.session_state.history = []
        st.success("✅ Conversation has been reset.")
        st.rerun()
