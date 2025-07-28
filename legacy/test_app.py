import streamlit as st
import os
import io
import platform 
import asyncio
import nest_asyncio
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver

from tools.rfp_extracter import extract_text_from_pdf, extract_text_from_docx, convert_rfp_for_RAG

# 환경 변수 세팅 (필요시)
from dotenv import load_dotenv
load_dotenv()

# CSS
st.markdown(
    """
    <style>
    .title {
        color: white;
        font-size: 4em;
        text-align: center;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    body, .stApp, .block-container, .main, footer {
        background: #23272F !important;
    }
    /* 입력창 폼 자체를 더 넓게 */
    .stChatInputContainer form {
        width: 100vw !important;   /* 원하는 폭(%)로 조정, 80vw도 가능 */
        max-width: 1200px !important;
        min-width: 360px !important;
        margin: 0 auto !important;
    }
    /* 텍스트 박스(실제 입력창)도 크고 선명하게 */
    .stChatInputContainer textarea, .stChatInputContainer input {
        width: 100% !important;
        min-height: 56px !important;
        height: 56px !important;
        font-size: 1.2em !important;
        padding: 17px 22px !important;
        border-radius: 12px !important;
        background: #23272F !important;
        color: #F3F6FA !important;
        border: 2px solid #3AAED8 !important;
        box-sizing: border-box;
    }
    /* 비활성화 버튼도 톤 맞추기 */
    .stButton>button:disabled, .stButton>button[disabled] {
        background: #384150 !important;
        color: #bfc6d2 !important;
        opacity: 0.70 !important;
    }
    /* 업로드 안내 메시지 밝게 */
    .stSidebar .stAlert, .stFileUploader .css-1p89gle, .stFileUploader .css-115gedg, .stFileUploader label, .stFileUploader span {
        color: #aee6ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 메인 페이지
st.markdown("<h1 style='color: #ffffff;'>🖥️ KBN 총판 제품 추천 챗봇</h1>", unsafe_allow_html=True)
st.markdown("<span style='color: #e0e0e0; font-size: 18px;'>✨ Ask questions to the GPT-4o agent</span>", unsafe_allow_html=True)

# ===== 사이드바에 버튼 & 파일 업로드 =====
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
                st.success("텍스트 추출 완료!")
                rag_term = convert_rfp_for_RAG(txt_path)
                st.session_state["rag_term"] = rag_term
                st.markdown("#### ➡️ 추출된 RFP 요약")
                st.text_area("->", rag_term, height=300, disabled=True)
        except Exception as e:
            st.error(f"{file_type.upper()} 처리 중 오류: {e}")
    else:
        st.session_state["uploaded_file_path"] = None
        st.info("PDF 또는 DOCX 파일을 선택해 주세요.")
        st.session_state["rag_term"] = ""

# ===== Windows 환경 비동기 설정 =====
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()

if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

# 대화 세션 초기화
if "history" not in st.session_state:
    st.session_state.history = []

MODEL_NAME = "gpt-4o"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(
    model = MODEL_NAME,
    temperature = 0.2,
    max_tokens = 16384,
    api_key= OPENAI_API_KEY,
)

# 대화 기록 출력 함수
def print_message():
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
        elif message["role"] == "assistant":
            # 텍스트를 span 태그로 감싸서 색상 지정 (HTML 지원)
            assistant_text = f"<span style='color:#fff'>{message['content']}</span>"
            st.chat_message("assistant", avatar="🤖").markdown(assistant_text, unsafe_allow_html=True)


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
    st.subheader("Reset Button")
    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.history = []
        st.success("✅ Conversation has been reset.")
        st.rerun()