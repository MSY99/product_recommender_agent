import streamlit as st
import os
import io
import json
import platform 
import asyncio
import nest_asyncio

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from utils import astream_graph, random_uuid

from tools.rfp_extracter import extract_text_from_hwp, extract_text_from_pdf_by_olmocr, convert_rfp_for_RAG

from dotenv import load_dotenv
load_dotenv("/workspace/Dhq_chatbot/chat_demo_test/.env")

CONFIG_FILE_PATH = "/workspace/Dhq_chatbot/chat_demo_test/mcp_config.json"
MODEL_NAME = "claude-3-7-sonnet-latest"

SYSTEM_PROMPT = """<ROLE>
You are a smart agent with an ability to use tools. 
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question. 
If you are failed to answer the question, try different tools to get context.
Your answer should be very polite and professional.
</ROLE>

----

<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question is consist of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- If you are failed to answer the question, try different tools to get context.

Step 3: Answer the question
- Answer the question in the same language as the question.
- Your answer should be very polite and professional.

Step 4: Provide the source of the answer(if applicable)
- If you've used the tool, provide the source of the answer.
- Valid sources are either a website(URL) or a document(PDF, etc).

Guidelines:
- If you've used the tool, your answer should be based on the tool's output(tool's output is more important than your own knowledge).
- If you've used the tool, and the source is valid URL, provide the source(URL) of the answer.
- Skip providing the source if the source is not URL.
- Answer in the same language as the question.
- Answer should be concise and to the point.
- Avoid response your output with any other information than the answer and the source.  
</INSTRUCTIONS>

----

<OUTPUT_FORMAT>
(concise answer to the question)

**Source**(if applicable)
- (source1: valid URL)
- (source2: valid URL)
- ...
</OUTPUT_FORMAT>
"""

OUTPUT_TOKEN_INFO = {
    "gpt-4o": {"max_tokens": 16000},
    "claude-3-7-sonnet-latest": {"max_tokens": 16000},
    "claude-sonnet-4-20250514": {"max_tokens": 64000},
}

# 메인 페이지 Header
st.markdown("<h1>🖥️ KBN 총판 제품 추천 챗봇</h1>", unsafe_allow_html=True)
st.markdown("<span>✨ Ask questions to the Agent</span>", unsafe_allow_html=True)

# ===== Windows 환경 비동기 설정 =====
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()

if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

# ===== 초기 상태 변수 정의 (맨 위 or 페이지 로드 직후 위치에 넣기) =====
if "mcp_agent" not in st.session_state:
    st.session_state.mcp_agent = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "recursion_limit" not in st.session_state:
    st.session_state.recursion_limit = 120

# ====== MCP 서버 연결 설정 ======
def load_mcp_config():
    default_config = {}
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            st.error(f"MCP Server Connection config file not found: {CONFIG_FILE_PATH}")

    except Exception as e:
        st.error(f"Error loading MCP config: {e}")
        return default_config

# ====== MCP_Client 초기화 함수 ======
async def init_mcp_client(config_path = CONFIG_FILE_PATH, model_name = MODEL_NAME):
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        client = MultiServerMCPClient(config)
        tools = await client.get_tools()

        if "claude" in model_name:
            llm = ChatAnthropic(
                model=model_name,
                temperature=0.2,
                max_tokens=OUTPUT_TOKEN_INFO[model_name]["max_tokens"],
            )
        else:
            llm = ChatOpenAI(
                model=model_name,
                max_completion_tokens=OUTPUT_TOKEN_INFO[model_name]["max_tokens"],
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

        agent = create_react_agent(
            llm,
            tools,
            checkpointer=MemorySaver(),
            prompt=SYSTEM_PROMPT,
        )

        return agent, tools, client
    except Exception as e:
        st.error(f"Error when initializing MCP client: {e}")
        return None, [], None

# ====== MCP Agent 설정 세션 변수 초기화 ======
if "mcp_agent" not in st.session_state or st.session_state.mcp_agent is None:
    with st.spinner("🔄 Initializing Agent..."):
        mcp_agent, tools, client = st.session_state.event_loop.run_until_complete(
            init_mcp_client()
        )
        st.session_state.mcp_agent = mcp_agent
        st.session_state.mcp_tools = tools
        st.session_state.mcp_client = client

    if st.session_state.mcp_agent is None:
        st.error("❌ Initializing MCP Agent error.")
        st.stop()

# ====== 사이드 바: 파일 업로드 및 전처리 결과 표시 ======
with st.sidebar:
    st.subheader("📨 File Upload")

    uploaded_file = st.file_uploader(
        "PDF 또는 HWP 확장자의 RFP(규격서) 파일을 올려주세요.",
        type=["pdf", "hwp"],
        key="file_uploader"
    )

    # ✅ 파일이 새로 업로드되면 이전 상태 초기화
    if uploaded_file is not None:
        if "last_uploaded_filename" not in st.session_state or \
           st.session_state["last_uploaded_filename"] != uploaded_file.name:
            st.session_state["rfp_processed"] = False
            st.session_state["rag_term"] = ""
            st.session_state["last_uploaded_filename"] = uploaded_file.name

    if uploaded_file is not None and not st.session_state.get("rfp_processed"):
        try:
            file_bytes = uploaded_file.read()
            tmp_path = os.path.join("/tmp", uploaded_file.name)

            with open(tmp_path, "wb") as f:
                f.write(file_bytes)

            st.session_state["uploaded_file_path"] = tmp_path

            file_type = None
            if uploaded_file.name.lower().endswith(".pdf"):
                file_type = "pdf"
            elif uploaded_file.name.lower().endswith(".hwp"):
                file_type = "hwp"

            file_buffer = io.BytesIO(file_bytes)
            file_buffer.seek(0)

            if file_type == "pdf":
                st.info("PDF 텍스트 추출 중...")
                # txt_path = extract_text_from_pdf(file_buffer)
                txt_path = extract_text_from_pdf_by_olmocr(file_buffer, uploaded_file.name)
            elif file_type == "hwp":
                st.info("HWP 텍스트 추출 중...")
                txt_path = extract_text_from_hwp(file_buffer)
            else:
                st.warning("지원하지 않는 파일 형식입니다.")
                txt_path = None

            if txt_path:
                rag_term = convert_rfp_for_RAG(txt_path)
                st.session_state["rag_term"] = rag_term
                st.session_state["rfp_processed"] = True
                st.success("RFP 텍스트 요약 완료!")

                # ✅ 텍스트 파일로 저장
                summary_path = "/tmp/rfp_summary.txt"
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(rag_term)

        except Exception as e:
            st.error(f"파일 처리 중 오류 발생: {e}")

    if st.session_state.get("rfp_processed") and st.session_state.get("rag_term"):
        st.markdown("#### ➡️ 추출된 RFP 요약")
        st.text_area(
            label="->",
            value=st.session_state["rag_term"],
            height=300,
            disabled=True,
            key="rfp_summary_area"
        )

    elif uploaded_file is None:
        st.session_state["rfp_processed"] = False
        st.session_state["rag_term"] = ""
        st.info("RFP 파일을 업로드 해주세요.")

        
# 대화 세션 초기화
if "history" not in st.session_state:
    st.session_state.history = []

# LLM 모델 설정
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 대화 기록 출력 함수
def print_messages():
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
        elif message["role"] == "assistant":
            assistant_msg = f"<span>{message['content']}</span>"
            st.chat_message("assistant", avatar="🤖").markdown(assistant_msg, unsafe_allow_html=True)

print_messages()

def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    Creates a streaming callback function.

    This function creates a callback function to display responses generated from the LLM in real-time.
    It displays text responses and tool call information in separate areas.

    Args:
        text_placeholder: Streamlit component to display text responses
        tool_placeholder: Streamlit component to display tool call information

    Returns:
        callback_func: Streaming callback function
        accumulated_text: List to store accumulated text responses
        accumulated_tool: List to store accumulated tool call information
    """
    accumulated_text = []
    accumulated_tool = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        message_content = message.get("content", None)

        if isinstance(message_content, AIMessageChunk):
            content = message_content.content
            # If content is in list form (mainly occurs in Claude models)
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                # Process text type
                if message_chunk["type"] == "text":
                    accumulated_text.append(message_chunk["text"])
                    text_placeholder.markdown("".join(accumulated_text))
                # Process tool use type
                elif message_chunk["type"] == "tool_use":
                    if "partial_json" in message_chunk:
                        accumulated_tool.append(message_chunk["partial_json"])
                    else:
                        tool_call_chunks = message_content.tool_call_chunks
                        tool_call_chunk = tool_call_chunks[0]
                        accumulated_tool.append(
                            "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                        )
                    with tool_placeholder.expander(
                        "🔧 Tool Call Information", expanded=True
                    ):
                        st.markdown("".join(accumulated_tool))
            # Process if tool_calls attribute exists (mainly occurs in OpenAI models)
            elif (
                hasattr(message_content, "tool_calls")
                and message_content.tool_calls
                and len(message_content.tool_calls[0]["name"]) > 0
            ):
                tool_call_info = message_content.tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # Process if content is a simple string
            elif isinstance(content, str):
                accumulated_text.append(content)
                text_placeholder.markdown("".join(accumulated_text))
            # Process if invalid tool call information exists
            elif (
                hasattr(message_content, "invalid_tool_calls")
                and message_content.invalid_tool_calls
            ):
                tool_call_info = message_content.invalid_tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information (Invalid)", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # Process if tool_call_chunks attribute exists
            elif (
                hasattr(message_content, "tool_call_chunks")
                and message_content.tool_call_chunks
            ):
                tool_call_chunk = message_content.tool_call_chunks[0]
                accumulated_tool.append(
                    "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                )
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # Process if tool_calls exists in additional_kwargs (supports various model compatibility)
            elif (
                hasattr(message_content, "additional_kwargs")
                and "tool_calls" in message_content.additional_kwargs
            ):
                tool_call_info = message_content.additional_kwargs["tool_calls"][0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
        # Process if it's a tool message (tool response)
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append(
                "\n```json\n" + str(message_content.content) + "\n```\n"
            )
            with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                st.markdown("".join(accumulated_tool))
        return None

    return callback_func, accumulated_text, accumulated_tool

# 사용자 질의 처리 함수 (비동기)
async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=300):
    """
    Processes user questions and generates responses.

    This function passes the user's question to the agent and streams the response in real-time.
    Returns a timeout error if the response is not completed within the specified time.

    Args:
        query: Text of the question entered by the user
        text_placeholder: Streamlit component to display text responses
        tool_placeholder: Streamlit component to display tool call information
        timeout_seconds: Response generation time limit (seconds)

    Returns:
        response: Agent's response object
        final_text: Final text response
        final_tool: Final tool call information
    """
    try:
        if st.session_state.mcp_agent:
            streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
                get_streaming_callback(text_placeholder, tool_placeholder)
            )
            try:
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.mcp_agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=streaming_callback,
                        config=RunnableConfig(
                            recursion_limit=st.session_state.recursion_limit,
                            thread_id=st.session_state.thread_id,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ Request time exceeded {timeout_seconds} seconds. Please try again later."
                return {"error": error_msg}, error_msg, ""

            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            return response, final_text, final_tool
        else:
            return (
                {"error": "🚫 Agent has not been initialized."},
                "🚫 Agent has not been initialized.",
                "",
            )
    except Exception as e:
        import traceback

        error_msg = f"❌ Error occurred during query processing: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""

# 사용자 입력 처리
user_query = st.chat_input("💬 Enter your question")
if user_query:
    st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
    with st.chat_message("assistant", avatar="🤖"):
        tool_placeholder = st.empty()
        text_placeholder = st.empty()
        with st.spinner("Generating answer..."):
            response, final_text, final_tool = st.session_state.event_loop.run_until_complete(
                process_query(user_query,
                              text_placeholder,
                              tool_placeholder))
            # st.markdown(final_text)
    st.session_state.history.append({"role": "user", "content": user_query})
    st.session_state.history.append({"role": "assistant", "content": final_text})
    # print_messages()

# 리셋 버튼(사이드바)
with st.sidebar:
    st.subheader("Reset Button")
    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.history = []
        st.success("✅ Conversation has been reset.")
        st.rerun()