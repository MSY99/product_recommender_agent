import streamlit as st
import asyncio
import nest_asyncio
import json
import os
import time
import platform
import shutil
#from agent_graph import build_agent

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Apply nest_asyncio: Allow nested calls within an already running event loop
nest_asyncio.apply()

# Create and reuse global event loop (create once and continue using)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from pdf_pipeline import split_pdf_by_page, extract_text_with_data_check

# Load environment variables (get API keys and settings from .env file)
load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# config.json file path setting
CONFIG_FILE_PATH = os.path.join(BASE_DIR, "config.json")

# Function to load settings from JSON file
def load_config_from_json():
    """
    Loads settings from config.json file.
    Creates a file with default settings if it doesn't exist.

    Returns:
        dict: Loaded settings
    """
    default_config = {
        "server-sequential-thinking": {
            "command": "npx",
            "args": [
            "-y",
            "@smithery/cli@latest",
            "run",
            "@smithery-ai/server-sequential-thinking",
            "--key",
            "47a2dcfa-3670-464c-9ebf-7b0ab9005f2a"
            ],
            "transport": "stdio"
        },
    }
    
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Create file with default settings if it doesn't exist
            save_config_to_json(default_config)
            return default_config
    except Exception as e:
        st.error(f"Error loading settings file: {str(e)}")
        return default_config

# Function to save settings to JSON file
def save_config_to_json(config):
    """
    Saves settings to config.json file.

    Args:
        config (dict): Settings to save
    
    Returns:
        bool: Save success status
    """
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving settings file: {str(e)}")
        return False

# Initialize login session variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Check if login is required
use_login = os.environ.get("USE_LOGIN", "false").lower() == "true"

# Change page settings based on login status
if use_login and not st.session_state.authenticated:
    # Login page uses default (narrow) layout
    st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠")
else:
    # Main app uses wide layout
    st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠", layout="wide")


# ====== 사이드 바: 파일 업로드 및 전처리 결과 표시 ======
with st.sidebar:
    st.subheader("📨 File Upload")

    uploaded_file = st.file_uploader(
        "PDF 또는 DOCX 확장자의 RFP(규격서) 파일을 올려주세요.",
        type=["pdf", "docx"],
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
            elif uploaded_file.name.lower().endswith(".docx"):
                file_type = "docx"

            file_buffer = io.BytesIO(file_bytes)
            file_buffer.seek(0)

            if file_type == "pdf":
                st.info("PDF 텍스트 추출 중...")
                txt_path = extract_text_from_pdf(file_buffer)
            elif file_type == "docx":
                st.info("DOCX 텍스트 추출 중...")
                txt_path = extract_text_from_docx(file_buffer)
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


# 본문(main)에는 결과나 알림 메시지 없이, 채팅 기능만 유지

st.sidebar.divider()  # Add divider

# Existing page title and description
st.title("🖥️ Agent Demo")
st.markdown("✨ Ask questions to the ReAct agent that utilizes MCP tools.")

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
    "claude-3-5-sonnet-latest": {"max_tokens": 8192},
    "claude-3-5-haiku-latest": {"max_tokens": 8192},
    "claude-3-7-sonnet-latest": {"max_tokens": 16000},
    "claude-sonnet-4-20250514": {"max_tokens": 16000},
    "gpt-4o": {"max_tokens": 16000},
    "gpt-4o-mini": {"max_tokens": 16000},
}

# Initialize session state
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False  # Session initialization flag
    st.session_state.agent = None  # Storage for ReAct agent object
    st.session_state.history = []  # List for storing conversation history
    st.session_state.mcp_client = None  # Storage for MCP client object
    st.session_state.timeout_seconds = (
        300  # Response generation time limit (seconds), default 120 seconds
    )
    st.session_state.selected_model = (
        "claude-sonnet-4-20250514"  # Default model selection
    )
    st.session_state.recursion_limit = 100  # Recursion call limit, default 100

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()


# --- Function Definitions ---


async def cleanup_mcp_client():
    """
    Safely terminates the existing MCP client.

    Properly releases resources if an existing client exists.
    """
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:

            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
        except Exception as e:
            import traceback

            # st.warning(f"Error while terminating MCP client: {str(e)}")
            # st.warning(traceback.format_exc())


def print_message():
    """
    Displays chat history on the screen.

    Distinguishes between user and assistant messages on the screen,
    and displays tool call information within the assistant message container.
    """
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]

        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            # Create assistant message container
            with st.chat_message("assistant", avatar="🤖"):
                # Display assistant message content
                st.markdown(message["content"])

                # Check if the next message is tool call information
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    # Display tool call information in the same container as an expander
                    with st.expander("🔧 Tool Call Information", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2  # Increment by 2 as we processed two messages together
                else:
                    i += 1  # Increment by 1 as we only processed a regular message
        else:
            # Skip assistant_tool messages as they are handled above
            i += 1


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


async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
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
        if st.session_state.agent:
            streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
                get_streaming_callback(text_placeholder, tool_placeholder)
            )
            try:
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
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


async def initialize_session(mcp_config=None):
    """
    Initializes MCP session and agent.

    Args:
        mcp_config: MCP tool configuration information (JSON). Uses default settings if None

    Returns:
        bool: Initialization success status
    """
    with st.spinner("🔄 Connecting to MCP server..."):
        # First safely clean up existing client
        await cleanup_mcp_client()

        if mcp_config is None:
            # Load settings from config.json file
            mcp_config = load_config_from_json()
        client = MultiServerMCPClient(mcp_config)
        tools = await client.get_tools()
        st.session_state.tool_count = len(tools)
        st.session_state.mcp_client = client

        # Initialize appropriate model based on selection
        selected_model = st.session_state.selected_model

        if selected_model in [
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ]:
            model = ChatAnthropic(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
            )
        elif selected_model in [
            "gpt-4o", 
            "gpt-4o-mini"
        ]:  # Use OpenAI model
            model = ChatOpenAI(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
            )
        # elif selected_model in [
        #     "gemini-2.0-flash",
        #     "gemini-2.5-flash",
        #     "gemini-2.5-pro"
        # ]:
        #     model = genai.Client()

        # agent = build_agent()
        agent = create_react_agent(
            model,
            tools,
            checkpointer=MemorySaver(),
            prompt=SYSTEM_PROMPT,
        )
        st.session_state.agent = agent
        st.session_state.session_initialized = True
        return True


# --- Initialize default session (if not initialized) ---
if not st.session_state.session_initialized:
    st.info(
        "MCP server and agent are not initialized. Please click the 'Apply Settings' button in the left sidebar to initialize."
    )


# --- Print conversation history ---
print_message()

# --- User input and processing ---
user_query = st.chat_input("💬 Enter your question")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
        with st.chat_message("assistant", avatar="🤖"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_query(
                        user_query,
                        text_placeholder,
                        tool_placeholder,
                        st.session_state.timeout_seconds,
                    )
                )
            )
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append(
                {"role": "assistant", "content": final_text}
            )
            if final_tool.strip():
                st.session_state.history.append(
                    {"role": "assistant_tool", "content": final_tool}
                )
            st.rerun()
    else:
        st.warning(
            "⚠️ MCP server and agent are not initialized. Please click the 'Apply Settings' button in the left sidebar to initialize."
        )