import streamlit as st

# CSS
st.markdown(
    """
    <style>
    /* Latar belakang aplikasi */
    .stApp {
        background-color: #1E1E2F; /* Latar belakang abu-abu gelap */
        color: white;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;  /* Mengatur warna latar belakang tombol */
    }
    .title {
        color: white;
        font-size: 4em;
        text-align: center;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .content {
        color: white;
        font-size: 1em;
    }
    label {
        color: white !important;
    }
    input {
        color: white !important;
        background-color: black !important; /* Jika ingin kotak input hitam */
    }
    ::placeholder { /* Untuk placeholder teks */
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

import os

from dotenv import load_dotenv
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
#from langchain_community.document_loaders import PyMuPDFLoader

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

st.markdown('<div class="title"> Dhq Chatbot Demo </div>', unsafe_allow_html=True)

# === RAG Retriever ===
persist_dir = "/workspace/Dhq_chatbot/embedding_pdfs"
ENCODER = "dragonkue/BGE-m3-ko"

embedding_model = HuggingFaceEmbeddings(model_name=ENCODER)
vectorstore = Chroma(
    persist_directory=persist_dir,
    collection_name="IBM_customer_cases_sample1",
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(search_kwars={"k": 2})

# === RAG -> Context ===

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

user_id = "chat_test"

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///./chat_history.db")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Initiate Chat history"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

llm = ChatOpenAI(
    model = "gpt-4o",
    temperature = 0.7,
    max_tokens = 16000,
)

system = SystemMessagePromptTemplate.from_template(
    """You are a helpful AI Assistant who answer user question based on teh provided context.
    Do not answer in more than {words} words."""
)

human_prompt = """Answer user question based on the porivded context. If you do not know the answer, say 'I don't know.'
    ### Context:
    {context}
    
    ### Question:
    {input}
    
    ### Answer: """

human = HumanMessagePromptTemplate.from_template(human_prompt)

messages = [system, MessagesPlaceholder(variable_name='history'), human]

prompt = ChatPromptTemplate(messages=messages)

chain = prompt | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key='input',history_messages_key='history')

def chat_with_llm(session_id, input):

    context = rag_to_context(input)

    for output in runnable_with_history.stream({'context': context, 'input': input, 'words':1000}, config={'configurable': {'session_id': session_id}}):

        yield output

st.markdown("""
<style> 
    .st-emotion-cache-128upt6 {
        background-color: transparent !important;
    }
            
    .st-emotion-cache-1flajlm{
        color: white        
    }
            
    .st-emotion-cache-1p2n2i4 {
        height: 500px 
    }
            
    .st-emotion-cache-8p7l3w {
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)


prompt = st.chat_input("What is up?")

if prompt:
    st.markdown("""
        <style>
            .st-emotion-cache-1p2n2i4 {
                height: unset !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(chat_with_llm(user_id, prompt))

    st.session_state.chat_history.append({"role": "assistant", "content": response})