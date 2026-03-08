### RAG Q&A Conversation with pdf including chat history

import os
import streamlit as st
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="PDF Chat RAG", page_icon="📄")

st.title("📄 Conversational RAG with PDF uploads")
st.write("Upload PDFs and ask questions about them.")

# Sidebar settings
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Groq API Key", type="password")
    session_id = st.text_input("Session ID", value="default_session")

# ---------------- SESSION STATE ---------------- #

if "store" not in st.session_state:
    st.session_state.store = {}

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------------- API KEY CHECK ---------------- #

if api_key:

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="qwen/qwen3-32b"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    # ---------------- PDF PROCESSING ---------------- #

    if uploaded_files and st.session_state.vectorstore is None:

        documents = []

        with st.spinner("Processing PDFs..."):
            for uploaded_file in uploaded_files:

                temppdf = "./temp.pdf"

                with open(temppdf, "wb") as file:
                    file.write(uploaded_file.getvalue())

                loader = PyPDFLoader(temppdf)
                docs = loader.load()

                documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = text_splitter.split_documents(documents)

        st.session_state.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        st.success("Documents processed and stored in vector database")

    # ---------------- RETRIEVER ---------------- #

    if st.session_state.vectorstore:

        retriever = st.session_state.vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history "
            "formulate a standalone question that can be understood "
            "without the chat history."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        # ---------------- ANSWER PROMPT ---------------- #

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        # ---------------- CHAT HISTORY ---------------- #

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # ---------------- CHAT UI ---------------- #

        st.subheader("💬 Ask Questions")

        user_input = st.text_input("Your question")

        if user_input:

            session_history = get_session_history(session_id)

            with st.spinner("Thinking..."):

                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": session_id}
                    }
                )

            st.success("Assistant")
            st.write(response["answer"])

            st.write("Chat History:", session_history.messages)

    else:
        st.info("Please upload a PDF file to begin.")

else:
    st.warning("Please enter your Groq API key.")