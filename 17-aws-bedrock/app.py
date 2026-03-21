import os
import boto3
import streamlit as st
import tempfile
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_community.llms import Bedrock

# ---------------- CONFIG ----------------
logging.basicConfig(level=logging.INFO)
AWS_REGION = "us-east-1"

# ---------------- BEDROCK CLIENT ----------------
@st.cache_resource
def get_bedrock_client():
    try:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        return client
    except Exception as e:
        st.error(f"❌ AWS Client Error: {e}")
        return None

bedrock_client = get_bedrock_client()

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def get_embeddings():
    return BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v2:0"
    )

# ---------------- LLM ----------------
@st.cache_resource
def get_llm():
    return Bedrock(
        client=bedrock_client,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={
            "max_tokens": 512,
            "temperature": 0.5
        }
    )

# ---------------- PROMPT ----------------
PROMPT_TEMPLATE = """
Human: Use the following context to answer the question.
Provide a detailed answer (minimum 200 words).
If unsure, say you don't know.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ---------------- PDF PROCESSING ----------------
def process_pdf(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = text_splitter.split_documents(documents)
        return docs

    except Exception as e:
        st.error(f"❌ PDF Processing Error: {e}")
        return []

# ---------------- VECTOR STORE ----------------
def create_vector_store(docs):
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"❌ Vector Store Error: {e}")
        return None

# ---------------- QA CHAIN ----------------
def get_qa_chain(vectorstore):
    try:
        llm = get_llm()

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        return qa

    except Exception as e:
        st.error(f"❌ QA Chain Error: {e}")
        return None

# ---------------- UI ----------------
def main():
    st.set_page_config(
        page_title="📄 Chat with PDF (AWS Bedrock)",
        layout="wide"
    )

    st.title("📄 Chat with PDF using AWS Bedrock")
    st.markdown("Upload a PDF and chat with it using Claude 3 🚀")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        if uploaded_file:
            if st.button("🔄 Process PDF"):
                with st.spinner("Processing PDF..."):
                    docs = process_pdf(uploaded_file)

                    if not docs:
                        st.error("No documents found!")
                        return

                    vectorstore = create_vector_store(docs)

                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.success("✅ PDF processed successfully!")

    # Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat Input
    user_input = st.chat_input("Ask a question about your PDF...")

    if user_input:
        if "vectorstore" not in st.session_state:
            st.warning("⚠️ Please upload and process a PDF first.")
            return

        qa_chain = get_qa_chain(st.session_state.vectorstore)

        if qa_chain:
            with st.spinner("Thinking..."):
                try:
                    response = qa_chain({"query": user_input})

                    answer = response["result"]

                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("assistant", answer))

                except Exception as e:
                    st.error(f"❌ Query Error: {e}")

    # Display Chat
    for role, msg in st.session_state.chat_history:
        if role == "user":
            with st.chat_message("user"):
                st.write(msg)
        else:
            with st.chat_message("assistant"):
                st.write(msg)


if __name__ == "__main__":
    main()