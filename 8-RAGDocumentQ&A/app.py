import os
import streamlit as st
import time

from dotenv import load_dotenv
load_dotenv()

# Initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None

# LLM
from langchain_groq import ChatGroq

# Embeddings 
from langchain_community.embeddings import HuggingFaceEmbeddings

# RAG Components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Load Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="qwen/qwen3-32b"
)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based only on the provided context.

<context>
{context}
</context>

Question: {input}
"""
)

# Vector DB Creation
def create_vector_embedding():
    if st.session_state.vectors is None:

        # Embedding Model
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load PDFs
        st.session_state.loader = PyPDFDirectoryLoader("./docs")
        st.session_state.docs = st.session_state.loader.load()

        # Split documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )

        # Vector Store
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )


# UI
st.title("📄 Chat with Research Papers (Groq RAG)")

user_prompt = st.text_input(
    "Enter your query from the research paper",
    disabled=st.session_state.vectors is None
)

if st.button("Document Embedding"):
    create_vector_embedding()
    st.success("Vector Database is Ready ✅")
    st.rerun()


if user_prompt:

    if st.session_state.vectors is None:
        st.warning("Please click 'Document Embedding' first.")
        st.stop()

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.process_time()

    response = retrieval_chain.invoke({
        "input": user_prompt
    })

    st.write(response["answer"])

    # Show retrieved chunks
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------")