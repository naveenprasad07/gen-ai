from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith Tracking (Optional)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With GROQ"

# Prompt Chat Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Function to generate response
def generate_response(question, engine, temperature, max_tokens):
    try:
        llm = OllamaLLM(
            model=engine,
            temperature=temperature
        )

        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({"question": question})

        return answer

    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit App Title
st.title("Enhanced Q&A Chatbot with Groq")

# Sidebar Settings
st.sidebar.title("Settings")

# Model Selection
engine = st.sidebar.selectbox(
    "Select a Groq AI Model",
    [
        "llama3.1:latest",
        "phi3:latest"
    ]
)

# Model Parameters
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=50,
    max_value=300,
    value=150
)

# User Input
st.write("Go ahead and ask any question")

user_input = st.text_input("You:")

# Response Handling
if user_input:
    response = generate_response(
        user_input,
        engine,
        temperature,
        max_tokens
    )
    st.write(response)

else:
    st.write("Please provide the user input")