import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith Tracking (Optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With GROQ"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Function to generate response
def generate_response(question, api_key, llm_model, temperature, max_tokens):
    try:
        llm = ChatGroq(
            model=llm_model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
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

api_key = st.sidebar.text_input(
    "Enter your Groq API Key",
    type="password"
)

# Model Selection
llm = st.sidebar.selectbox(
    "Select a Groq AI Model",
    [
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b"
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
if user_input and api_key:
    response = generate_response(
        user_input,
        api_key,
        llm,
        temperature,
        max_tokens
    )
    st.write(response)

elif user_input and not api_key:
    st.warning("Please enter the Groq API key in the sidebar")

else:
    st.write("Please provide the user input")