import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assisstant. Please respond to the question asked"),
        ("user","Question : {question}")
    ]
)

## streamlit framework
st.title("Langchain Demo with Gemma")
input_text = st.text_input("What question you have in mind?")

## Ollama LLama2 model
llm = OllamaLLM(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))