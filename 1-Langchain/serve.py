from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)

# 1. Create Prompt template


system_template = "Translate the folowing into {language}:"
promp_template = ChatPromptTemplate.from_messages([
    ("system",system_template),
    ("user","{text}")
])

parser = StrOutputParser()

## create chain

chain = promp_template | model | parser

## App definition

app = FastAPI(title="Langchain Server",
              version="1.0",
              description="A simple api server using Langchain runnable interfaces")


# Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)