import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.agents import Tool, initialize_agent
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool


# ---------------------------
# Streamlit App Config
# ---------------------------
st.set_page_config(
    page_title="Math Problem Solver",
    page_icon="🧮"
)

st.title("🧮 Text to Math Problem Solver using Llama")

# ---------------------------
# Sidebar API Key
# ---------------------------
groq_api_key = st.sidebar.text_input("Enter your GROQ API KEY", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()

# ---------------------------
# LLM
# ---------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

# ---------------------------
# Wikipedia Tool
# ---------------------------
wikipedia = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="Useful for answering general knowledge questions."
)

# ---------------------------
# Python Calculator Tool
# ---------------------------
python_calculator = PythonREPLTool()

calculator_tool = Tool(
    name="Calculator",
    func=python_calculator.run,
    description="Useful for solving mathematical calculations."
)

# ---------------------------
# Reasoning Tool
# ---------------------------
reasoning_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are a helpful assistant that solves logic and math problems step by step.

Question:
{input}

Provide clear reasoning and the final answer.
"""
)

reasoning_chain = LLMChain(
    llm=llm,
    prompt=reasoning_prompt
)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_chain.run,
    description="Useful for solving logical reasoning or word problems."
)

# ---------------------------
# Tools List
# ---------------------------
tools = [
    wikipedia_tool,
    calculator_tool,
    reasoning_tool
]

# ---------------------------
# Agent
# ---------------------------
assistant_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# ---------------------------
# Chat History
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi 👋 I am a Math chatbot. Ask me any math question!"
        }
    ]

# Display chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------------------
# User Input
# ---------------------------
question = st.chat_input("Enter your math question")

if question:

    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.spinner("Solving..."):

        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )

        response = assistant_agent.run(
            question,
            callbacks=[st_cb]
        )

    # Add assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.chat_message("assistant").write(response)