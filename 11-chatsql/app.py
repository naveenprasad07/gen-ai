import streamlit as st
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent, SQLDatabaseToolkit
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="Langchain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 Langchain: Chat with SQL DB")

# ---------------------------
# Database Options
# ---------------------------

LOCAL_DB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = [
    "Use SQLite3 Database - student.db",
    "Connect to your MySQL Database"
]

selected_opt = st.sidebar.radio(
    "Choose the DB which you want to chat",
    radio_opt
)

# ---------------------------
# Database Selection
# ---------------------------

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCAL_DB
    mysql_host = mysql_user = mysql_password = mysql_db = None

# ---------------------------
# API Key
# ---------------------------

api_key = st.sidebar.text_input("Groq API Key", type="password")

if not api_key:
    st.info("Please enter your Groq API Key")
    st.stop()

# ---------------------------
# LLM Model
# ---------------------------

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    streaming=True
)

# ---------------------------
# Configure Database
# ---------------------------

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):

    if db_uri == LOCAL_DB:

        db_file_path = (Path(__file__).parent / "student.db").absolute()

        creator = lambda: sqlite3.connect(
            f"file:{db_file_path}?mode=ro",
            uri=True
        )

        return SQLDatabase(create_engine("sqlite:///", creator=creator))

    elif db_uri == MYSQL:

        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details")
            st.stop()

        return SQLDatabase(
            create_engine(
                f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            )
        )

# ---------------------------
# Load Database
# ---------------------------

if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

# ---------------------------
# Toolkit + Agent
# ---------------------------

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# ---------------------------
# Chat Memory
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

if st.sidebar.button("Clear Message History"):
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# ---------------------------
# Display Chat History
# ---------------------------

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------------------
# User Input
# ---------------------------

user_query = st.chat_input("Ask anything about your database")

if user_query:

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):

        streamlit_callback = StreamlitCallbackHandler(st.container())

        response = agent.run(
            user_query,
            callbacks=[streamlit_callback]
        )

        st.write(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )