__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from main import RAGSystem  # Assuming your class is in main.py

# Page Configuration
st.set_page_config(page_title="Agentic PDF Assistant", layout="wide")
st.title("📄 Technical Support AI Assistant")

# Initialize RAG System in Session State to prevent re-loading on every click
if "rag" not in st.session_state:
    st.session_state.rag = RAGSystem()
    # Check if DB already exists, if not, try to index default docs
    if not os.path.exists("./chroma_db"):
        with st.spinner("Initializing Knowledge Base..."):
            st.session_state.rag.load_and_index()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Management
with st.sidebar:
    st.header("Admin Controls")
    if st.button("Refresh Knowledge Base"):
        with st.spinner("Re-indexing documents..."):
            st.session_state.rag.load_and_index()
            st.success("Indexing complete!")

    st.divider()
    st.info("This assistant uses Gemini 2.5-Flash to answer questions based on your support documentation.")

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Logic
if prompt := st.chat_input("Ask a question about production incidents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            response = st.session_state.rag.ask(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})