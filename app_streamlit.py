import streamlit as st
from utils import load_env
from graph import build_app_graph
import os

st.set_page_config(page_title="FAQ RAG Demo", page_icon="💬")
st.title("💬 Customer Support FAQ – RAG Demo")
st.caption("LangChain + LangGraph + LangSmith + Chroma + Streamlit")

cfg = load_env()

st.sidebar.subheader("Settings")
top_k = st.sidebar.slider("Retriever k", 1, 10, 4)
temperature = st.sidebar.slider("Generation temperature", 0.0, 1.0, 0.2, 0.1)

st.sidebar.markdown("---")
st.sidebar.write("**Data**")
st.sidebar.write("Dataset: `MakTek/Customer_support_faqs_dataset`")

if "graph" not in st.session_state:
    st.session_state.graph = build_app_graph()

# Chat UI
if "history" not in st.session_state:
    st.session_state.history = []

with st.form("ask"):
    user_q = st.text_input("Ask a question:", placeholder="e.g., How do I track my order?")
    submitted = st.form_submit_button("Ask")

if submitted and user_q.strip():
    with st.spinner("Thinking..."):
        state = {"question": user_q.strip(), "context_docs": [], "answer": None, "reason": None}
        out = st.session_state.graph.invoke(state)
        st.session_state.history.append(("user", user_q.strip()))
        st.session_state.history.append(("assistant", out["answer"]))

for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)