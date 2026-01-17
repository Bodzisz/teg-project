import streamlit as st
from ui.chat import render_chat
from ui.rfp import render_rfp

# Set page config
st.set_page_config(
    page_title="Elite Match AI",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatMessage.user {
        background-color: #2b313e;
    }
    .stChatMessage.assistant {
        background-color: #1e2329;
    }
    .cypher-query {
        font-family: monospace;
        background-color: #0e1117;
        padding: 10px;
        border-radius: 5px;
        font-size: 0.9em;
        color: #00ff00;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 Elite Match AI")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("", ["Chat", "RFPs"])
    st.markdown("---")
    st.markdown("### About")
    st.info("This AI assistant helps you query the CV Knowledge Graph to find the best candidates.")

if page == "Chat":
    render_chat()
elif page == "RFPs":
    render_rfp()
