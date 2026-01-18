import streamlit as st
import uuid
from query_knowledge_graph import CVGraphRAGSystem

@st.cache_resource
def get_system():
    # Initialize the system
    system = CVGraphRAGSystem()
    return system

def render_chat():
    try:
        with st.spinner("Connecting to Knowledge Graph..."):
            system = get_system()
            if "connection_toast_shown" not in st.session_state:
                st.toast("Connected to Knowledge Graph!", icon="✅", duration=3)
                st.session_state.connection_toast_shown = True
    except Exception as e:
        st.error(f"Failed to connect to system: {e}")
        st.stop()

    st.markdown("Ask questions about candidates, skills, companies, and more based on the CV database.")
    

    
    # Add a button to start a new conversation in the sidebar
    with st.sidebar:
        if st.button("Start New Conversation", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = uuid.uuid4().hex
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize conversation ID
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = uuid.uuid4().hex

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "cypher" in message and message["cypher"]:
                with st.expander("View generated Cypher Query"):
                    st.code(message["cypher"], language="cypher")

    # Accept user input
    if prompt := st.chat_input("Ask a question (e.g., 'Who has Python skills?')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                try:
                    # Query the graph
                    response = system.query_graph(prompt, conversation_id=st.session_state.conversation_id)
                    
                    # Update conversation ID if new
                    if response.get("conversation_id"):
                        st.session_state.conversation_id = response["conversation_id"]
                    
                    answer = response.get("answer", "No answer found.")
                    cypher = response.get("cypher_query", "")
                    
                    message_placeholder.markdown(answer)
                    
                    if cypher:
                        with st.expander("View generated Cypher Query"):
                            st.code(cypher, language="cypher")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "cypher": cypher
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
