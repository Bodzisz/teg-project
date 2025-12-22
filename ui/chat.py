import streamlit as st
import importlib.util
import sys
import os

# Load the CVGraphRAGSystem class dynamically because the filename starts with a number
# The file is in the root directory, so we need to go up one level from ui/
@st.cache_resource
def get_system():
    module_name = "3_query_knowledge_graph"
    # Get absolute path to the file in the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "3_query_knowledge_graph.py")
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Initialize the system
    system = module.CVGraphRAGSystem()
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
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

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
                    response = system.query_graph(prompt)
                    
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
