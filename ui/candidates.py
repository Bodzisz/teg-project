import streamlit as st
import os
import tempfile
import asyncio
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
from cv_knowledge_graph_builder import DataKnowledgeGraphBuilder

# Load environment variables
load_dotenv(override=True)

def get_graph_connection():
    """Establish connection to Neo4j."""
    try:
        graph = Neo4jGraph(
            url="bolt://localhost:7687",
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        return graph
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return None

def fetch_candidates(graph):
    """Fetch all Person nodes with their details and skills."""
    query = """
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[r:HAS_SKILL]->(s:Skill)
    RETURN p, 
           collect({
               name: s.name, 
               proficiency: r.proficiency
           }) as skills
    ORDER BY p.name
    """
    try:
        return graph.query(query)
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return []

def process_uploaded_cvs(uploaded_files):
    """Process uploaded CV PDF files."""
    if not uploaded_files:
        return

    builder = DataKnowledgeGraphBuilder(clear_graph=False) # Don't clear existing data
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    success_count = 0
    errors = []

    with tempfile.TemporaryDirectory() as temp_dir:
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name} ({i+1}/{total_files})...")
            
            try:
                # Save uploaded file to temp directory
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Convert CV to graph (async wrapper)
                # Since Streamlit is sync, we run the async method here
                graph_documents = asyncio.run(builder.convert_cv_to_graph(file_path))
                
                if graph_documents:
                   builder.store_graph_documents(graph_documents)
                   success_count += 1
                else:
                     errors.append(f"{uploaded_file.name}: Failed to extract graph data.")

            except Exception as e:
                errors.append(f"{uploaded_file.name}: {str(e)}")
            
            finally:
                progress_bar.progress((i + 1) / total_files)

    status_text.empty()
    progress_bar.empty()

    if success_count > 0:
        st.toast(f"Processed {success_count} CVs", icon="✅")
    
    if errors:
        with st.expander("⚠️ Processing Errors", expanded=True):
            for err in errors:
                st.error(err)
        
    return success_count > 0

def render_candidates():
    """Render the Candidates Explorer UI."""
    # Initialize session state for uploader key and success flag if not exists
    if "candidates_uploader_key" not in st.session_state:
        st.session_state["candidates_uploader_key"] = 0

    if "candidates_upload_success" not in st.session_state:
        st.session_state["candidates_upload_success"] = False

    st.header("👥 Candidates Explorer")
    st.markdown(" Browse candidates and manage the talent pool.")

    # Show success message if flag is set, then clear it
    if st.session_state.get("candidates_upload_success"):
        st.success("✅ CVs successfully processed and saved!")
        st.session_state["candidates_upload_success"] = False

    graph = get_graph_connection()
    if not graph:
        return

    # --- File Upload Section ---
    with st.expander("📤 Upload New CVs", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload CV PDFs", 
            type=["pdf"], 
            accept_multiple_files=True,
            key=f"candidates_uploader_{st.session_state['candidates_uploader_key']}"
        )
        
        if uploaded_files:
             if st.button("Process CVs", type="primary"):
                if process_uploaded_cvs(uploaded_files):
                    st.session_state["candidates_uploader_key"] += 1
                    st.session_state["candidates_upload_success"] = True
                    st.rerun()

    # Fetch candidates
    candidates_data = fetch_candidates(graph)
    
    if not candidates_data:
        st.info("No candidates found in the database. Upload some CVs to get started!")
        return

    # Prepare list for selectbox
    # Use name (email) as unique identifier for display if possible, or just name
    candidates_options = {
        f"{item['p'].get('name', 'Unknown')} ({item['p'].get('email', 'No Email')})": item 
        for item in candidates_data
    }
    
    selected_option = st.selectbox(
        "Select a person to view details:", 
        options=list(candidates_options.keys()),
        index=None,
        placeholder="Type to search..."
    )

    if selected_option:
        data = candidates_options[selected_option]
        person = data['p']
        skills = data['skills']
        
        # Header
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(person.get('name', 'Unknown Person'))
            location = person.get('location', 'N/A')
            st.caption(f"**📍 Location:** {location} • **📧 Email:** {person.get('email', 'N/A')}")
        
        with col2:
             # Just a placeholder metric or could be something relevant like experience
             years_exp = person.get('years_experience')
             if years_exp:
                 st.metric("Experience", f"{years_exp} Years")

        # Bio / Description
        if person.get('description') or person.get('bio'):
             with st.container(border=True):
                st.markdown("### 📝 Bio")
                st.write(person.get('description') or person.get('bio'))

        # Details Columns
        st.markdown("### 👤 Details")
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Phone:**\n{person.get('phone', 'N/A')}")
        # Add more fields if available in your schema
        c2.info(f"**Role:**\n{person.get('role', 'N/A')}") 
        c3.info(f"**Availability:**\n{person.get('availability', 'N/A')}")

        # Skills
        st.markdown("### 🛠 Skills")
        if skills and skills[0].get('name'):
            skills_html = ""
            for s in skills:
                name = s.get('name')
                proficiency = s.get('proficiency')
                prof_str = f" ({proficiency})" if proficiency else ""
                
                # Simple style
                skills_html += f"<span style='background-color: #333; padding: 4px 8px; border-radius: 4px; margin-right: 5px; font-size: 0.9em; display: inline-block; margin-bottom: 5px;'>{name}{prof_str}</span>"
            
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.info("No specific skills listed.")
