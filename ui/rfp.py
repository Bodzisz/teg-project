import streamlit as st
import os
import tempfile
import shutil
import time
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
from datetime import datetime, timedelta
from parsers.rfp_parser import RFPParser

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

def delete_rfp(graph, rfp_id):
    """Delete an RFP and its relationships from the database."""
    query = """
    MATCH (r:RFP {entity_id: $rfp_id})
    DETACH DELETE r
    """
    try:
        graph.query(query, {"rfp_id": rfp_id})
        return True
    except Exception as e:
        st.error(f"Error deleting RFP: {e}")
        return False

def fetch_rfps(graph):
    """Fetch all RFP nodes with their details and requirements."""
    query = """
    MATCH (r:RFP)
    OPTIONAL MATCH (r)-[req:NEEDS]->(s:Skill)
    RETURN r, 
           collect({
               skill: s.name, 
               proficiency: req.min_proficiency,
               mandatory: req.is_mandatory,
               certifications: req.preferred_certifications
           }) as requirements
    ORDER BY r.title
    """
    try:
        return graph.query(query)
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return []

def calculate_end_date(start_date_str, duration_months):
    """Calculate end date based on start date and duration."""
    try:
        if not start_date_str or not duration_months:
            return "N/A"
        
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        # Approximate 30 days per month
        end_date = start_date + timedelta(days=int(duration_months) * 30)
        return end_date.strftime("%Y-%m-%d")
    except Exception:
        return "N/A"

def process_uploaded_files(uploaded_files):
    """Process uploaded RFP PDF files."""
    if not uploaded_files:
        return

    if len(uploaded_files) > 10:
        st.error(f"⚠️ Limit exceeded: You uploaded {len(uploaded_files)} files. Please upload a maximum of 10 files.")
        return

    rfp_parser = RFPParser()
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

                # Extract text
                text = rfp_parser.extract_text_from_pdf(file_path)
                if not text:
                    errors.append(f"{uploaded_file.name}: No text extracted.")
                    continue

                # Parse and Save
                rfp_data = rfp_parser.parse_rfp(text)
                rfp_parser.save_to_neo4j(rfp_data)
                
                success_count += 1
            
            except Exception as e:
                errors.append(f"{uploaded_file.name}: {str(e)}")
            
            finally:
                progress_bar.progress((i + 1) / total_files)

    status_text.empty()
    progress_bar.empty()

    if success_count > 0:
        # Success message is handled in render_rfp via session state
        st.toast(f"Processed {success_count} RFPs", icon="✅")
    
    if errors:
        with st.expander("⚠️ Processing Errors", expanded=True):
            for err in errors:
                st.error(err)
        
    return success_count > 0



def render_rfp():
    """Render the RFP Explorer UI."""
    # Initialize session state for uploader key and success flag
    if "rfp_uploader_key" not in st.session_state:
        st.session_state["rfp_uploader_key"] = 0

    if "rfp_upload_success" not in st.session_state:
        st.session_state["rfp_upload_success"] = False
    st.header("📂 RFP Explorer")
    st.markdown("Browse and analyze available Request for Proposals.")

    # Show success message if flag is set, then clear it
    if st.session_state.get("rfp_upload_success"):
        st.success("✅ Files successfully processed and saved!")
        st.session_state["rfp_upload_success"] = False

    graph = get_graph_connection()
    if not graph:
        return

    # --- File Upload Section ---
    # Use session state to control 'expanded' would be complex, but defaulting to False
    # combined with st.rerun() works because rerun resets the widget state unless persisted.
    # We want it closed after success (rerun), so expanded=False is correct.
    with st.expander("📤 Upload New RFPs", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload RFP PDFs (Max 10 files)", 
            type=["pdf"], 
            accept_multiple_files=True,
            key=f"rfp_uploader_{st.session_state['rfp_uploader_key']}"
        )
        
        if uploaded_files:
             if st.button("Process Files", type="primary"):
                if process_uploaded_files(uploaded_files):
                    st.session_state["rfp_uploader_key"] += 1
                    st.session_state["rfp_upload_success"] = True
                    st.rerun()

    # Fetch RFPs
    rfp_data = fetch_rfps(graph)
    
    if not rfp_data:
        st.info("No RFPs found in the database.")
        return

    # Prepare list for selectbox
    rfp_options = {
        f"{item['r']['title']} ({item['r']['client']})": item 
        for item in rfp_data
    }
    
    selected_option = st.selectbox(
        "Select an RFP to view details:", 
        options=list(rfp_options.keys()),
        index=None,
        placeholder="Type to search..."
    )

    if selected_option:
        data = rfp_options[selected_option]
        rfp = data['r']
        requirements = data['requirements']
        
        # Sort requirements: Mandatory first
        if requirements:
            requirements.sort(key=lambda x: x.get('mandatory', False), reverse=True)

        # Calculate dates
        start_date = rfp.get('start_date', 'N/A')
        duration = rfp.get('duration_months', 0)
        end_date = calculate_end_date(start_date, duration)

        # Title and Client Header
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(rfp.get('title', 'Untitled RFP'))
            st.caption(f"Client: **{rfp.get('client', 'Unknown')}** • Location: {rfp.get('location', 'N/A')}")
        with col2:
             st.metric("Budget", rfp.get('budget_range', 'N/A'))

        # Main Description
        with st.container(border=True):
            st.markdown("### 📝 Description")
            st.write(rfp.get('description', 'No description available.'))

        # Project Details
        st.markdown("### 📊 Project Details")
        c1, c2, c3, c4 = st.columns(4)
        c1.info(f"**Type:**\n{rfp.get('project_type', 'N/A')}")
        c2.info(f"**Timeline:**\n{start_date} ➔ {end_date}")
        c3.info(f"**Team Size:**\n{rfp.get('team_size', 'N/A')}")
        c4.info(f"**Remote:**\n{'✅ Yes' if rfp.get('remote_allowed') else '❌ No'}")

        # Requirements
        st.markdown("### 🎯 Skill Requirements")
        if requirements and requirements[0].get('skill'): # Check if any skills exist
            for req in requirements:
                is_mandatory = req.get('mandatory', False)
                md_icon = "🔴" if is_mandatory else "🔵"
                md_text = "Mandatory" if is_mandatory else "Optional"
                
                with st.expander(f"{md_icon} **{req['skill']}** ({md_text})"):
                    st.write(f"**Min Proficiency:** {req['proficiency']}/5")
                    if req['certifications']:
                        st.write(f"**Preferred Certifications:** {', '.join(req['certifications'])}")
        else:
            st.info("No specific skill requirements listed.")
        
        # Action Buttons (Mockup)
        st.markdown("### 🚀 Actions")
        ac1, ac2 = st.columns(2)

        ## TODO: Implement matching candidates
        ac1.button("Find Matching Candidates", use_container_width=True, type="primary")
        
        if ac2.button("Complete RFP", use_container_width=True, type="secondary"):
            if delete_rfp(graph, rfp.get('entity_id')):
                st.toast(f"✅ RFP '{rfp.get('title')}' completed and removed.", icon="✅")
                time.sleep(1) # Allow toast to appear
                st.rerun()
