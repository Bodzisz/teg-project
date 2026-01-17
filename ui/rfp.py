import streamlit as st
import os
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
from datetime import datetime, timedelta

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

def render_rfp():
    """Render the RFP Explorer UI."""
    st.header("📂 RFP Explorer")
    st.markdown("Browse and analyze available Request for Proposals.")

    graph = get_graph_connection()
    if not graph:
        return

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
        
        ## TODO: Implement complete RFP (this will remove RFP from db after candidates are assigned to projects)
        ac2.button("Complete RFP", use_container_width=True, type="secondary")
