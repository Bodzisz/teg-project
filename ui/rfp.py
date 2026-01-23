import streamlit as st
import os
import tempfile
import shutil
import time
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
from datetime import datetime, timedelta
from parsers.rfp_parser import RFPParser
from save_data_proxy import SaveDataProxy
from pipeline_service import PipelineService

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


def simulate_allocations(selected_matches):
    """Simulate allocation split among selected matches without persisting.
    selected_matches: list of dicts with keys `person_id` and `availability` (0-100)
    Returns: (alloc_map, warnings)
    """
    warnings = []
    n = len(selected_matches)
    if n == 0:
        return {}, ["No candidates selected for preview."]

    # Normalize availability
    avail_map = {}
    for m in selected_matches:
        pid = m.get('person_id')
        try:
            avail = int(float(m.get('availability') or 0))
        except Exception:
            avail = 0
        avail_map[pid] = max(0, min(100, avail))

    desired = round(100 / n)
    alloc = {pid: 0 for pid in avail_map}

    # First pass
    for pid in alloc:
        alloc[pid] = min(desired, avail_map[pid])

    remaining = 100 - sum(alloc.values())

    # Round-robin distribute remaining by 1% to those with capacity
    if remaining > 0:
        pids_with_capacity = [pid for pid in alloc if avail_map[pid] - alloc[pid] > 0]
        if not pids_with_capacity:
            warnings.append("Selected candidates do not have enough combined availability to reach 100%.")
        else:
            i = 0
            while remaining > 0 and pids_with_capacity:
                pid = pids_with_capacity[i % len(pids_with_capacity)]
                if avail_map[pid] - alloc[pid] > 0:
                    alloc[pid] += 1
                    remaining -= 1
                else:
                    # remove from capacity list
                    pids_with_capacity = [p for p in pids_with_capacity if avail_map[p] - alloc[p] > 0]
                i += 1

    # Generate warnings for zero allocations
    for pid in alloc:
        if alloc[pid] == 0:
            warnings.append(f"Candidate {pid} has no available capacity and will not be allocated.")

    return alloc, warnings

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

    save_proxy = SaveDataProxy()
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

                # Use proxy to save to both systems
                if save_proxy.save_rfp(file_path):
                    success_count += 1
                else:
                    errors.append(f"{uploaded_file.name}: Failed to process/save.")
            
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

    # Prepare list for selectbox (show entity_id and title)
    rfp_options = {
        f"{item['r'].get('entity_id','')} — {item['r'].get('title','Untitled RFP')} ({item['r'].get('client','')})": item
        for item in rfp_data
    }
    
    selected_option = st.selectbox(
        "Select an RFP to view details:", 
        options=list(rfp_options.keys()),
        index=None,
        placeholder="Type to search..."
    )

    # Ensure selected candidates list in session
    if 'selected_candidates' not in st.session_state:
        st.session_state['selected_candidates'] = []

    if selected_option:
        data = rfp_options[selected_option]
        rfp = data['r']
        
        # Reset matches if RFP changed
        if st.session_state.get('last_viewed_rfp_id') != rfp.get('entity_id'):
            st.session_state['current_matches'] = []
            st.session_state['match_rfp_id'] = None
            st.session_state['last_viewed_rfp_id'] = rfp.get('entity_id')

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
        
        # Action Buttons (Find / Preview / Assign)
        st.markdown("### 🚀 Actions")
        ac1, ac2 = st.columns([1,1])

        if ac1.button("Find Matching Candidates", use_container_width=True, type="primary"):
            with st.spinner("Finding best matches..."):
                try:
                    from matching_engine import MatchingEngine
                    engine = MatchingEngine()
                    lookup_id = rfp.get('entity_id') or rfp.get('title')
                    matches = engine.rank_candidates(lookup_id, top_n=10)
                    st.session_state['current_matches'] = matches
                    st.session_state['match_rfp_id'] = rfp.get('entity_id')
                    # clear previous selections
                    st.session_state['selected_candidates'] = []
                except Exception as e:
                    st.error(f"Error finding matches: {e}")

        # (Removed UI force-assign toggle: assignments from checkboxes are deterministic)

        # Display matches if they exist for the current RFP
        if st.session_state.get('match_rfp_id') == rfp.get('entity_id') and 'current_matches' in st.session_state:
            st.markdown("### 🏆 Top Candidates")
            matches = st.session_state['current_matches'] or []
            if matches:
                for idx, match in enumerate(matches):
                    score = match.get('score', 0)
                    person_id = match.get('person_id') or match.get('id') or match.get('name') or f"person_{idx}"
                    person_name = match.get('display_name') or match.get('name') or person_id
                    mandatory_icon = "✅" if match.get('mandatory_met') else "⚠️"
                    header = f"{person_name} — Score: {score:.1f} {mandatory_icon}"

                    # Checkbox for selection (disabled if no availability)
                    avail = 0
                    try:
                        avail = int(float(match.get('availability') or 0))
                    except Exception:
                        avail = 0

                    chk_key = f"chk_{idx}_{rfp.get('entity_id')}"
                    checked = st.checkbox(f"Select {person_name} | Availability: {avail}% | Score: {score:.1f}",
                                          value=(person_id in st.session_state['selected_candidates']),
                                          key=chk_key,
                                          disabled=(avail <= 0))

                    # Update session_state selected list
                    if checked and person_id not in st.session_state['selected_candidates']:
                        st.session_state['selected_candidates'].append(person_id)
                    if not checked and person_id in st.session_state['selected_candidates']:
                        st.session_state['selected_candidates'].remove(person_id)

                    with st.expander(header):
                        dc1, dc2 = st.columns(2)
                        dc1.markdown(f"**📍 Location:** {match.get('location', 'N/A')}")
                        dc1.markdown(f"**📧 Email:** {match.get('email', 'N/A')}")
                        dc1.markdown(f"**📅 Experience:** {match.get('years_experience', 0)} years")
                        dc2.metric("Availability", f"{avail}%")
                        st.markdown("**📝 Bio:**")
                        st.write(match.get('description') or "No description available.")

                        if match.get('skills'):
                            st.markdown("**🛠 Skills:**")
                            required_skill_names = {req.get('skill') for req in requirements if req.get('skill')}
                            skills_html = ""
                            for s in match.get('skills', []):
                                skill_name = s.get('name')
                                is_match = skill_name in required_skill_names
                                bg_color = "#2e7d32" if is_match else "#333"
                                skills_html += f"<span style='background-color: {bg_color}; padding: 4px 8px; border-radius: 4px; margin-right: 5px; font-size: 0.8em;'>{skill_name} ({s.get('proficiency', 'N/A')})</span>"
                            st.markdown(skills_html, unsafe_allow_html=True)
                        else:
                            st.info("No skills listed.")

                # Preview and assign controls
                st.markdown("---")
                prc, asc = st.columns([1,1])

                if prc.button("Preview Allocation for Selected", use_container_width=True):
                    selected_ids = st.session_state.get('selected_candidates', [])
                    if not selected_ids:
                        st.warning("No candidates selected for preview.")
                    else:
                        # Build selected_matches list from current matches
                        selected_matches = [m for m in matches if (m.get('person_id') or m.get('id') or m.get('name')) in selected_ids]
                        alloc_map, warnings = simulate_allocations(selected_matches)
                        # Display allocations
                        alloc_table = [{"person_id": pid, "proposed_allocation": alloc} for pid, alloc in alloc_map.items()]
                        st.table(alloc_table)
                        if warnings:
                            with st.expander("Warnings", expanded=True):
                                for w in warnings:
                                    st.warning(w)

                if asc.button("Assign Selected Candidates", use_container_width=True, type="primary"):
                    selected_ids = st.session_state.get('selected_candidates', [])
                    if not selected_ids:
                        st.error("No candidates selected. Please select at least one candidate to assign.")
                    else:
                        with st.spinner("Assigning selected candidates..."):
                            try:
                                service = PipelineService()
                                rfp_id_for_assign = rfp.get('entity_id') or rfp.get('title')
                                # Always force assignment from UI selection (do not rely on assignment_probability)
                                result = service.assign_selected_candidates_for_rfp(rfp_id_for_assign, selected_ids, force=True)
                                st.success("Assignments saved.")
                                # Show returned assignments if present
                                if isinstance(result, dict) and result.get('assignments'):
                                    st.table(result.get('assignments'))
                                # Refresh matches to reflect updated availability
                                try:
                                    from matching_engine import MatchingEngine
                                    engine = MatchingEngine()
                                    new_matches = engine.rank_candidates(rfp.get('entity_id') or rfp.get('title'), top_n=10)
                                    st.session_state['current_matches'] = new_matches
                                    st.session_state['selected_candidates'] = []
                                except Exception:
                                    pass
                            except Exception as e:
                                st.error(f"Error during assignment: {e}")
            else:
                st.info("No matching candidates found.")
        
        if st.button("Complete RFP", use_container_width=True, type="secondary"):
            if delete_rfp(graph, rfp.get('entity_id')):
                st.toast(f"✅ RFP '{rfp.get('title')}' completed and removed.", icon="✅")
                time.sleep(1) # Allow toast to appear
                st.rerun()
