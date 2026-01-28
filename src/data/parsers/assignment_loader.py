import argparse
import json
import logging
import random
import tomllib
import yaml
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any
from src.data.models.project_models import ProjectData, Requirement
from src.data.models.programmer_models import ProgrammerData, Skill
from langchain_neo4j import Neo4jGraph
import random

# Load environment variables
load_dotenv(override=True)

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AssignmentLoader")


class ConfigLoader:
    """Load configuration from TOML file."""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            logger.error("❌ Failed to load config file: %s", e)
            raise

    def get(self, section: str, default=None):
        """Retrieve a section from the configuration."""
        return self.config.get(section, default)


class AssignmentLoader:
    def __init__(self, config_path: str):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        self.graph = Neo4jGraph()
        logger.info("✅ AssignmentLoader initialized.")

    def load_projects(self, project_file: str) -> List[ProjectData]:
        """Load projects from JSON or YAML file."""
        try:
            with open(project_file, "r", encoding="utf-8") as f:
                if project_file.endswith(('.yaml', '.yml')):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            projects = []
            for item in data:
                # Handle potential difference in structure if necessary
                # YAML currently matches JSON structure mostly
                projects.append(ProjectData(**item))
                
            logger.info(f"✅ Loaded {len(projects)} projects from {project_file}")
            return projects
        except Exception as e:
            logger.error("❌ Failed to load projects: %s", e)
            raise

    def load_programmers_from_json(self, programmers_file: str) -> List[ProgrammerData]:
        """Load and validate programmers from JSON file."""
        try:
            with open(programmers_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            programmers = [ProgrammerData(**item) for item in data]
            logger.info(f"✅ Loaded {len(programmers)} programmers from {programmers_file}")
            return programmers
        except Exception as e:
            logger.error("❌ Failed to load programmers: %s", e)
            raise

    def load_programmers_from_graph(self) -> List[ProgrammerData]:
        """Fetch programmers from Neo4j."""
        query = """
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[hs:HAS_SKILL]->(s:Skill)
        RETURN p.name AS name, p.email AS email, p.location AS location,
               p.years_experience AS years_experience, p.availability AS availability,
               collect({name: s.name, proficiency: hs.proficiency}) AS skills
        """
        results = self.graph.query(query)
        programmers = []
        for result in results:
            if not result["name"]:
                continue
            skills = [
                Skill(name=skill["name"], proficiency=skill["proficiency"] or "Intermediate")
                for skill in result["skills"] if skill["name"] is not None
            ]
            programmer = ProgrammerData(
                id=result["name"],
                name=result["name"],
                email=result["email"],
                location=result["location"],
                years_experience=result.get("years_experience", 0),
                availability=result.get("availability", 100.0),
                skills=skills
            )
            programmers.append(programmer)
        logger.info(f"✅ Loaded {len(programmers)} programmers from graph")
        return programmers

    def load_projects_from_graph(self) -> List[ProjectData]:
        """Load projects from Neo4j graph."""
        query = """
        MATCH (p:Project)
        OPTIONAL MATCH (p)-[r:REQUIRES]->(s:Skill)
        WITH p, collect(DISTINCT {skill_name: s.name, min_proficiency: r.min_proficiency, is_mandatory: r.is_mandatory}) AS requirements
        RETURN p.entity_id AS id, p.name AS name, p.client AS client, p.description AS description,
               p.start_date AS start_date, p.end_date AS end_date,
               p.estimated_duration_months AS estimated_duration_months,
               p.budget AS budget, p.status AS status, p.team_size AS team_size,
               requirements
        """
        results = self.graph.query(query)
        projects = []
        for result in results:
            if not result["id"] or not result["name"] or not result["client"] or not result["start_date"]:
                # logger.warning(f"Skipping project with missing essential data...") 
                continue

            start_date = None
            if result["start_date"]:
                if isinstance(result["start_date"], str):
                    start_date = datetime.fromisoformat(result["start_date"]).date()
                else:
                    start_date = result["start_date"]

            end_date = None
            if result["end_date"]:
                if isinstance(result["end_date"], str):
                    end_date = datetime.fromisoformat(result["end_date"]).date()
                else:
                    end_date = result["end_date"]

            requirements = [
                Requirement(**req) for req in result["requirements"]
                if req["skill_name"] is not None
            ]

            project = ProjectData(
                id=result["id"],
                name=result["name"],
                client=result["client"],
                description=result["description"] or "No description available",
                start_date=start_date,
                end_date=end_date,
                estimated_duration_months=result["estimated_duration_months"] or 6,
                budget=result["budget"],
                status=result["status"] or "planned",
                team_size=result["team_size"] or 1,
                requirements=requirements
            )
            projects.append(project)
        logger.info(f"✅ Loaded {len(projects)} projects from graph")
        return projects

    def calculate_availability(self, person_id: str) -> int:
        """
        Calculates programmer availability based on ASSIGNED_TO relationships in Neo4j.
        """
        query = """
        MATCH (p:Person {name: $person_id})-[r:ASSIGNED_TO]->(:Project)
        RETURN sum(coalesce(r.allocation_percentage, 0)) AS allocated
        """
        result = self.graph.query(query, {"person_id": person_id})
        allocated = result[0]["allocated"] or 0
        availability = max(0, 100 - allocated)
        logger.info(f"✅ Availability for {person_id}: {availability}%")
        return availability

    def update_graph_with_availability(self, person_id: str, availability: int):
        """
        Updates the Person node in Neo4j with the availability property.
        """
        query = """
        MATCH (p:Person {name: $person_id})
        SET p.availability = $availability
        """
        self.graph.query(query, {"person_id": person_id, "availability": availability})
        logger.info(f"✅ Updated graph availability: {person_id} -> {availability}%")

    def load_assignments_from_yaml(self, yaml_file: str) -> List[dict]:
        """
        Loads allocations from a YAML file that mirrors the projects.json structure.
        Returns a flattened list of assignments.
        """
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                projects = yaml.safe_load(f)
            
            assignments = []
            if not isinstance(projects, list):
                logger.warning(f"YAML file {yaml_file} does not contain a list of projects.")
                return assignments

            for project in projects:
                project_name = project.get("name")
                for assigned in project.get("assigned_programmers", []):
                    assignments.append({
                        "project_id": project_name,
                        "programmer_id": assigned.get("programmer_name") or assigned.get("programmer_id"),
                        "end_date": assigned.get("assignment_end_date"),
                        "allocation": assigned.get("allocation_percentage", 100)
                    })
            
            logger.info(f"✅ Loaded {len(assignments)} allocations from {yaml_file}")
            return assignments
        except Exception as e:
            logger.error(f"❌ Error loading YAML file: {e}")
            raise


    def assign_candidates_to_single_project(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Assigns programmers to a single project based on RFP matches.
        """
        # Load the project
        project_query = """
        MATCH (p:Project)
        WHERE p.id = $project_id OR p.entity_id = $project_id
        RETURN p.id as id, p.name as name, p.start_date as start_date, p.end_date as end_date, p.team_size as team_size
        """
        project_result = self.graph.query(project_query, {"project_id": project_id})
        if not project_result:
            logger.warning(f"Project {project_id} not found")
            return []
        project = project_result[0]

        assignments = []

        # Find the RFP that generated this project
        rfp_query = """
        MATCH (r:RFP)-[:GENERATES]->(p:Project)
        WHERE p.id = $project_id OR p.entity_id = $project_id
        RETURN r.id as rfp_id, r.title as rfp_title
        """
        rfp_result = self.graph.query(rfp_query, {"project_id": project_id})
        if not rfp_result:
            logger.warning(f"No originating RFP found for project {project_id}")
            return []
        rfp_id = rfp_result[0]["rfp_id"]
        logger.info(f"Found originating RFP: {rfp_id} for project {project_id}")

        # Retrieve top-ranked candidates from MATCHED_TO for this RFP
        match_query = """
        MATCH (person:Person)-[m:MATCHED_TO]->(r:RFP {id: $rfp_id})
        RETURN person.name as person_name, person.name as person_id, m.score as score, m.mandatory_met as mandatory_met
        ORDER BY m.score DESC
        """
        candidates = self.graph.query(match_query, {"rfp_id": rfp_id})

        if not candidates:
            logger.warning(f"No matched candidates found for RFP {rfp_id} (project {project_id})")
            return []

        # Check programmer availability (relaxed threshold)
        available_candidates = []
        for candidate in candidates:
            person_id = candidate["person_id"]
            availability = self.calculate_availability(person_id)
            if availability > 10:  # Relaxed threshold
                available_candidates.append({
                    "person_id": person_id,
                    "person_name": candidate["person_name"],
                    "score": candidate["score"],
                    "mandatory_met": candidate.get("mandatory_met", False),
                    "availability": availability
                })
            else:
                logger.info(f"Candidate {candidate['person_name']} excluded due to low availability ({availability}%)")

        if not available_candidates:
            logger.warning(f"No available candidates for project {project_id} (RFP {rfp_id})")
            return []

        # Prioritize candidates who met mandatory requirements
        mandatory_candidates = [c for c in available_candidates if c["mandatory_met"]]
        if mandatory_candidates:
            available_candidates = mandatory_candidates
            logger.info(f"Prioritizing {len(mandatory_candidates)} candidates who met mandatory requirements for project {project_id}")

        # Take top candidates up to team_size
        top_candidates = available_candidates[:project["team_size"]]

        if not top_candidates:
            logger.warning(f"No top candidates selected for project {project_id} (team_size: {project['team_size']})")
            return []

        # Distribute allocation dynamically based on availability
        total_available = sum(c["availability"] for c in top_candidates)
        for candidate in top_candidates:
            if total_available > 0:
                allocation_percentage = min(candidate["availability"], (candidate["availability"] / total_available) * 100)
            else:
                allocation_percentage = 100 // len(top_candidates)

            logger.info(f"Assigned {candidate['person_name']} to {project['name']} with {allocation_percentage:.1f}% allocation")

            assignments.append({
                "person_name": candidate["person_name"],
                "project_title": project["name"],
                "allocation_percentage": round(allocation_percentage, 1),
                "start_date": str(project["start_date"]),
                "end_date": str(project["end_date"]) if project["end_date"] else None
            })

            # Update availability in real-time
            new_availability = candidate["availability"] - allocation_percentage
            self.update_graph_with_availability(candidate["person_id"], max(0, int(new_availability)))

        logger.info(f"✅ Completed assignments for project {project_id}, created {len(assignments)} assignments")
        return assignments


    def save_assignments_to_neo4j(self, assignments_summary: List[Dict[str, Any]]):
        """
        Saves the assignments to Neo4j by creating ASSIGNED_TO relationships.
        """
        try:
            for assignment in assignments_summary:
                person_name = assignment["person_name"]
                project_title = assignment["project_title"]
                allocation_percentage = assignment["allocation_percentage"]
                start_date = assignment["start_date"]
                end_date = assignment["end_date"]
                # Robust Cypher: match project by id, entity_id or name
                query = """
                MATCH (p:Person {name: $person_name})
                MATCH (pr:Project)
                WHERE pr.id = $project_title OR pr.entity_id = $project_title OR pr.name = $project_title
                MERGE (p)-[a:ASSIGNED_TO]->(pr)
                SET a.allocation_percentage = $allocation_percentage
                """
                params = {
                    "person_name": person_name,
                    "project_title": project_title,
                    "allocation_percentage": allocation_percentage,
                }

                # Only set dates if provided and non-empty
                if start_date:
                    query += "\nSET a.start_date = date($start_date)"
                    params["start_date"] = start_date
                if end_date:
                    query += "\nSET a.end_date = date($end_date)"
                    params["end_date"] = end_date

                self.graph.query(query, params)
                logger.info(f"✅ Created ASSIGNED_TO for {person_name} -> {project_title} ({allocation_percentage}%)")
            
            logger.info(f"✅ Saved {len(assignments_summary)} assignments to Neo4j.")
        except Exception as e:
            logger.error(f"❌ Error saving assignments to Neo4j: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate availability and assign programmers.")
    parser.add_argument("--config", type=str, default="config/config.toml", help="Path to the configuration file.")
    parser.add_argument("--projects", type=str, default="data/projects/projects.json", help="Path to the projects file.")
    parser.add_argument("--assignments", type=str, help="Optional YAML file with allocations (offline mode).")
    args = parser.parse_args()

    loader = AssignmentLoader(config_path=args.config)

    # Calculate availability
    if args.assignments:
        # Offline mode with YAML
        assignments_yaml = loader.load_assignments_from_yaml(args.assignments)
        
        # Aggregate allocations per person
        person_allocations = {}
        for a in assignments_yaml:
            pid = a["programmer_id"]
            person_allocations[pid] = person_allocations.get(pid, 0) + a.get("allocation", 0)
            
        for person_id, total_allocation in person_allocations.items():
            availability = max(0, 100 - total_allocation)
            loader.update_graph_with_availability(person_id, availability)
            
        # Also save these YAML assignments to Neo4j if desired, referencing the method simply:
        # loader.save_to_neo4j(assignments_yaml) 
        # But the original code only updated availability in this block. 
        # User request implies "rely on assignment_loader... structure in methods related to yaml".
        # Let's save them too, as usually loading assignments implies wanting them in the DB.
        loader.save_to_neo4j(assignments_yaml)

    else:
        # Neo4j mode: calculate availability for all programmers
        # We need to load programmers first
        programmers_for_availability = loader.load_programmers_from_graph()
        for p in programmers_for_availability:
            availability = loader.calculate_availability(p.id)
            loader.update_graph_with_availability(p.id, availability)

    # Optionally: assign programmers to projects
    if not args.assignments:
        # Only run auto-assignment if not in offline/YAML mode
        programmers = loader.load_programmers_from_graph()
        projects = loader.load_projects(args.projects)
        new_assignments = loader.assign_programmers(projects, programmers)
        loader.save_to_neo4j(new_assignments)
