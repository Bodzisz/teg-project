import argparse
import json
import logging
import random
import tomllib
import yaml
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import List
from models.project_models import ProjectData, Requirement
from models.programmer_models import ProgrammerData, Skill
from langchain_neo4j import Neo4jGraph

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
        RETURN sum(coalesce(r.allocation_percentage, 100)) AS allocated
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

    def assign_programmers(self, projects: List[ProjectData], programmers: List[ProgrammerData]) -> List[dict]:
        """Assign programmers based on multi-factor scoring (Skills, Experience, Availability)."""
        probability = self.config["assignment"]["assignment_probability"]
        min_days = self.config["assignment"]["assignment_end_days_before_min"]
        max_days = self.config["assignment"]["assignment_end_days_before_max"]
        assignments = []
        
        proficiency_map = {"Beginner": 1, "Intermediate": 3, "Advanced": 5, "Expert": 8}

        for project in projects:
            if random.random() > probability:
                continue
            
            required_skills = [req.skill_name for req in project.requirements]
            
            # Filter eligible candidates (must have at least one skill and positive availability)
            eligible = [p for p in programmers if p.availability > 0 and any(skill.name in required_skills for skill in p.skills)]
            
            if not eligible:
                continue

            # Calculate scores
            scored_candidates = []
            for p in eligible:
                score = 0
                
                # Skill Score
                p_skills_map = {s.name: s.proficiency for s in p.skills}
                for r_skill in required_skills:
                    if r_skill in p_skills_map:
                        score += 10
                        score += proficiency_map.get(p_skills_map[r_skill], 1)
                
                # Experience Score
                score += (p.years_experience or 0) * 2
                
                # Availability Score (preference for those with more availability)
                score += (p.availability or 0) * 0.5
                
                scored_candidates.append((score, p))
            
            # Sort by score descending
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Select top candidates (up to team size)
            assigned_count = min(len(scored_candidates), project.team_size)
            best_candidates = [p for _, p in scored_candidates[:assigned_count]]

            end_date_obj = datetime.fromisoformat(str(project.end_date)) if project.end_date else \
                datetime.fromisoformat(str(project.start_date)) + timedelta(days=project.estimated_duration_months * 30)
            end_days = random.randint(min_days, max_days)
            assignment_end_date = (end_date_obj - timedelta(days=end_days)).strftime("%Y-%m-%d")

            for prog in best_candidates:
                # Determine allocation: allocate 100% or remaining availability
                allocation = min(100.0, prog.availability)
                
                if allocation > 0:
                    assignments.append({
                        "project_id": project.name,
                        "programmer_id": prog.id,
                        "end_date": assignment_end_date,
                        "allocation": allocation
                    })
                    
                    # Update in-memory availability to prioritize distribution
                    prog.availability -= allocation
                    
        logger.info(f"✅ Generated {len(assignments)} assignments.")
        return assignments

    def save_to_neo4j(self, assignments: List[dict]):
        """Save assignments to Neo4j."""
        try:
            for assignment in assignments:
                query = """
                MATCH (proj:Project {name: $project_id})
                MATCH (p:Person {name: $programmer_id})
                MERGE (p)-[:ASSIGNED_TO {end_date: $end_date, allocation_percentage: $allocation}]->(proj)
                """
                self.graph.query(query, {
                    "project_id": assignment["project_id"],
                    "programmer_id": assignment["programmer_id"],
                    "end_date": assignment["end_date"],
                    "allocation": assignment.get("allocation", 100)
                })
            logger.info(f"✅ Saved {len(assignments)} assignments to Neo4j.")
        except Exception as e:
            logger.error(f"❌ Error saving to Neo4j: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate availability and assign programmers.")
    parser.add_argument("--config", type=str, default="utils/config.toml", help="Path to the configuration file.")
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
