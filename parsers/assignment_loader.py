import argparse
import json
import logging
import random
import tomllib
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
    """Load configuration from TOML file for Assignment Loader."""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            logger.error("Failed to load config file: %s", e)
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
        """Load and validate projects from JSON file."""
        try:
            with open(project_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            projects = [ProjectData(**item) for item in data]
            logger.info(f"✅ Loaded {len(projects)} projects from {project_file}")
            return projects
        except Exception as e:
            logger.error("Failed to load projects: %s", e)
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
            logger.error("Failed to load programmers: %s", e)
            raise

    def load_programmers_from_graph(self) -> List[ProgrammerData]:
        """Fetch programmers and their skills from Neo4j."""
        query = """
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[hs:HAS_SKILL]->(s:Skill)
        RETURN p.name AS name, p.email AS email, p.location AS location,
               collect({name: s.name, proficiency: hs.proficiency}) AS skills
        """
        results = self.graph.query(query)
        programmers = []
        for result in results:
            # Skip programmers with missing name
            if not result["name"]:
                logger.warning(f"Skipping programmer with missing name")
                continue

            # Filter out null skills
            skills = [
                Skill(name=skill["name"], proficiency=skill["proficiency"] or "Intermediate")
                for skill in result["skills"]
                if skill["name"] is not None
            ]

            programmer = ProgrammerData(
                id=result["name"],
                name=result["name"],
                email=result["email"],
                location=result["location"],
                skills=skills
            )
            programmers.append(programmer)
        logger.info(f"✅ Loaded {len(programmers)} programmers from graph")
        return programmers

    def load_projects_from_graph(self) -> List[ProjectData]:
        """Load projects from Neo4j graph."""
        from datetime import datetime

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
            # Skip projects with missing essential data
            if not result["id"] or not result["name"] or not result["client"] or not result["start_date"]:
                logger.warning(f"Skipping project with missing essential data: id={result['id']}, name={result['name']}, client={result['client']}")
                continue

            # Parse date strings to date objects
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

            # Filter out null requirements
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

    def assign_programmers(self, projects: List[ProjectData], programmers: List[ProgrammerData]) -> List[dict]:
        """Assign programmers to projects based on config rules."""
        probability = self.config["assignment"]["assignment_probability"]
        min_days = self.config["assignment"]["assignment_end_days_before_min"]
        max_days = self.config["assignment"]["assignment_end_days_before_max"]

        assignments = []

        for project in projects:
            if random.random() > probability:
                logger.info(f"Skipping assignment for project {project.id}")
                continue

            required_skills = [req.skill_name for req in project.requirements]
            eligible = [
                p for p in programmers
                if any(skill.name in required_skills for skill in p.skills)
            ]

            if not eligible:
                logger.warning(f"No eligible programmers for project {project.id}")
                continue

            assigned = random.sample(eligible, min(len(eligible), project.team_size))

            # Calculate assignment end date
            if project.end_date:
                end_date_obj = datetime.fromisoformat(str(project.end_date))
            else:
                end_date_obj = datetime.fromisoformat(str(project.start_date)) + timedelta(days=project.estimated_duration_months * 30)

            end_days = random.randint(min_days, max_days)
            assignment_end_date = (end_date_obj - timedelta(days=end_days)).strftime("%Y-%m-%d")

            for prog in assigned:
                assignments.append({
                    "project_id": project.name,
                    "programmer_id": prog.id,
                    "end_date": assignment_end_date
                })

        logger.info(f"✅ Generated {len(assignments)} assignments.")
        return assignments

    
    def save_to_neo4j(self, assignments: List[dict]):
        """
        Save assignments to Neo4j as relationships between Project and Person.
        Eliminates Cartesian Product warning and ensures performance with indexes.
        """
        try:
            # Ensure indexes exist for fast lookup
            index_queries = [
                "CREATE INDEX project_name IF NOT EXISTS FOR (proj:Project) ON (proj.name)",
                "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)"
            ]
            for query in index_queries:
                try:
                    self.graph.query(query)
                except Exception as e:
                    logger.debug(f"Index creation skipped or already exists: {e}")

            # Insert assignments
            for assignment in assignments:
                query = """
                MATCH (proj:Project {name: $project_id})
                MATCH (p:Person {name: $programmer_id})
                MERGE (p)-[:ASSIGNED_TO {end_date: $end_date}]->(proj)
                """
                self.graph.query(query, {
                    "project_id": assignment["project_id"],
                    "programmer_id": assignment["programmer_id"],
                    "end_date": assignment["end_date"]
                })

            logger.info(f"✅ Saved {len(assignments)} assignments to Neo4j.")
        except Exception as e:
            logger.error(f"❌ Failed to save assignments to Neo4j: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign programmers to projects and save to Neo4j.")
    parser.add_argument("--config", type=str, default="utils/config.toml", help="Path to config file.")
    parser.add_argument("--projects", type=str, default="data/projects/projects.json", help="Path to projects JSON.")
    parser.add_argument("--programmers", type=str, help="Optional JSON file with programmers (offline mode).")
    args = parser.parse_args()

    loader = AssignmentLoader(config_path=args.config)
    projects = loader.load_projects(args.projects)

    if args.programmers:
        programmers = loader.load_programmers_from_json(args.programmers)
        programmers = [p.dict() for p in programmers]
    else:
        programmers = loader.load_programmers_from_graph()

    assignments = loader.assign_programmers(projects, programmers)
    loader.save_to_neo4j(assignments)
