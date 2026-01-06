import argparse
import json
import logging
import random
import tomllib
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import List
from models.project_models import ProjectData
from models.programmer_models import ProgrammerData
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

    def load_programmers_from_graph(self):
        """Fetch programmers and their skills from Neo4j."""
        query = """
        MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
        RETURN p.id AS id, p.name AS name, collect({skill_name: s.id}) AS skills
        """
        return self.graph.query(query)

    def assign_programmers(self, projects: List[ProjectData], programmers: List[dict]) -> List[dict]:
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
                if any(skill.get("skill_name") in required_skills for skill in p["skills"])
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
                    "project_id": project.id,
                    "programmer_id": prog["id"],
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
                "CREATE INDEX project_id IF NOT EXISTS FOR (proj:Project) ON (proj.id)",
                "CREATE INDEX person_id IF NOT EXISTS FOR (p:Person) ON (p.id)"
            ]
            for query in index_queries:
                try:
                    self.graph.query(query)
                except Exception as e:
                    logger.debug(f"Index creation skipped or already exists: {e}")

            # Insert assignments
            for assignment in assignments:
                query = """
                MATCH (proj:Project {id: $project_id})
                MATCH (p:Person {id: $programmer_id})
                MERGE (proj)-[:ASSIGNED_TO {end_date: $end_date}]->(p)
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
