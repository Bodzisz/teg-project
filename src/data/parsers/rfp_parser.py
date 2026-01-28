import argparse
from datetime import datetime
from dotenv import load_dotenv
import json
import logging
import os
from pathlib import Path
import tomllib  # For Python 3.11+, use 'toml' for older versions
import uuid

from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from unstructured.partition.pdf import partition_pdf

from src.data.models.rfp_models import RFPData
from src.core.utils.prompt_loader import load_prompt


# Load environment variables
load_dotenv(override=True)

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RFPParser")


class ConfigLoader:
    """Load configuration from TOML file for RFP Parser."""
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


class RFPParser:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables.")
        self.llm = ChatOpenAI(model=model_name, temperature=0, api_key=self.openai_api_key)
        self.graph = Neo4jGraph()
        logger.info("RFPParser initialized with model %s", model_name)

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using unstructured."""
        try:
            elements = partition_pdf(filename=file_path)
            full_text = "\n\n".join([str(el) for el in elements])
            logger.info("Extracted text from %s", file_path)
            return full_text
        except Exception as e:
            logger.error("Failed to extract text: %s", e)
            return ""

    
    def parse_rfp(self, rfp_text: str) -> RFPData:
        """
        Parse RFP text using LLM with StructuredOutputParser and validate with Pydantic.
        """

        # 1. Definition of response schema
        response_schemas = [
            ResponseSchema(name="id", description="Unique RFP identifier"),
            ResponseSchema(name="title", description="Title of the RFP"),
            ResponseSchema(name="client", description="Client name"),
            ResponseSchema(name="description", description="Description of the RFP"),
            ResponseSchema(name="project_type", description="Type of project"),
            ResponseSchema(name="duration_months", description="Estimated duration in months"),
            ResponseSchema(name="team_size", description="Expected team size"),
            ResponseSchema(name="budget_range", description="Budget range"),
            ResponseSchema(name="start_date", description="Start date in YYYY-MM-DD format"),
            ResponseSchema(name="location", description="Project location"),
            ResponseSchema(name="remote_allowed", description="Boolean indicating if remote work is allowed"),
            ResponseSchema(name="requirements", description="List of requirements with skill_name, min_proficiency, is_mandatory, preferred_certifications (always as list of strings)")
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        # 2. Prompt construction
        prompts_dir = Path(__file__).parent / "prompts"
        template_text = load_prompt(prompts_dir / "rfp_parsing.txt")
        
        prompt_template = PromptTemplate(
            input_variables=["rfp_text", "format_instructions"],
            template=template_text
        )

        prompt = prompt_template.format(rfp_text=rfp_text, format_instructions=format_instructions)
        logger.debug("Prompt sent do LLM:\n%s", prompt)

        # 3. LLM call
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw_output = response.content.strip()
            logger.debug("Raw output from LLM:\n%s", raw_output)
        except Exception as e:
            logger.error("Error during LLM call: %s", e)
            raise

        # 4. Parsing the response into a structured format
        try:
            parsed_output = output_parser.parse(raw_output)
            logger.info("Correct answer parsed by StructuredOutputParser.")
        except Exception as e:
            logger.error("Error parsing response: %s", e)
            logger.debug("Raw output:\n%s", raw_output)
            raise

        # 5. Validation with Pydantic
        try:
            rfp_data = RFPData(**parsed_output)
            logger.info("RFP data correctly validated by Pydantic.")
            return rfp_data
        except Exception as e:
            logger.error("Pydantic validation error: %s", e)
            logger.debug("Parsed output:\n%s", parsed_output)
            raise

    
    def save_to_neo4j(self, rfp_data: RFPData):
        """Save RFP data to Neo4j. Project creation will be handled separately after approval."""
        try:
            query = """
            CREATE (r:RFP {
                id: $rfp_id,
                entity_id: $rfp_id,
                title: $title,
                client: $client,
                description: $description,
                project_type: $project_type,
                duration_months: $duration_months,
                team_size: $team_size,
                budget_range: $budget_range,
                start_date: $start_date,
                location: $location,
                remote_allowed: $remote_allowed
            })
            WITH r
            UNWIND $requirements AS req
            MERGE (s:Skill {entity_id: req.skill_name, name: req.skill_name})
            MERGE (r)-[:NEEDS {
                min_proficiency: req.min_proficiency,
                is_mandatory: req.is_mandatory,
                preferred_certifications: req.preferred_certifications
            }]->(s)
            """
            self.graph.query(query, {
                "rfp_id": uuid.uuid4().hex,
                "title": rfp_data.title,
                "client": rfp_data.client,
                "description": rfp_data.description,
                "project_type": rfp_data.project_type,
                "duration_months": rfp_data.duration_months,
                "team_size": rfp_data.team_size,
                "budget_range": rfp_data.budget_range,
                "start_date": str(rfp_data.start_date),
                "location": rfp_data.location,
                "remote_allowed": rfp_data.remote_allowed,
                "requirements": [req.model_dump() for req in rfp_data.requirements]
            })
            logger.info(f"✅ RFP {rfp_data.id} saved to Neo4j.")
        except Exception as e:
            logger.error("❌ Failed to save RFP to Neo4j: %s", e)

    def create_project_from_rfp(self, rfp_id: str):
        """Convert an approved RFP into a Project node."""
        from datetime import timedelta
        try:
            # Check if Project already exists
            check_query = "MATCH (p:Project {id: $project_id}) RETURN p"
            existing = self.graph.query(check_query, {"project_id": f"PRJ-{rfp_id}"})
            if existing:
                logger.info(f"Project PRJ-{rfp_id} already exists.")
                # return both id and name for downstream callers
                p = existing[0].get("p") or {}
                return {"id": f"PRJ-{rfp_id}", "name": p.get("name")}

            # Get RFP data
            rfp_query = """
            MATCH (r:RFP {id: $rfp_id})
            OPTIONAL MATCH (r)-[needs:NEEDS]->(s:Skill)
            RETURN r, collect({skill: s.name, min_proficiency: needs.min_proficiency, is_mandatory: needs.is_mandatory, preferred_certifications: needs.preferred_certifications}) as requirements
            """
            result = self.graph.query(rfp_query, {"rfp_id": rfp_id})
            if not result:
                logger.warning(f"RFP {rfp_id} not found.")
                return
            rfp = result[0]["r"]
            requirements = result[0]["requirements"]

            # Calculate end_date
            start_date = datetime.fromisoformat(rfp["start_date"])
            end_date = start_date + timedelta(days=rfp["duration_months"] * 30)

            # Create Project
            project_query = """
            CREATE (p:Project {
                id: $project_id,
                entity_id: $project_id,
                name: $name,
                client: $client,
                description: $description,
                start_date: $start_date,
                end_date: $end_date,
                estimated_duration_months: $duration_months,
                budget: $budget,
                status: "planned",
                team_size: $team_size
            })
            WITH p
            MATCH (r:RFP {id: $rfp_id})
            CREATE (r)-[:GENERATES]->(p)
            WITH p
            UNWIND $requirements AS req
            MERGE (s:Skill {name: req.skill})
            MERGE (p)-[:REQUIRES {
                min_proficiency: req.min_proficiency,
                is_mandatory: req.is_mandatory,
                preferred_certifications: req.preferred_certifications
            }]->(s)
            """
            self.graph.query(project_query, {
                "project_id": f"PRJ-{rfp_id}",
                "name": rfp["title"],
                "client": rfp["client"],
                "description": rfp["description"],
                "start_date": rfp["start_date"],
                "end_date": str(end_date.date()),
                "duration_months": rfp["duration_months"],
                "budget": rfp["budget_range"],
                "team_size": rfp["team_size"],
                "rfp_id": rfp_id,
                "requirements": requirements
            })
            logger.info(f"✅ Project PRJ-{rfp_id} created from RFP {rfp_id}.")
            return {"id": f"PRJ-{rfp_id}", "name": rfp.get("title")}
        except Exception as e:
            logger.error("❌ Failed to create Project from RFP: %s", e)
            return None



if __name__ == "__main__":
    parser_cli = argparse.ArgumentParser(description="Parse RFP PDFs and save to Neo4j.")
    parser_cli.add_argument("--config", type=str, default="config/config.toml", help="Path to config file.")
    parser_cli.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model for parsing.")
    parser_cli.add_argument("pdf_files", nargs="*", help="Paths to RFP PDF files. If empty, will use config directory.")
    args = parser_cli.parse_args()

    # Load config
    config_loader = ConfigLoader(args.config)
    
    # Determine files to process based on configuration
    rfp_dir = config_loader.get("parser", {}).get("rfp_dir", "data/RFP")
    pdf_files = args.pdf_files if args.pdf_files else list(Path(rfp_dir).glob("*.pdf"))

    if not pdf_files:
        logger.error("No PDF files found in directory: %s", rfp_dir)
        exit(1)

    logger.info("Found %d PDF files to process in %s", len(pdf_files), rfp_dir)
    # Initialize parser
    rfp_parser = RFPParser(model_name=args.model)

    for pdf_path in pdf_files:
        logger.info("Processing file: %s", pdf_path)
        text = rfp_parser.extract_text_from_pdf(str(pdf_path))
        if not text:
            logger.warning("Skipping file %s (no text extracted)", pdf_path)
            continue
        try:
            rfp_data = rfp_parser.parse_rfp(text)
            rfp_parser.save_to_neo4j(rfp_data)
            print(f"\nJSON for {pdf_path}:\n{json.dumps(rfp_data.model_dump(), indent=2)}\n")
        except Exception as e:
            logger.error("Error processing file %s: %s", pdf_path, e)
