import argparse
from dotenv import load_dotenv
import json
import logging
import os
from pathlib import Path
import tomllib  # For Python 3.11+, use 'toml' for older versions

from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from unstructured.partition.pdf import partition_pdf

from models.rfp_models import RFPData


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
            ResponseSchema(name="title", description="Title of the RFP"),
            ResponseSchema(name="description", description="Short description of the RFP"),
            ResponseSchema(name="skills", description="List of skills with name and experience_level"),
            ResponseSchema(name="budget", description="Budget range or value"),
            ResponseSchema(name="deadline", description="Deadline or project completion date"),
            ResponseSchema(name="team_size", description="Expected team size as integer")
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        # 2. Prompt construction
        prompt_template = PromptTemplate(
            input_variables=["rfp_text", "format_instructions"],
            template="""
    Analyze the following RFP text and return ONLY valid JSON according to the format instructions below.

    {format_instructions}

    RFP Text:
    {rfp_text}
    """
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
        """Save RFP data to Neo4j as node and relationships."""
        try:
            query = """
            CREATE (r:RFP {title: $title, description: $description, budget: $budget, deadline: $deadline, team_size: $team_size})
            WITH r
            UNWIND $skills AS skill
            MERGE (s:Skill {name: skill.name})
            MERGE (r)-[:NEEDS {experience_level: skill.experience_level}]->(s)
            """
            self.graph.query(query, {
                "title": rfp_data.title,
                "description": rfp_data.description,
                "budget": rfp_data.budget,
                "deadline": rfp_data.deadline,
                "team_size": rfp_data.team_size,
                "skills": [skill.model_dump() for skill in rfp_data.skills]
            })
            logger.info("RFP saved to Neo4j.")
        except Exception as e:
            logger.error("Failed to save RFP to Neo4j: %s", e)


if __name__ == "__main__":
    parser_cli = argparse.ArgumentParser(description="Parse RFP PDFs and save to Neo4j.")
    parser_cli.add_argument("--config", type=str, default="utils/config.toml", help="Path to config file.")
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

