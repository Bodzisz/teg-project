import argparse
import logging
import asyncio
from pathlib import Path
from parsers.rfp_parser import RFPParser, ConfigLoader
from parsers.assignment_loader import AssignmentLoader
from matching_engine import MatchingEngine
import sys
sys.path.append('.')
from cv_knowledge_graph_builder import DataKnowledgeGraphBuilder

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GraphPipeline")


class GraphPipeline:
    """Comprehensive pipeline: CVs -> RFP -> Projects -> Assignments -> Matching."""
    def __init__(self, config_path: str, clear_graph: bool = False):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        self.cv_builder = DataKnowledgeGraphBuilder(config_path, clear_graph=clear_graph)
        self.rfp_parser = RFPParser(model_name=self.config.get("llm", {}).get("model", "gpt-4o-mini"))
        self.assignment_loader = AssignmentLoader(config_path=config_path)
        self.matching_engine = MatchingEngine()
        logger.info("✅ Comprehensive GraphPipeline initialized.")

    async def run(self, process_cvs: bool = True, parse_rfp: bool = True, assign_programmers: bool = True, run_matching: bool = True):
        """Runs the pipeline in cascade mode."""
        if process_cvs:
            logger.info("🔍 Stage 1: Processing CVs to knowledge graph...")
            try:
                await self.cv_builder.process_all_cvs()
                self.cv_builder.create_indexes()
                self.cv_builder.validate_graph()
                logger.info("✅ CVs processed to knowledge graph.")
            except Exception as e:
                logger.error("❌ Error during CV processing: %s", e)
                raise

        if parse_rfp:
            logger.info("🔍 Stage 2: Parsing RFPs...")
            rfp_dir = self.config.get("parser", {}).get("rfp_dir", "data/RFP")
            pdf_files = list(Path(rfp_dir).glob("*.pdf"))
            if not pdf_files:
                logger.warning("⚠ No PDF files found in directory: %s", rfp_dir)
            else:
                for pdf_path in pdf_files:
                    logger.info("📄 Processing file: %s", pdf_path)
                    text = self.rfp_parser.extract_text_from_pdf(str(pdf_path))
                    if not text:
                        logger.warning("⚠ Skipping file %s (no text)", pdf_path)
                        continue
                    try:
                        rfp_data = self.rfp_parser.parse_rfp(text)
                        self.rfp_parser.save_to_neo4j(rfp_data)
                        logger.info("✅ RFP saved to graph: %s", rfp_data.title)
                    except Exception as e:
                        logger.error("❌ Error during RFP processing: %s", e)

        if run_matching:
            logger.info("🔍 Stage 3: Matching Candidates to RFPs...")
            try:
                # Find all RFPs in the graph
                query = "MATCH (r:RFP) RETURN r.id as id, r.title as title"
                rfps = self.matching_engine.graph.query(query)
                
                if not rfps:
                    logger.warning("⚠ No RFPs found in graph for matching.")
                else:
                    for rfp in rfps:
                        rfp_id = rfp.get("id") or rfp.get("title")
                        logger.info(f"  Ranking candidates for RFP: {rfp_id}")
                        self.matching_engine.rank_candidates(rfp_id)
                    logger.info("✅ Matching completed for all RFPs.")
            except Exception as e:
                logger.error("❌ Error during candidate matching: %s", e)

        if assign_programmers:
            logger.info("🔍 Stage 4: Assigning programmers to Projects...")
            try:
                # First, calculate and update availability for all programmers based on existing graph state
                programmers = self.assignment_loader.load_programmers_from_graph()
                logger.info("  Calculated availability for %d programmers...", len(programmers))
                for p in programmers:
                    availability = self.assignment_loader.calculate_availability(p.id)
                    self.assignment_loader.update_graph_with_availability(p.id, availability)
                
                # Then proceed with assignments
                # Prefer loading projects from YAML/JSON file where requirements are explicitly defined
                projects_dir = self.config.get("output", {}).get("projects_dir", "data/projects")
                projects_yaml = Path(projects_dir) / "projects.yaml"
                projects_json = Path(projects_dir) / "projects.json"
                
                if projects_yaml.exists():
                     projects_file = str(projects_yaml)
                else:
                     projects_file = str(projects_json)
                     
                logger.info(f"  Loading projects from: {projects_file}")
                
                # Perform assignment based on RFP matches and project relationships
                assignments_summary = self.assignment_loader.assign_candidates_to_projects(projects_file)
                
                # Save assignments to Neo4j
                self.assignment_loader.save_assignments_to_neo4j(assignments_summary)
                
                logger.info("✅ Assignments completed. Summary:")
                for assignment in assignments_summary:
                    logger.info(f"  - {assignment['person_name']} -> {assignment['project_title']} ({assignment['allocation_percentage']}%)")
                
                # Validate by querying Neo4j for ASSIGNED_TO relationships
                validate_query = """
                MATCH (p:Person)-[a:ASSIGNED_TO]->(pr:Project)
                RETURN p.name as person_name, pr.name as project_title, a.allocation_percentage as allocation_percentage
                ORDER BY person_name, project_title
                """
                assigned_relationships = self.assignment_loader.graph.query(validate_query)
                logger.info(f"✅ Validation: Found {len(assigned_relationships)} ASSIGNED_TO relationships in graph.")
            except Exception as e:
                logger.error("❌ Error during programmer assignment: %s", e)
                raise

        logger.info("🎯 Pipeline completed.")


if __name__ == "__main__":
    import asyncio
    parser = argparse.ArgumentParser(description="Run the comprehensive graph building pipeline.")
    parser.add_argument("--config", type=str, default="utils/config.toml", help="Path to configuration file.")
    parser.add_argument("--skip-cvs", action="store_true", help="Skip CV processing stage.")
    parser.add_argument("--skip-rfp", action="store_true", help="Skip RFP parsing stage.")
    parser.add_argument("--skip-matching", action="store_true", help="Skip RFP candidate matching stage.")
    parser.add_argument("--skip-assign", action="store_true", help="Skip programmer assignment stage.")
    parser.add_argument("--clear-graph", action="store_true", help="Clear the entire graph before processing (WARNING: deletes all data).")
    args = parser.parse_args()

    async def main():
        pipeline = GraphPipeline(config_path=args.config, clear_graph=args.clear_graph)
        await pipeline.run(
            process_cvs=not args.skip_cvs, 
            parse_rfp=not args.skip_rfp, 
            run_matching=not args.skip_matching,
            assign_programmers=not args.skip_assign
        )

    asyncio.run(main())
