import argparse
import logging
import asyncio
from pathlib import Path
from parsers.rfp_parser import RFPParser, ConfigLoader
from parsers.assignment_loader import AssignmentLoader
import sys
sys.path.append('.')
from cv_knowledge_graph_builder import DataKnowledgeGraphBuilder

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GraphPipeline")


class GraphPipeline:
    """Comprehensive pipeline: CVs -> RFP (loads data into Neo4j)."""
    def __init__(self, config_path: str, clear_graph: bool = False):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        self.cv_builder = DataKnowledgeGraphBuilder(config_path, clear_graph=clear_graph)
        self.rfp_parser = RFPParser(model_name=self.config.get("llm", {}).get("model", "gpt-4o-mini"))
        # Initialize AssignmentLoader for use by UI-triggered assignment flows
        try:
            self.assignment_loader = AssignmentLoader(config_path=config_path)
        except Exception:
            self.assignment_loader = None
        logger.info("✅ Comprehensive GraphPipeline initialized (CVs & RFPs only).")

    async def run(self, process_cvs: bool = True, parse_rfp: bool = True):
        """Runs the pipeline: optionally process CVs and parse RFPs (load into Neo4j)."""
        if process_cvs:
            logger.info("🔍 Stage 1: Processing CVs to knowledge graph...")
            try:
                await self.cv_builder.process_all_cvs()
                self.cv_builder.create_indexes()
                self.cv_builder.validate_graph()
                # Ensure default availability for all Person nodes after ingest
                try:
                    res = self.cv_builder.graph.query(
                        """
                        MATCH (p:Person)
                        WHERE p.availability IS NULL OR p.availability <= 0
                        SET p.availability = 100
                        RETURN count(p) as updated
                        """
                    )
                    updated = res[0]["updated"] if res else 0
                    logger.info(f"✅ Set default availability=100 for {updated} Person nodes")
                except Exception as e:
                    logger.warning("⚠ Could not set default availability after ingest: %s", e)
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

        logger.info("🎯 Pipeline completed (CVs & RFPs loaded to graph).")


if __name__ == "__main__":
    import asyncio
    parser = argparse.ArgumentParser(description="Run the comprehensive graph building pipeline.")
    parser.add_argument("--config", type=str, default="utils/config.toml", help="Path to configuration file.")
    parser.add_argument("--skip-cvs", action="store_true", help="Skip CV processing stage.")
    parser.add_argument("--skip-rfp", action="store_true", help="Skip RFP parsing stage.")
    parser.add_argument("--clear-graph", action="store_true", help="Clear the entire graph before processing (WARNING: deletes all data).")
    args = parser.parse_args()

    async def main():
        pipeline = GraphPipeline(config_path=args.config, clear_graph=args.clear_graph)
        await pipeline.run(
            process_cvs=not args.skip_cvs,
            parse_rfp=not args.skip_rfp,
        )

    asyncio.run(main())
