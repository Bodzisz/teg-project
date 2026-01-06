import argparse
import logging
from pathlib import Path
from parsers.rfp_parser import RFPParser, ConfigLoader
from parsers.assignment_loader import AssignmentLoader

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GraphPipeline")


class GraphPipeline:
    """Pipeline do budowania grafu: RFP -> Projekty -> Przypisania programistów."""
    def __init__(self, config_path: str):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        self.rfp_parser = RFPParser(model_name=self.config.get("llm", {}).get("model", "gpt-4o-mini"))
        self.assignment_loader = AssignmentLoader(config_path=config_path)
        logger.info("✅ GraphPipeline initialized.")

    def run(self, parse_rfp: bool = True, assign_programmers: bool = True):
        """Uruchamia pipeline w trybie kaskadowym."""
        if parse_rfp:
            logger.info("🔍 Etap 1: Parsowanie RFP...")
            rfp_dir = self.config.get("parser", {}).get("rfp_dir", "data/RFP")
            pdf_files = list(Path(rfp_dir).glob("*.pdf"))
            if not pdf_files:
                logger.warning("⚠ Brak plików PDF w katalogu: %s", rfp_dir)
            else:
                for pdf_path in pdf_files:
                    logger.info("📄 Przetwarzanie pliku: %s", pdf_path)
                    text = self.rfp_parser.extract_text_from_pdf(str(pdf_path))
                    if not text:
                        logger.warning("⚠ Pominięto plik %s (brak tekstu)", pdf_path)
                        continue
                    try:
                        rfp_data = self.rfp_parser.parse_rfp(text)
                        self.rfp_parser.save_to_neo4j(rfp_data)
                        logger.info("✅ RFP zapisane w grafie: %s", rfp_data.title)
                    except Exception as e:
                        logger.error("❌ Błąd podczas przetwarzania RFP: %s", e)

        if assign_programmers:
            logger.info("🔍 Etap 2: Przypisywanie programistów...")
            projects_file = self.config.get("output", {}).get("projects_file", "data/projects/projects.json")
            try:
                projects = self.assignment_loader.load_projects(projects_file)
                programmers = self.assignment_loader.load_programmers_from_graph()
                assignments = self.assignment_loader.assign_programmers(projects, programmers)
                self.assignment_loader.save_to_neo4j(assignments)
                logger.info("✅ Przypisania zapisane w grafie.")
            except Exception as e:
                logger.error("❌ Błąd podczas przypisywania programistów: %s", e)

        logger.info("🎯 Pipeline zakończony.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uruchom pipeline budowania grafu.")
    parser.add_argument("--config", type=str, default="utils/config.toml", help="Ścieżka do pliku konfiguracyjnego.")
    parser.add_argument("--skip-rfp", action="store_true", help="Pomiń etap parsowania RFP.")
    parser.add_argument("--skip-assign", action="store_true", help="Pomiń etap przypisywania programistów.")
    args = parser.parse_args()

    pipeline = GraphPipeline(config_path=args.config)
    pipeline.run(parse_rfp=not args.skip_rfp, assign_programmers=not args.skip_assign)
