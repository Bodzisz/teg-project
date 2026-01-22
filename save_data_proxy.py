import logging
import asyncio
from pathlib import Path
from typing import List, Union

# Import underlying builders/parsers
from cv_knowledge_graph_builder import DataKnowledgeGraphBuilder
from parsers.rfp_parser import RFPParser
from naive_rag_loader import NaiveRAGLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SaveDataProxy:
    """
    Proxy to orchestrate data saving to both Graph RAG (Neo4j) and Naive RAG (ChromaDB).
    """

    def __init__(self):
        """Initialize the proxy with necessary components."""
        # Initialize Graph RAG components
        # Note: 'clear_graph=False' by default to append data
        self.cv_builder = DataKnowledgeGraphBuilder(clear_graph=False)
        self.rfp_parser = RFPParser()
        
        # Initialize Naive RAG component
        self.naive_loader = NaiveRAGLoader()

    def save_rfp(self, pdf_path: str) -> bool:
        """
        Save RFP to both Graph RAG and Naive RAG.
        
        Args:
            pdf_path: Path to the RFP PDF file
            
        Returns:
            bool: True if at least one save operation succeeded
        """
        success_graph = False
        success_naive = False
        
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            logger.error(f"File not found: {pdf_path}")
            return False

        # 1. Save to Graph RAG (Neo4j)
        try:
            logger.info(f"Saving RFP to Graph RAG: {path_obj.name}")
            text = self.rfp_parser.extract_text_from_pdf(str(pdf_path))
            if text:
                rfp_data = self.rfp_parser.parse_rfp(text)
                self.rfp_parser.save_to_neo4j(rfp_data)
                logger.info("✓ Saved to Neo4j")
                success_graph = True
            else:
                logger.warning("No text extracted for Graph RAG")
        except Exception as e:
            logger.error(f"Failed to save RFP to Graph RAG: {e}")

        # 2. Save to Naive RAG (ChromaDB)
        try:
            logger.info(f"Saving RFP to Naive RAG: {path_obj.name}")
            # NaiveRAGLoader expects a list of paths
            self.naive_loader.add_pdfs([str(pdf_path)])
            logger.info("✓ Saved to ChromaDB")
            success_naive = True
        except Exception as e:
            logger.error(f"Failed to save RFP to Naive RAG: {e}")

        return success_graph or success_naive

    async def save_cv(self, pdf_path: str) -> bool:
        """
        Save CV to both Graph RAG and Naive RAG.
        Async because DataKnowledgeGraphBuilder.convert_cv_to_graph is async.
        
        Args:
            pdf_path: Path to the CV PDF file
            
        Returns:
            bool: True if at least one save operation succeeded
        """
        success_graph = False
        success_naive = False
        
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            logger.error(f"File not found: {pdf_path}")
            return False

        # 1. Save to Graph RAG (Neo4j)
        try:
            logger.info(f"Saving CV to Graph RAG: {path_obj.name}")
            graph_documents = await self.cv_builder.convert_cv_to_graph(str(pdf_path))
            if graph_documents:
                self.cv_builder.store_graph_documents(graph_documents)
                success_graph = True
            else:
                logger.warning("No graph documents extracted")
        except Exception as e:
            logger.error(f"Failed to save CV to Graph RAG: {e}")

        # 2. Save to Naive RAG (ChromaDB)
        try:
            logger.info(f"Saving CV to Naive RAG: {path_obj.name}")
            # NaiveRAGLoader._determine_metadata needs checking if it handles single file paths correctly outside expected dir structure or if we need to mock it.
            # Looking at naive_rag_loader.py, it uses path strings to guess metadata.
            self.naive_loader.add_pdfs([str(pdf_path)])
            logger.info("✓ Saved to ChromaDB")
            success_naive = True
        except Exception as e:
            logger.error(f"Failed to save CV to Naive RAG: {e}")

        return success_graph or success_naive

    async def process_all_cvs(self, directory: str) -> None:
        """
        Batch process all CVs in a directory for both systems.
        """
        # This is larger scale, mostly for initialization scripts.
        # Implemented for completeness if needed by pipeline.
        
        # Graph RAG batch
        await self.cv_builder.process_all_cvs(directory)
        
        # Naive RAG batch
        self.naive_loader.load_from_directory(data_dir=directory, reset_db=False)
