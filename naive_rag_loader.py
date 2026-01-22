#!/usr/bin/env python3
"""
Naive RAG Loader
===============

Handles loading of documents (CVs and RFPs) into ChromaDB for the Naive RAG system.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NaiveRAGLoader:
    """Handles loading and processing of documents for Naive RAG."""

    def __init__(self, persist_dir: str = "./chroma_naive_rag_cv"):
        """
        Initialize the Naive RAG Loader.
        
        Args:
            persist_dir: Directory where ChromaDB will store its data.
        """
        self.persist_dir = Path(persist_dir)
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def _determine_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Determine metadata based on file path."""
        metadata = {
            "source_file": file_path.name,
        }
        
        path_str = str(file_path)
        
        # Simple heuristic based on directory structure
        if "programmers" in path_str:
            metadata["document_type"] = "cv"
            metadata["person_name"] = file_path.stem
        elif "RFP" in path_str:
            metadata["document_type"] = "rfp"
            metadata["rfp_name"] = file_path.stem
        else:
            metadata["document_type"] = "unknown"
            
        return metadata

    def load_from_directory(self, data_dir: str = "./data", reset_db: bool = False) -> None:
        """
        Load all PDF files from the data directory recursively.
        
        Args:
            data_dir: Root directory to search for PDFs.
            reset_db: If True, deletes existing database before loading.
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Find all PDFs recursively
        pdf_files = list(data_path.rglob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {data_dir}")
            return

        logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")
        
        # Initial load logic can reuse add_pdfs provided we handle the DB reset first
        if reset_db and self.persist_dir.exists():
            logger.info(f"Removing existing database at {self.persist_dir}")
            shutil.rmtree(self.persist_dir)
            
        self.add_pdfs([str(p) for p in pdf_files])

    def add_pdfs(self, pdf_paths: List[str]) -> None:
        """
        Process and add a specific list of PDF files to the database.
        
        Args:
            pdf_paths: List of file paths to PDF documents.
        """
        documents = []
        
        for path_str in pdf_paths:
            file_path = Path(path_str)
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                
                # Add metadata
                file_metadata = self._determine_metadata(file_path)
                
                for doc in docs:
                    doc.metadata.update(file_metadata)
                    
                documents.extend(docs)
                logger.info(f"Loaded {file_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        if not documents:
            logger.warning("No documents successfully loaded.")
            return

        # Split documents
        logger.info(f"Splitting {len(documents)} document pages...")
        texts = self.text_splitter.split_documents(documents)
        
        # Filter complex metadata to ensure compatibility with ChromaDB
        # This removes any metadata values that are not str, int, float, or bool
        texts = filter_complex_metadata(texts)
        
        logger.info(f"Created {len(texts)} chunks")

        # Create/Update vector store
        logger.info(f"Persisting to {self.persist_dir}...")
        
        # We use from_documents which automatically persists if persist_directory is set
        Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        
        logger.info("✓ Documents added to ChromaDB")

def main():
    """Main function to run the loader on default data directory."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load documents into Naive RAG ChromaDB")
    parser.add_argument("--data-dir", default="./data", help="Directory containing documents")
    parser.add_argument("--reset", action="store_true", help="Reset database before loading")
    
    args = parser.parse_args()
    
    loader = NaiveRAGLoader()
    loader.load_from_directory(data_dir=args.data_dir, reset_db=args.reset)

if __name__ == "__main__":
    main()
