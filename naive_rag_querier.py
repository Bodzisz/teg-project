#!/usr/bin/env python3
"""
Naive RAG Querier
================

Handles querying of the ChromaDB for the Naive RAG system.
Designed to be used by the Streamlit UI.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NaiveRAGQuerier:
    """Handles querying for Naive RAG."""

    def __init__(self, persist_dir: str = "./chroma_naive_rag_cv"):
        """
        Initialize the Naive RAG Querier.
        
        Args:
            persist_dir: Directory where ChromaDB is stored.
        """
        self.persist_dir = Path(persist_dir)
        
        if not self.persist_dir.exists():
            logger.warning(f"Vector store directory not found at {persist_dir}. Please run loader first.")
            self.vectorstore = None
            self.retriever = None
            self.rag_chain = None
            return

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        self._setup_chain()
        logger.info("✓ Naive RAG Querier initialized")

    def _setup_chain(self):
        """Setup the RAG chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an helpful assistant helping with CV and RFP analysis. Use the provided context to answer questions accurately.

IMPORTANT INSTRUCTIONS:
- Base your answers ONLY on the information provided in the context
- If you cannot determine something from the context, say so clearly
- Be specific about names, skills, and details when they appear in the context
- Context contains both CVs and RFPs.

Context:
{context}"""),
            ("human", "{question}")
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query and return results in a format suitable for the UI.
        """
        start_time = time.time()
        
        if self.rag_chain is None:
            return {
                "answer": "Error: Naive RAG system not initialized. Check if database exists.",
                "context_info": [],
                "success": False
            }

        try:
            # Get docs for context info
            relevant_docs = self.retriever.invoke(question)
            
            # Get answer
            answer = self.rag_chain.invoke(question)
            
            execution_time = time.time() - start_time
            
            context_info = []
            for i, doc in enumerate(relevant_docs):
                context_info.append({
                    "chunk_index": i,
                    "source_file": doc.metadata.get("source_file", "unknown"),
                    "document_type": doc.metadata.get("document_type", "unknown"),
                    "person_name": doc.metadata.get("person_name"),
                    "rfp_name": doc.metadata.get("rfp_name"),
                    "content_preview": doc.page_content[:200] + "..."
                })
                
            return {
                "answer": answer,
                "source_type": "naive_rag",
                "execution_time": execution_time,
                "context_info": context_info,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "context_info": [],
                "success": False
            }

def main():
    """Test the querier."""
    import argparse
    parser = argparse.ArgumentParser(description="Query Naive RAG")
    parser.add_argument("query", help="Question to ask")
    args = parser.parse_args()
    
    querier = NaiveRAGQuerier()
    result = querier.query(args.query)
    
    print("\nAnswer:")
    print("-" * 40)
    print(result["answer"])
    print("-" * 40)
    print(f"Time: {result.get('execution_time', 0):.2f}s")
    print(f"Sources: {len(result.get('context_info', []))}")

if __name__ == "__main__":
    main()
