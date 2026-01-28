"""
GraphRAG Query System for CV Knowledge Graph
============================================

Demonstrates GraphRAG capabilities by querying the knowledge graph
built from PDF CVs using natural language queries.

Shows advantages of structured graph queries over traditional RAG.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from src.core.utils.prompt_loader import load_prompt

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CVGraphRAGSystem:
    """GraphRAG system for querying CV-only knowledge graph.

    Enables natural language queries on knowledge graphs built from CV data,
    including Person nodes with skills, education, work experience, and certifications.
    """

    def __init__(self):
        """Initialize the GraphRAG system."""
        self.setup_neo4j()
        self.setup_qa_chain()
        self.load_example_queries()
        self.chat_histories = {}  # Store conversation histories in memory

    def _clean_cypher_query(self, query: str) -> str:
        """Clean up Cypher query by removing markdown formatting."""
        if not query:
            return ""
            
        # Remove markdown code blocks
        clean_query = query.replace("```cypher", "").replace("```", "").strip()
        
        # Handle case where language identifier might have a space or different casing
        if clean_query.lower().startswith("cypher"):
             clean_query = clean_query[6:].strip()
             
        return clean_query

    def setup_neo4j(self):
        """Setup Neo4j connection."""
        try:
            self.graph = Neo4jGraph(
                url="bolt://localhost:7687",
                username=os.getenv("NEO4J_USERNAME"),
                password=os.getenv("NEO4J_PASSWORD")
            )
            logger.info("✓ Connected to Neo4j successfully")

            # Refresh schema for accurate query generation
            self.graph.refresh_schema()
            logger.info("✓ Graph schema refreshed")

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            print("❌ Could not connect to Neo4j. Make sure it's running:")
            print("   ./start_session.sh")
            raise

    def setup_qa_chain(self):
        """Setup the GraphCypherQA chain."""
        # Initialize LLM for query generation
        self.llm = ChatOpenAI(
            model="gpt-4o",  # Use more powerful model for query generation
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Custom Cypher generation prompt with case-insensitive matching
        prompts_dir = Path(__file__).parent / "prompts"
        
        CYPHER_GENERATION_TEMPLATE = load_prompt(prompts_dir / "cypher_generation.txt")

        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["schema", "question", "chat_history"],
            partial_variables={"today": datetime.now().strftime("%Y-%m-%d")},
            template=CYPHER_GENERATION_TEMPLATE
        )

        # Custom QA prompt for better handling of numeric results
        CYPHER_QA_TEMPLATE = load_prompt(prompts_dir / "cypher_qa.txt")

        CYPHER_QA_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            partial_variables={"today": datetime.now().strftime("%Y-%m-%d")},
            template=CYPHER_QA_TEMPLATE
        )

        # Create the GraphCypher QA chain with custom prompts
        self.qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,  # Show generated Cypher queries
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            qa_prompt=CYPHER_QA_PROMPT,
            return_intermediate_steps=True,
            return_direct=False, # We want the QA chain to interpret results
            allow_dangerous_requests=True  # Allow DELETE operations for demo
        )

        logger.info("✓ GraphCypher QA chain initialized with custom prompts")

    def query_graph(self, question: str, conversation_id: str = None) -> Dict[str, Any]:
        """Execute a natural language query against the graph using simple detailed Chain.
        
        This bypasses the Coordinator/Planner and runs a direct Cypher QA Chain.
        """
        if not conversation_id:
            chat_history_str = "No history."
        else:
            if conversation_id not in self.chat_histories:
                self.chat_histories[conversation_id] = []
            history_list = self.chat_histories[conversation_id][-5:]
            chat_history_str = "\n".join([f"Human: {q}\nAI: {a}" for q, a in history_list]) if history_list else "No history."

        try:
            logger.info(f"Executing query via Simple Graph Chain: {question}")
            
            result = self.qa_chain.invoke({
                "query": question,
                "chat_history": chat_history_str
            })
            
            answer = result.get("result", "No answer")
            
            # Extract Cypher from intermediate steps if available
            cypher_query = ""
            context = []
            if "intermediate_steps" in result:
                steps = result["intermediate_steps"]
                # Usually steps[0] is query generation, steps[1] is context retrieval
                if len(steps) >= 1:
                    cypher_query = steps[0].get("query", "")
                if len(steps) >= 2:
                    context = steps[1].get("context", [])

            response = {
                "question": question,
                "answer": answer,
                "cypher_query": cypher_query,
                "context": context,
                "conversation_id": conversation_id,
                "success": True
            }
            
            if conversation_id:
                self.chat_histories[conversation_id].append((question, answer))
                
            logger.info(f"✓ Query executed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Simple query failed: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "cypher_query": "",
                "conversation_id": conversation_id,
                "success": False
            }
            
    def load_example_queries(self):
        """Load example queries that demonstrate GraphRAG capabilities for CV data."""
        self.example_queries = {
            "Basic Information": [
                "How many people are in the knowledge graph?",
                "What companies appear in the CVs?",
                "List all the skills mentioned in the CVs.",
                "What certifications do people have?",
                "Which universities appear in the CVs?",
                "What job titles are mentioned?",
                "Show me all the locations where people are based."
            ],

            "Skill-based Queries": [
                "Who has Python programming skills?",
                "Find all people with React experience.",
                "Who has both Docker and Kubernetes skills?",
                "List people with JavaScript skills.",
                "Find people who know both Python and Django.",
                "Who has cloud computing skills like AWS?",
                "What programming languages are most common?",
                "Find people with machine learning expertise."
            ],

            "Company Experience": [
                "Who worked at Google?",
                "Find people who worked at Microsoft.",
                "What companies have the most former employees in our database?",
                "Who worked at technology companies?",
                "Find people with startup experience.",
                "List all companies mentioned in the CVs.",
                "Who has experience at Fortune 500 companies?"
            ],

            "Education Background": [
                "Who studied at Stanford University?",
                "Find people with computer science education.",
                "What universities are most common in our database?",
                "Who has a Master's degree?",
                "Find people who studied at Ivy League schools.",
                "What are the most common degree types?",
                "Who has a PhD?"
            ],

            "Location and Geography": [
                "Who is located in San Francisco?",
                "Find people in California.",
                "What cities have the most people?",
                "Who is located in New York?",
                "Find people in the United States.",
                "Show all locations in our database.",
                "Find people willing to relocate."
            ],

            "Professional Experience": [
                "Who has the most years of experience?",
                "Find senior-level professionals.",
                "Who worked in software development roles?",
                "Find people with leadership experience.",
                "Who has experience in data science?",
                "List all job titles mentioned.",
                "Find people with consulting experience."
            ],

            "Multi-hop Reasoning": [
                "Find people who worked at the same companies.",
                "Who went to the same university and has similar skills?",
                "Find people who have complementary skills for a team.",
                "What skills are commonly paired together?",
                "Find potential colleagues based on shared experience.",
                "Who studied at top universities and has industry experience?",
                "Find people with both technical and business skills."
            ],

            "Certification Analysis": [
                "Who has AWS certifications?",
                "Find all Google Cloud certified people.",
                "What are the most common certifications?",
                "Who has multiple certifications?",
                "Find people with security certifications.",
                "List all certification providers.",
                "Who has recent certifications?"
            ]
        }

    def run_example_queries(self, category: str = None) -> List[Dict[str, Any]]:
        """Run example queries to demonstrate GraphRAG capabilities.

        Args:
            category: Optional category to filter queries

        Returns:
            List of query results
        """
        results = []

        categories_to_run = [category] if category else self.example_queries.keys()

        for cat in categories_to_run:
            if cat not in self.example_queries:
                logger.warning(f"Category '{cat}' not found")
                continue

            print(f"\\n{'='*60}")
            print(f"Category: {cat}")
            print(f"{'='*60}")

            for question in self.example_queries[cat]:
                print(f"\\n🔍 Query: {question}")
                print("-" * 40)

                result = self.query_graph(question)
                results.append(result)

                if result["success"]:
                    print(f"📊 Generated Cypher: {result['cypher_query']}")
                    print(f"💡 Answer: {result['answer']}")
                else:
                    print(f"❌ Error: {result['answer']}")

                print()

        return results