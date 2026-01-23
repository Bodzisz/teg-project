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
from typing import List, Dict, Any, Optional
import logging

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator

class AgentState(TypedDict):
    original_query: str
    chat_history: str
    
    # Coordinator -> Planner
    coordinator_feedback: str   # Why are we planning?
    
    # Planner -> Querier
    current_plan_step: str      # Specific Cypher-friendly question
    
    # Querier -> Coordinator
    latest_query_result: Dict   # Result of the last execution
    
    # Accumulator
    gathered_info: List[Dict]   # History of all findings
    
    final_answer: str
    iterations: int

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
        self.setup_workflow()
        self.load_example_queries()
        self.chat_histories = {}  # Store conversation histories in memory

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
        CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For skill matching, always use case-insensitive comparison using toLower() function.
For count queries, ensure you return meaningful column names.
Use the provided Chat History to filter results if the question refers to previous context (e.g., "they", "those people").

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Examples: Here are a few examples of generated Cypher statements for particular questions:

# How many Python programmers do we have?
MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
WHERE toLower(s.id) = toLower("Python")
RETURN count(p) AS pythonProgrammers

# Who has React skills?
MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
WHERE toLower(s.id) = toLower("React")
RETURN p.id AS name

# Find people with both Python and Django skills
MATCH (p:Person)-[:HAS_SKILL]->(s1:Skill), (p)-[:HAS_SKILL]->(s2:Skill)
WHERE toLower(s1.id) = toLower("Python") AND toLower(s2.id) = toLower("Django")
RETURN p.id AS name

The question is:
The question is:
{question}

Chat History (for context):
{chat_history}"""

        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["schema", "question", "chat_history"],
            template=CYPHER_GENERATION_TEMPLATE
        )

        # Custom QA prompt for better handling of numeric results
        CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the result(s) of a Cypher query that was executed against a knowledge graph.
Information is provided as a list of records from the graph database.

Guidelines:
- If the information contains count results or numbers, state the exact count clearly.
- For count queries that return 0, say "There are 0 [items]" - this is a valid result, not missing information.
- If the information is empty or null, then say you don't know the answer.
- Use the provided information to construct a helpful answer.
- Be specific and mention actual names, numbers, or details from the information.

Information:
{context}

Question: {question}
Helpful Answer:"""

        CYPHER_QA_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
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

    def setup_workflow(self):
        """Setup the LangGraph workflow with Coordinator-Planner-Querier architecture."""
        
        # --- Prompts ---
        
        COORDINATOR_TEMPLATE = """You are the Coordinator of a knowledge graph querying system. 
Your goal is to answer the user's `original_query` using the `gathered_info`.

Original Query: {original_query}
Chat History: {chat_history}
Gathered Information: {gathered_info}
Last Query Result: {latest_query_result}

Instructions:
1. Analyze the `gathered_info` and `latest_query_result`.
2. Determine if you have enough information to answer the `original_query` comprehensively.
3. If yes, output `NEXT: FINISH` followed by the final answer.
4. If no, output `NEXT: PLAN` followed by feedback/instructions for the Planner.

Important:
- If `latest_query_result` contains a list of entities in `context` (e.g. `['name': 'Alice']`), TRUST IT.
- If the query was "Find people with skill X" and you got a list of people, that IS the answer. Do not ask to "verify" their skills unless explicitly requested.
- If the answer is "I don't know" or empty, then ask for a retrial with a broader strategy.

Response Format:
NEXT: [FINISH or PLAN]
[Reasoning or Final Answer]"""

        COORDINATOR_PROMPT = PromptTemplate(
            input_variables=["original_query", "chat_history", "gathered_info", "latest_query_result"],
            template=COORDINATOR_TEMPLATE
        )

        PLANNER_TEMPLATE = """You are the Planner. Your job is to formulate a SINGLE, SPECIFIC natural language question for the Graph Querier.
        
Original Query: {original_query}
Coordinator Feedback: {coordinator_feedback}

Schema Overview:
- Nodes: Person, Skill, Company, Project, Certification, University, RFP
- Relationships: HAS_SKILL, WORKED_AT, WORKED_ON, EARNED, STUDIED_AT, NEEDS
- Properties: names are usually 'id' or 'name'. Skills are single words.

Strategies:
- **Concept Expansion**: If the feedback mentions a broad role (e.g. "Frontend"), create a query checking for specific skills (React, Vue, JS).
- **Simplification**: If the previous query was too complex, break it down.
- **Variation**: If the previous query returned 0, try synonyms or less restrictive conditions.

Output ONLY the new specific question to ask the graph."""

        PLANNER_PROMPT = PromptTemplate(
            input_variables=["original_query", "coordinator_feedback"],
            template=PLANNER_TEMPLATE
        )

        # --- Nodes ---

        def coordinator_node(state: AgentState) -> AgentState:
            """Decide next step based on state."""
            original_query = state["original_query"]
            chat_history = state.get("chat_history", "")
            gathered_info = state.get("gathered_info", [])
            latest_result = state.get("latest_query_result", {})
            iterations = state.get("iterations", 0)
            
            # Safety break
            if iterations > 5:
                # Ask LLM to summarize partial results
                cleanup_prompt = f"""You are the Coordinator. The system has reached its maximum iteration limit.
                
Original Query: {original_query}
Gathered Information: {gathered_info}

Task:
1. Summarize clearly what information was found (from the Gathered Information).
2. State clearly what information is still missing or could not be verified.
3. Be polite and concise.
4. Format the response nicely as a natural language answer. Do NOT show raw JSON structures."""
                
                try:
                    response = self.llm.invoke(cleanup_prompt).content.strip()
                except:
                    response = "I couldn't complete the request in time. Please try a more specific query."
                    
                return {
                    "final_answer": response,
                    "iterations": iterations
                }

            formatted_prompt = COORDINATOR_PROMPT.format(
                original_query=original_query,
                chat_history=chat_history,
                gathered_info=str(gathered_info),
                latest_query_result=str(latest_result)
            )

            try:
                response = self.llm.invoke(formatted_prompt).content.strip()
                
                if response.startswith("NEXT: FINISH"):
                    final_answer = response.replace("NEXT: FINISH", "").strip()
                    return {"final_answer": final_answer, "iterations": iterations}
                
                elif response.startswith("NEXT: PLAN"):
                    feedback = response.replace("NEXT: PLAN", "").strip()
                    logger.info(f"Coordinator Feedback: {feedback}")
                    return {"coordinator_feedback": feedback, "iterations": iterations + 1}
                
                else:
                    # Fallback
                    return {"coordinator_feedback": "Please continue searching.", "iterations": iterations + 1}
                    
            except Exception as e:
                logger.error(f"Coordinator error: {e}")
                return {"final_answer": "Error in coordination.", "iterations": iterations}

        def planner_node(state: AgentState) -> AgentState:
            """Formulate the next query."""
            original_query = state["original_query"]
            feedback = state.get("coordinator_feedback", "")
            
            formatted_prompt = PLANNER_PROMPT.format(
                original_query=original_query,
                coordinator_feedback=feedback
            )
            
            try:
                plan_step = self.llm.invoke(formatted_prompt).content.strip()
                logger.info(f"Planned step: {plan_step}")
                return {"current_plan_step": plan_step}
            except Exception as e:
                logger.error(f"Planner error: {e}")
                return {"current_plan_step": original_query} # Fallback to original

        def graph_querier_node(state: AgentState) -> AgentState:
            """Execute the planned query."""
            question = state["current_plan_step"]
            chat_history = state.get("chat_history", "")
            
            try:
                # Reuse the existing QA chain
                result = self.qa_chain.invoke({
                    "query": question,
                    "chat_history": chat_history
                })
                
                answer = result.get("result", "No answer")
                steps = result.get("intermediate_steps", [])
                cypher = steps[0].get("query", "") if steps else ""
                context = steps[1].get("context", []) if len(steps) > 1 else []
                
                query_result = {
                    "asked": question,
                    "cypher": cypher,
                    "context": context,
                    "answer": answer
                }
                
                # Update gathered info
                new_gathered = state.get("gathered_info", []) + [query_result]
                
                return {
                    "latest_query_result": query_result,
                    "gathered_info": new_gathered
                }
                
            except Exception as e:
                error_result = {"asked": question, "error": str(e)}
                new_gathered = state.get("gathered_info", []) + [error_result]
                return {
                    "latest_query_result": error_result,
                    "gathered_info": new_gathered
                }

        def check_finish(state: AgentState) -> str:
            if state.get("final_answer"):
                return "finish"
            return "continue"

        # --- Graph Construction ---
        workflow = StateGraph(AgentState)
        
        workflow.add_node("coordinator", coordinator_node)
        workflow.add_node("planner", planner_node)
        workflow.add_node("graph_querier", graph_querier_node)
        
        # Start at Coordinator
        workflow.add_edge(START, "coordinator")
        
        # Coordinator decides to End or Plan
        workflow.add_conditional_edges(
            "coordinator",
            check_finish,
            {
                "finish": END,
                "continue": "planner"
            }
        )
        
        workflow.add_edge("planner", "graph_querier")
        workflow.add_edge("graph_querier", "coordinator")
        
        self.app = workflow.compile()
        logger.info("✓ LangGraph workflow initialized (Coordinator-Planner-Querier)")

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

    def query_graph(self, question: str, conversation_id: str = None) -> Dict[str, Any]:
        """Execute a natural language query against the graph.

        Args:
            question: Natural language question
            conversation_id: Optional ID to maintain conversation context

        Returns:
            Dict containing query results, metadata, and conversation_id
        """
        if not conversation_id:
            # Stateless mode - do not use or store history
            chat_history_str = "No history."
        else:
            # Initialize history for new conversation
            if conversation_id not in self.chat_histories:
                self.chat_histories[conversation_id] = []
                
            # Format history string (last 5 turns)
            history_list = self.chat_histories[conversation_id][-5:]
            chat_history_str = "\n".join([f"Human: {q}\nAI: {a}" for q, a in history_list]) if history_list else "No history."

        try:
            logger.info(f"Executing query via LangGraph Coordinator: {question}")

            initial_state = {
                "original_query": question,
                "chat_history": chat_history_str,
                "coordinator_feedback": "Initial request",
                "gathered_info": [],
                "iterations": 0,
                "final_answer": ""
            }
            
            # Run the workflow
            final_state = self.app.invoke(initial_state)

            # Extract results
            response = {
                "question": question,
                "answer": final_state.get("final_answer", "No answer"),
                "conversation_id": conversation_id,
                "success": True if final_state.get("final_answer") and "Error" not in final_state.get("final_answer") else False
            }
            
            # Add simplified cypher query field for compatibility with UI
            # We take the cypher from the LAST gathered info if available
            gathered = final_state.get("gathered_info", [])
            if gathered:
                response["cypher_query"] = gathered[-1].get("cypher", "")
            else:
                response["cypher_query"] = ""
            
            # Update history with success only if conversation_id exists
            if conversation_id:
                self.chat_histories[conversation_id].append((question, response["answer"]))

            logger.info(f"✓ Query executed successfully")
            return response

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "cypher_query": "",
                "question": question,
                "answer": f"Error: {str(e)}",
                "cypher_query": "",
                "conversation_id": conversation_id,
                "success": False
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

            print(f"\n{'='*60}")
            print(f"Category: {cat}")
            print(f"{'='*60}")

            for question in self.example_queries[cat]:
                print(f"\n🔍 Query: {question}")
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

    def custom_query(self, question: str) -> None:
        """Execute a custom user query.

        Args:
            question: User's natural language question
        """
        print(f"\n🔍 Custom Query: {question}")
        print("-" * 50)

        result = self.query_graph(question)

        if result["success"]:
            print(f"📊 Generated Cypher: {result['cypher_query']}")
            print(f"💡 Answer: {result['answer']}")
        else:
            print(f"❌ Error: {result['answer']}")

    def validate_graph_content(self) -> bool:
        """Validate that the graph has content for querying."""
        # First, get all available node labels
        try:
            labels_result = self.graph.query(
                "CALL db.labels() YIELD label RETURN label ORDER BY label"
            )
            available_labels = [row["label"] for row in labels_result if row["label"] != "__Entity__"]
        except:
            available_labels = []

        # Build validation queries based on available labels
        validation_queries = [
            ("Total nodes", "MATCH (n) RETURN count(n) as count"),
            ("Total relationships", "MATCH ()-[r]->() RETURN count(r) as count")
        ]

        # Add queries for available node types
        common_labels = ["Person", "Company", "Skill", "University", "Certification", "Project", "Location"]
        for label in common_labels:
            if label in available_labels:
                validation_queries.append((f"{label} count", f"MATCH (n:{label}) RETURN count(n) as count"))

        print("\n📊 Graph Validation")
        print("-" * 30)

        total_nodes = 0
        person_count = 0

        for description, query in validation_queries:
            try:
                result = self.graph.query(query)
                count = result[0]["count"] if result else 0
                print(f"{description}: {count:,}")

                if description == "Total nodes":
                    total_nodes = count
                elif description == "Person count":
                    person_count = count

            except Exception as e:
                print(f"{description}: Error - {e}")

        # Show relationship breakdown based on available relationships
        print("\n🔗 Key Relationships:")

        # Get all relationship types
        try:
            rel_result = self.graph.query(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
            )
            available_rels = [row["relationshipType"] for row in rel_result]
        except:
            available_rels = []

        # Build relationship queries for common patterns
        common_rel_patterns = [
            ("Person-Skill connections", "MATCH (p:Person)-[r]->(s:Skill) RETURN count(*) as count"),
            ("Person-Company connections", "MATCH (p:Person)-[r]->(c:Company) RETURN count(*) as count"),
            ("Person-University connections", "MATCH (p:Person)-[r]->(u:University) RETURN count(*) as count"),
            ("Person-Location connections", "MATCH (p:Person)-[r]->(l:Location) RETURN count(*) as count"),
            ("Person-Certification connections", "MATCH (p:Person)-[r]->(cert:Certification) RETURN count(*) as count")
        ]

        for description, query in common_rel_patterns:
            try:
                result = self.graph.query(query)
                count = result[0]["count"] if result else 0
                if count > 0:  # Only show relationships that exist
                    print(f"{description}: {count:,}")
            except Exception as e:
                # Silently skip if node types don't exist
                pass

        # Show available labels and relationships
        if available_labels:
            print(f"\n🏷️  Available Node Types: {', '.join(available_labels)}")
        if available_rels:
            print(f"🔗 Available Relationships: {', '.join(available_rels)}")

        # Check if we have enough data
        if person_count == 0:
            print("\n⚠️  Warning: No Person nodes found in the graph!")
            print("Please run comprehensive_pipeline.py first to populate the graph.")
            print("Or check: uv run python 0_setup.py --check")
            return False

        if total_nodes < 10:
            print(f"\n⚠️  Warning: Only {total_nodes} nodes found. Consider generating more data.")

        print(f"\n✅ Graph validation successful: {person_count} people, {total_nodes} total nodes")
        return True

    def show_graph_schema(self) -> None:
        """Display the current graph schema."""
        print("\n📋 Graph Schema")
        print("-" * 30)

        try:
            # Get all node labels using CALL db.labels()
            try:
                labels_result = self.graph.query("CALL db.labels() YIELD label RETURN label ORDER BY label")
                node_labels = [row["label"] for row in labels_result if row["label"] != "__Entity__"]
            except:
                # Fallback method
                labels_result = self.graph.query(
                    "MATCH (n) RETURN DISTINCT labels(n) as labels ORDER BY labels[0]"
                )
                node_labels = []
                for label_row in labels_result:
                    labels = label_row["labels"]
                    if labels:
                        main_labels = [l for l in labels if l != "__Entity__"]
                        node_labels.extend(main_labels)
                node_labels = sorted(list(set(node_labels)))

            print("🏷️  Node Types:")
            if node_labels:
                for label in node_labels:
                    # Get count for each label
                    try:
                        count_result = self.graph.query(f"MATCH (n:{label}) RETURN count(n) as count")
                        count = count_result[0]["count"] if count_result else 0
                        print(f"   • {label} ({count:,} nodes)")
                    except:
                        print(f"   • {label}")
            else:
                print("   No node types found")

            # Get relationship types
            try:
                rel_result = self.graph.query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
                rel_types = [row["relationshipType"] for row in rel_result]
            except:
                # Fallback method
                rel_result = self.graph.query(
                    "MATCH ()-[r]->() RETURN DISTINCT type(r) as rel_type ORDER BY rel_type"
                )
                rel_types = [row["rel_type"] for row in rel_result]

            print("\n🔗 Relationship Types:")
            if rel_types:
                for rel_type in rel_types:
                    # Get count for each relationship type
                    try:
                        count_result = self.graph.query(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                        count = count_result[0]["count"] if count_result else 0
                        print(f"   • {rel_type} ({count:,} relationships)")
                    except:
                        print(f"   • {rel_type}")
            else:
                print("   No relationships found")

            # Show sample data
            print("\n📊 Sample Data:")

            # Sample people (try different property names)
            people_queries = [
                ("MATCH (p:Person) RETURN p.id as name LIMIT 3", "id"),
                ("MATCH (p:Person) RETURN p.name as name LIMIT 3", "name"),
                ("MATCH (p:Person) RETURN keys(p)[0] as prop, p[keys(p)[0]] as name LIMIT 3", "first_property")
            ]

            people_found = False
            for query, prop_type in people_queries:
                try:
                    people_sample = self.graph.query(query)
                    if people_sample and people_sample[0].get("name"):
                        print(f"   People (via {prop_type}):")
                        for person in people_sample[:3]:
                            if person.get("name"):
                                print(f"     - {person['name']}")
                        people_found = True
                        break
                except:
                    continue

            if not people_found:
                try:
                    # Show first 3 Person nodes with any available properties
                    people_sample = self.graph.query("MATCH (p:Person) RETURN p LIMIT 3")
                    if people_sample:
                        print("   People (raw properties):")
                        for person in people_sample:
                            node_props = person.get("p", {})
                            if isinstance(node_props, dict) and node_props:
                                # Show first property value
                                first_key = list(node_props.keys())[0]
                                print(f"     - {node_props[first_key]} ({first_key})")
                except:
                    pass

            # Sample skills
            skills_queries = [
                ("MATCH (s:Skill) RETURN s.id as skill LIMIT 5", "id"),
                ("MATCH (s:Skill) RETURN s.name as skill LIMIT 5", "name")
            ]

            for query, prop_type in skills_queries:
                try:
                    skills_sample = self.graph.query(query)
                    if skills_sample and skills_sample[0].get("skill"):
                        print(f"   Skills (via {prop_type}):")
                        for skill in skills_sample[:5]:
                            if skill.get("skill"):
                                print(f"     - {skill['skill']}")
                        break
                except:
                    continue

        except Exception as e:
            print(f"Error displaying schema: {e}")
            print("\nFallback schema:")
            try:
                print(self.graph.schema)
            except:
                print("Unable to retrieve schema")

    def interactive_mode(self) -> None:
        """Start interactive query mode."""
        print("\n🎯 Interactive GraphRAG Query Mode")
        print("Type your questions or 'quit' to exit")
        print("-" * 40)

        while True:
            try:
                question = input("\n❓ Your question: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if not question:
                    continue

                self.custom_query(question)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to demonstrate GraphRAG capabilities on CV-only knowledge graph."""
    print("CV Knowledge Graph - GraphRAG Query System")
    print("Natural Language Queries for CV Data")
    print("=" * 50)

    try:
        # Initialize system
        system = CVGraphRAGSystem()

        # Validate graph content
        if not system.validate_graph_content():
            return

        # Show schema
        system.show_graph_schema()

        # Menu system
        while True:
            print("\n🎯 GraphRAG Demo Options:")
            print("1. Basic Information queries")
            print("2. Skill-based queries")
            print("3. Company Experience queries")
            print("4. Education Background queries")
            print("5. Professional Experience queries")
            print("6. Multi-hop Reasoning queries")
            print("7. Certification Analysis queries")
            print("8. Run ALL example queries")
            print("9. Interactive query mode")
            print("0. Exit")

            choice = input("\nSelect option (0-9): ").strip()

            if choice == "1":
                system.run_example_queries("Basic Information")
            elif choice == "2":
                system.run_example_queries("Skill-based Queries")
            elif choice == "3":
                system.run_example_queries("Company Experience")
            elif choice == "4":
                system.run_example_queries("Education Background")
            elif choice == "5":
                system.run_example_queries("Professional Experience")
            elif choice == "6":
                system.run_example_queries("Multi-hop Reasoning")
            elif choice == "7":
                system.run_example_queries("Certification Analysis")
            elif choice == "8":
                system.run_example_queries()
            elif choice == "9":
                system.interactive_mode()
            elif choice == "0":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 0-9.")

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()