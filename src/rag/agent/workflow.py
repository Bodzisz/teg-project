import logging
import operator
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_core.prompts.prompt import PromptTemplate
from langgraph.graph import StateGraph, START, END
from src.rag.agent.state import AgentState

# Configure logging
logger = logging.getLogger(__name__)

class CVGraphAgent:
    """
    Manages the LangGraph agent workflow for the CV Knowledge Graph.
    Implements Coordinator-Planner-Querier architecture.
    """
    def __init__(self, llm, graph_rag, naive_rag):
        """
        Initialize the agent with necessary dependencies.
        
        Args:
            llm: ChatOpenAI instance for driving the agent nodes
            graph_rag: CVGraphRAGSystem instance (for graph queries)
            naive_rag: NaiveRAGQuerier instance (for vector fallback)
        """
        self.llm = llm
        self.graph_rag = graph_rag
        self.naive_rag = naive_rag
        self.app = None
        self._setup_workflow()
    
    def _setup_workflow(self):
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
- If the question is about relative date (e.g. tomorrow, yesterday, last week, next week), use {today} as the date. Treat Q1, Q2, Q3, Q4 as 1st, 2nd, 3rd, 4th quarter of current year.

Response Format:
NEXT: [FINISH or PLAN]
[Reasoning or Final Answer]"""

        COORDINATOR_PROMPT = PromptTemplate(
            input_variables=["original_query", "chat_history", "gathered_info", "latest_query_result"],
            partial_variables={"today": datetime.now().strftime("%Y-%m-%d")},
            template=COORDINATOR_TEMPLATE
        )

        PLANNER_TEMPLATE = """You are the Planner. Your job is to formulate a SINGLE, SPECIFIC step for the Querier.
        
Original Query: {original_query}
Coordinator Feedback: {coordinator_feedback}

Schema Overview:
{schema}
- Properties: names are usually 'id' or 'name'. Skills are single words.

Capabilities:
1. **Graph Search**: For structured data (counts, specific relationships, explicit filters).
2. **Vector Search**: For unstructured concepts (e.g. "leadership style", "soft skills", "project details") or when Graph Search fails.

Strategies:
- **Concept Expansion**: If the feedback mentions a broad role (e.g. "Frontend"), create a query checking for specific skills (React, Vue, JS).
- **Simplification**: If the previous query was too complex, break it down.
- **Variation**: If the previous query returned 0, try synonyms or less restrictive conditions.
- **Fallback**: If Graph Search keeps failing or the question is about unstructured text, use Vector Search.
- **Matching**: In where conditions prefer to use contains() function instead of = for better matching. For example, if user asks for "security" projects, use category contains "security" instead of category = "security".
- **Generic, Broad Questions**: If the question is generic, answer it in the context of assignments to projects, people skills and relations in graph
- **Dates**: If the question is about relative date (e.g. tomorrow, yesterday, last week, next week), use {today} as the date. Treat Q1, Q2, Q3, Q4 as 1st, 2nd, 3rd, 4th quarter of current year.

Examples:
- User: "Who knows React and Vue?"
  Plan: "Find people who have BOTH React AND Vue skills." (Use single query with AND)
- User: "Find best Python dev"
  Plan: "Find people with 'Python' skill and return their project count."
- User: "What is the leadership style of John?"
  Plan: "VECTOR: leadership style of John"

Output Format:
- For Graph Search: Just the natural language question (e.g. "Find people with React").
- For Vector Search: Prefix with `VECTOR:` (e.g. "VECTOR: Find candidates with strong leadership style")."""

        PLANNER_PROMPT = PromptTemplate(
            input_variables=["original_query", "coordinator_feedback", "schema"],
            partial_variables={"today": datetime.now().strftime("%Y-%m-%d")},
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
            
            # Access schema from the graph_rag system
            try:
                schema = self.graph_rag.graph.get_schema
            except:
                schema = "Schema access unavailable."

            formatted_prompt = PLANNER_PROMPT.format(
                original_query=original_query,
                coordinator_feedback=feedback,
                schema=schema
            )
            
            try:
                plan_step = self.llm.invoke(formatted_prompt).content.strip()
                logger.info(f"Planned step: {plan_step}")
                return {"current_plan_step": plan_step}
            except Exception as e:
                logger.error(f"Planner error: {e}")
                return {"current_plan_step": original_query} # Fallback to original

        def graph_querier_node(state: AgentState) -> AgentState:
            """Execute the planned query by delegating to GraphRAG or NaiveRAG."""
            step = state["current_plan_step"]
            
            try:
                # Check for Vector Search request
                if step.strip().upper().startswith("VECTOR:"):
                    # Execute Vector Search
                    query_text = step.replace("VECTOR:", "").strip()
                    logger.info(f"Delegate to Vector Search: {query_text}")
                    
                    vector_result = self.naive_rag.query(query_text)
                    
                    answer = vector_result.get("answer", "No answer from vector search.")
                    context_info = vector_result.get("context_info", [])
                    
                    query_result = {
                        "asked": step,
                        "type": "vector",
                        "answer": answer,
                        "context": context_info
                    }
                else:
                    # Execute Graph Search via System
                    logger.info(f"Delegate to Graph System: {step}")
                    
                    graph_res = self.graph_rag.query_graph(step)
                    
                    answer = graph_res.get("answer", "No answer.")
                    cypher = graph_res.get("cypher_query", "")
                    context = graph_res.get("context", [])
                    
                    query_result = {
                        "asked": step,
                        "type": "graph",
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
                error_result = {"asked": step, "error": str(e)}
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
        logger.info("✓ CVGraphAgent Workflow initialized (delegating mode)")

    def query(self, question: str, chat_history_str: str = "No history.") -> Dict[str, Any]:
        """
        Public entry point to run the agent for a question.
        Returns a standardised result dictionary.
        """
        logger.info(f"Agent starting query: {question}")
        initial_state = {
            "original_query": question,
            "chat_history": chat_history_str,
            "coordinator_feedback": "Initial request",
            "gathered_info": [],
            "iterations": 0,
            "final_answer": ""
        }
        
        final_state = self.app.invoke(initial_state)
        
        # Extract results
        answer = final_state.get("final_answer", "No answer")
        
        # Find the last helpful cypher query if any
        gathered = final_state.get("gathered_info", [])
        last_cypher = ""
        for info in reversed(gathered):
            if info.get("type") == "graph" and info.get("cypher"):
                last_cypher = info.get("cypher")
                break
                
        return {
            "question": question,
            "answer": answer,
            "cypher_query": last_cypher,
            "success": "Error" not in answer
        }

    def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Raw LangGraph invoke access if needed."""
        if not self.app:
            raise RuntimeError("Agent workflow not initialized.")
        return self.app.invoke(initial_state)
