import logging
import operator
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_core.prompts.prompt import PromptTemplate
from langgraph.graph import StateGraph, START, END

from src.rag.agent.state import AgentState
from src.core.utils.prompt_loader import load_prompt

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
        prompts_dir = Path(__file__).parent / "prompts"
        
        COORDINATOR_TEMPLATE = load_prompt(prompts_dir / "coordinator.txt")
        
        COORDINATOR_PROMPT = PromptTemplate(
            input_variables=["original_query", "chat_history", "gathered_info", "latest_query_result"],
            partial_variables={"today": datetime.now().strftime("%Y-%m-%d")},
            template=COORDINATOR_TEMPLATE
        )

        PLANNER_TEMPLATE = load_prompt(prompts_dir / "planner.txt")

        PLANNER_PROMPT = PromptTemplate(
            input_variables=["original_query", "coordinator_feedback", "schema"],
            partial_variables={"today": datetime.now().strftime("%Y-%m-%d")},
            template=PLANNER_TEMPLATE
        )
        
        CLEANUP_TEMPLATE = load_prompt(prompts_dir / "cleanup.txt")

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
                cleanup_prompt = CLEANUP_TEMPLATE.format(
                    original_query=original_query,
                    gathered_info=gathered_info
                )
                
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
                    contexts = vector_result.get("contexts", [])
                    
                    query_result = {
                        "asked": step,
                        "type": "vector",
                        "answer": answer,
                        "context": contexts or context_info
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
        contexts = []
        for info in reversed(gathered):
            if info.get("type") == "graph" and info.get("cypher"):
                last_cypher = info.get("cypher")
                break
        for info in gathered:
            if info.get("context"):
                if isinstance(info["context"], list):
                    contexts.extend(info["context"])
                else:
                    contexts.append(info["context"])
                
        return {
            "question": question,
            "answer": answer,
            "cypher_query": last_cypher,
            "contexts": contexts,
            "success": "Error" not in answer
        }

    def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Raw LangGraph invoke access if needed."""
        if not self.app:
            raise RuntimeError("Agent workflow not initialized.")
        return self.app.invoke(initial_state)
