import logging
import uuid
from typing import Dict, Any, List, Optional
from src.rag.graph.querier import CVGraphRAGSystem
from src.rag.naive.querier import NaiveRAGQuerier
from src.rag.agent.workflow import CVGraphAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryDataProxy:
    """
    Proxy to orchestrate querying between Graph RAG and Naive RAG.
    Manages conversation history centrally.
    """

    def __init__(self):
        """Initialize the query proxy."""
        self.graph_rag = CVGraphRAGSystem()
        self.naive_rag = NaiveRAGQuerier()
        
        # Initialize the agent with the sub-systems
        # Assuming CVGraphRAGSystem has an 'llm' attribute we can reuse, or we rely on agent to reuse it.
        # Actually CVGraphAgent takes (llm, graph_rag, naive_rag). 
        # graph_rag has 'llm' attribute (ChatOpenAI).
        self.agent = CVGraphAgent(
            llm=self.graph_rag.llm, 
            graph_rag=self.graph_rag, 
            naive_rag=self.naive_rag
        )
        
        # Centralized chat history storage
        # Format: {conversation_id: [(human, ai), ...]}
        self.chat_histories: Dict[str, List[tuple]] = {}

    def query(self, question: str, mode: str = "graph", conversation_id: str = None) -> Dict[str, Any]:
        """
        Execute a query against the selected RAG system.
        
        Args:
            question: The user's question
            mode: "graph" or "naive"
            conversation_id: Unique ID for the conversation
            
        Returns:
            Dict containing the answer and metadata
        """
        if not conversation_id:
            conversation_id = uuid.uuid4().hex
            
        # Ensure conversation exists in history
        if conversation_id not in self.chat_histories:
            self.chat_histories[conversation_id] = []
            
        # Prepare context from history (last 5 turns)
        history_list = self.chat_histories[conversation_id][-5:]
        chat_history_str = "\n".join([f"Human: {q}\nAI: {a}" for q, a in history_list]) if history_list else "No history."

        response = {
            "conversation_id": conversation_id,
            "mode": mode,
            "success": False
        }

        try:
            if mode == "agent":
                # Agent RAG (Multi-Step / LangGraph)
                # Pass history string to agent
                result = self.agent.query(question, chat_history_str)
                response.update(result)

            elif mode == "graph":
                # Simple Graph RAG
                # We update the history in graph_rag internally just in case (legacy), 
                # but really we should pass it or handle it here.
                # The updated CVGraphRAGSystem.query_graph accepts conversation_id to handle internal history,
                # but since we are centralizing it here, we might just pass the string if we updated signatures,
                # OR we just rely on `conversation_id` passing.
                # Let's pass conversation_id as before, assuming query_graph handles it.
                # NOTE: We renamed query_graph_simple to query_graph.
                self.graph_rag.chat_histories[conversation_id] = self.chat_histories[conversation_id]
                result = self.graph_rag.query_graph(question, conversation_id)
                response.update(result)
                
            elif mode == "naive":
                result = self.naive_rag.query(question)
                response.update(result)
                response["conversation_id"] = conversation_id
            
            else:
                response["answer"] = f"Unknown mode: {mode}"
                return response

            # Update our central history if successful
            if response.get("success", False):
                answer = response.get("answer", "")
                self.chat_histories[conversation_id].append((question, answer))
                
        except Exception as e:
            logger.error(f"Query failed in mode {mode}: {e}")
            response["answer"] = f"Error processing query: {str(e)}"
            
        return response
