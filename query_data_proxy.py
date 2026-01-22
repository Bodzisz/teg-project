import logging
import uuid
from typing import Dict, Any, List, Optional
from query_knowledge_graph import CVGraphRAGSystem
from naive_rag_querier import NaiveRAGQuerier

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
            if mode == "graph":
                # For Graph RAG, we can pass the history if the underlying system supports it.
                # Currently CVGraphRAGSystem.query_graph manages its own history but takes conversation_id.
                # Ideally, we should sync them or rely on one.
                # To avoid conflict, we can either:
                # 1. Let Graph RAG manage its own history (it does) and we just pass the ID.
                # 2. Modify Graph RAG to accept history string.
                # Given the current implementation of CVGraphRAGSystem.query_graph, it retrieves history internally using conversation_id.
                # However, if we switch modes, the internal history of one system won't know about the other.
                # So passing history is better.
                # BUT, modifying CVGraphRAGSystem might be invasive.
                # The prompt construction in CVGraphRAGSystem uses self.chat_histories[conversation_id].
                # We can inject our history into the graph_rag instance before querying.
                
                self.graph_rag.chat_histories[conversation_id] = self.chat_histories[conversation_id]
                
                result = self.graph_rag.query_graph(question, conversation_id)
                response.update(result)
                
            elif mode == "naive":
                # Naive RAG implementation in naive_rag_querier.py doesn't seem to explicitly handle chat history in `query` method args
                # checking naive_rag_querier.py again... it just takes `question`.
                # If we want history in Naive RAG, we'd need to modify it or prepend it to the question.
                # For now, we will just query it directly.
                result = self.naive_rag.query(question)
                response.update(result)
                
                # Naive RAG result doesn't natively include conversation_id or history update
                # So we manually construct the answer part for our history
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
