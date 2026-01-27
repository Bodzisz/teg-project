from typing import TypedDict, List, Dict, Any, Optional

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
