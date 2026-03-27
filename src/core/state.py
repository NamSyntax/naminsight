from typing import TypedDict, Annotated, List, Any, Optional
from langchain_core.messages import BaseMessage
import operator

class GraphState(TypedDict):
    """Strict typed GraphState schema."""
    messages: Annotated[list[BaseMessage], operator.add]
    current_plan: str
    tool_results: list[dict[str, Any]]
    iteration_count: int
    retry_count: int
    critic_feedback: dict[str, Any]
    final_response: str
