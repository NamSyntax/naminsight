from langgraph.graph import StateGraph, END
from src.core.state import GraphState
from src.agents.architect import architect_node
from src.agents.dispatcher import dispatcher_node
from src.agents.governor import governor
from src.agents.critic import critic_node
import logging

logger = logging.getLogger(__name__)

async def finalize_node(state: GraphState) -> dict:
    try:
        from src.core.memory import LongTermMemory
        messages = state.get("messages", [])
        user_task = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                user_task = msg.content
                break
                
        current_plan = state.get("current_plan", "")
        if user_task and current_plan:
            ltm = LongTermMemory()
            ltm.store_memory(user_task, current_plan, score=1.0)
            ltm.prune_memory()
            
        import os
        export_dir = os.path.join(os.getcwd(), "exports")
        if os.path.exists(export_dir):
            for file_name in os.listdir(export_dir):
                file_path = os.path.join(export_dir, file_name)
                # cleanup exports, keep active output.png
                if file_name != "output.png" and os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Could not remove residual file {file_path}: {e}")
                        
    except ImportError:
        logger.warning("LongTermMemory module not found, skipping storage.")
    except Exception as e:
        logger.error(f"Failed to store memory: {e}")
        
    final_response = "✅ Workflow Execution Complete."
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai":
            content = msg.content.strip()
            # Filter out any message acting as a Tool Payload
            if '"tool"' in content and '"payload"' in content:
                continue
            final_response = content
            break
            
    return {"final_response": final_response}

def retry_manager_node(state: GraphState) -> dict:
    feedback = state.get("critic_feedback", {})
    evaluation = feedback.get("evaluation", "PASS")
    error_type = feedback.get("error_type", "None")
    retry_count = state.get("retry_count", 0)
    
    if evaluation == "PASS":
        return {"retry_count": 0}
    elif evaluation == "FAIL" and error_type == "Code_Error" and retry_count < 3:
        return {"retry_count": retry_count + 1}
    elif evaluation == "FAIL" and (retry_count >= 3 or error_type == "Logic_Conflict"):
        return {"retry_count": 0} 
    return {}

def route_after_critic(state: GraphState) -> str:
    feedback = state.get("critic_feedback", {})
    evaluation = feedback.get("evaluation", "PASS")
    error_type = feedback.get("error_type", "None")
    
    if evaluation == "PASS":
        return "governor_proxy"
    elif evaluation == "FAIL" and error_type == "Code_Error" and state.get("retry_count", 0) <= 3: 
        return "dispatcher"
    return "architect"

def governor_proxy(state: GraphState): 
    return {}

def route_after_architect(state: GraphState) -> str:
    """Smart routing: Distinguish between Chitchat and Tool Execution"""
    current_plan = state.get("current_plan", "")
    # Strictly check if the Architect generated a JSON tool payload
    if '"tool"' in current_plan and '"payload"' in current_plan:
        return "dispatcher"
    # Otherwise, it's just conversational text. Route to end.
    return "finalize"

# Build LangGraph Application workflow
builder = StateGraph(GraphState)

builder.add_node("architect", architect_node)
builder.add_node("dispatcher", dispatcher_node)
builder.add_node("critic", critic_node)
builder.add_node("retry_manager", retry_manager_node)
builder.add_node("governor_proxy", governor_proxy)
builder.add_node("finalize", finalize_node)

builder.set_entry_point("architect")
builder.add_conditional_edges(
    "architect",
    route_after_architect,
    {
        "dispatcher": "dispatcher",
        "finalize": "finalize"
    }
)
builder.add_edge("dispatcher", "critic")
builder.add_edge("critic", "retry_manager")

builder.add_conditional_edges(
    "retry_manager",
    route_after_critic,
    {
        "governor_proxy": "governor_proxy",
        "dispatcher": "dispatcher",         
        "architect": "architect"            
    }
)

builder.add_conditional_edges(
    "governor_proxy", 
    governor, 
    {
        "critic_or_next_step": "architect", 
        "finalize": "finalize"              
    }
)
builder.add_edge("finalize", END)

from langgraph.checkpoint.memory import MemorySaver
# We must use a checkpointer to enable interruptions
memory = MemorySaver()
app_workflow = builder.compile(checkpointer=memory, interrupt_before=["dispatcher"])
