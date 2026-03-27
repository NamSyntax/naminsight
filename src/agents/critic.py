import json
import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, AIMessage
from src.core.state import GraphState
from src.core.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are the Critic Agent, the Quality Controller for the NamInsight system.
Your goal is to evaluate the latest tool execution result against the Architect's current plan and determine if the sub-task was successfully completed.
If there are errors in the code, knowledge gaps in the RAG return, or logic problems, mark it as FAIL.

Respond ONLY with a valid JSON object matching this schema:
{
    "evaluation": "PASS" | "FAIL",
    "reasoning": "Detailed explanation of why it passed or failed.",
    "error_type": "Knowledge_Gap" | "Code_Error" | "Logic_Conflict" | "None"
}"""

def extract_json(content: str) -> dict:
    # strip markdown wrapping tags
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
    return json.loads(content)

async def critic_node(state: GraphState) -> Dict[str, Any]:
    logger.info("--- CRITIC NODE ---")
    llm = LLMFactory.get_dispatcher_llm()
    
    current_plan = state.get("current_plan", "")
    tool_results = state.get("tool_results", [])
    
    if not tool_results:
        return {"critic_feedback": {"evaluation": "PASS", "reasoning": "No tools executed.", "error_type": "None"}}
        
    latest_result = tool_results[-1]
    
    prompt = f"Architect's Plan:\n{current_plan}\n\nLatest Tool Execution ({latest_result.get('tool')}):\n{json.dumps(latest_result.get('result', {}), indent=2)}\n\nEvaluate the success based on the result vs plan."
    
    sys_msg = SystemMessage(content=CRITIC_SYSTEM_PROMPT)
    user_msg = AIMessage(content=prompt)
    
    try:
        response = await llm.ainvoke([sys_msg, user_msg])
        feedback = extract_json(response.content)
        
        logger.info(f"Critic Evaluation: {feedback.get('evaluation')} | Error: {feedback.get('error_type')}")
        
        status = "PASS" if feedback.get("evaluation") == "PASS" else f"FAIL ({feedback.get('error_type')})"
        msg = f"Critic Evaluation: {status}\nReasoning: {feedback.get('reasoning')}"
        
        return {
            "critic_feedback": feedback,
            "messages": [SystemMessage(content=msg)]
        }
    except Exception as e:
        logger.error(f"Critic failed: {e}")
        return {"critic_feedback": {"evaluation": "PASS", "reasoning": "Critic parser failed.", "error_type": "None"}}
