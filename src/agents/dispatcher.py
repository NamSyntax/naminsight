import json
import logging
from typing import Dict, Any
from langchain_core.messages import AIMessage, SystemMessage
from src.core.state import GraphState
from src.core.llm_factory import LLMFactory
from src.tools.python_sandbox import execute_python_code
from src.tools.rag_engine import RAGTool
from src.tools.sql_engine import SQLEngine

logger = logging.getLogger(__name__)

DISPATCHER_SYSTEM_PROMPT = """You are the Dispatcher Agent, the tactical execution layer.
Your job is to read the Architect's plan and select the right tool to execute it.
Available tools:
1. "python": For data processing, algorithm execution, or general code. Payload needs "code" string. NEVER use plt.show(), ALWAYS use plt.savefig('/scratch/output.png').
2. "rag": For retrieving context from the vector database. Payload needs "query" string.
3. "sql": For querying the local PostgreSQL database (SELECT only). Payload needs "query" string.

CRITICAL INSTRUCTION: Do NOT alter the Architect's SQL query or Python code. Pass the exact code/query string provided by the Architect into your syntax without modifications.

Respond ONLY with a valid JSON object matching this schema:
{
  "tool": "python" | "rag" | "sql",
  "payload": {
     // for python: {"code": "print(1)"}
     // for rag: {"query": "search query", "limit": 3}
     // for sql: {"query": "SELECT * from users"}
  }
}"""

def extract_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
    return json.loads(content)

async def dispatcher_node(state: GraphState) -> Dict[str, Any]:
    logger.info("--- DISPATCHER NODE ---")
    llm = LLMFactory.get_dispatcher_llm()
    
    current_plan = state.get("current_plan", "")
    iteration_count = state.get("iteration_count", 0)
    retry_count = state.get("retry_count", 0)
    critic_feedback = state.get("critic_feedback", {})
    tool_results = state.get("tool_results", [])
    
    system_message = SystemMessage(content=DISPATCHER_SYSTEM_PROMPT)
    
    prompt_content = f"Architect's Plan Payload:\n\n{current_plan}\n\n"
    if retry_count > 0 and tool_results:
        prompt_content += f"PREVIOUS ATTEMPT FAILED. Retry {retry_count}/3\n"
        prompt_content += f"Critic Feedback: {json.dumps(critic_feedback, indent=2)}\n"
        prompt_content += f"Previous Tool Result: {json.dumps(tool_results[-1], indent=2)}\n\n"
        prompt_content += "Fix the error and output a new corrected Code/Query based on the feedback.\n\n"
        
    prompt_content += "Select and configure the exact tool to execute."
    prompt_message = AIMessage(content=prompt_content)
    
    response = await llm.ainvoke([system_message, prompt_message])
    messages_out = [response]
    
    try:
        action = extract_json(response.content)
        tool_name = action.get("tool")
        payload = action.get("payload", {})
        
        logger.info(f"Dispatcher selected tool: {tool_name}")
        
        if tool_name == "python":
            code = payload.get("code", "")
            sandbox_res = await execute_python_code(code)
            result = sandbox_res
            log_output = f"Sandbox output: {sandbox_res.get('output', '')}\nError: {sandbox_res.get('error', '')}"
        elif tool_name == "rag":
            rag = RAGTool()
            rag_res = await rag.run(**payload)
            result = {"status": "success", "output": rag_res}
            log_output = f"RAG output: {rag_res}"
        elif tool_name == "sql":
            sql = SQLEngine()
            sql_res = await sql.run(**payload)
            result = {"status": "success", "output": sql_res}
            log_output = f"SQL output: {sql_res}"
        else:
            result = {"status": "error", "error": f"Unknown tool: {tool_name}"}
            log_output = f"Error: Unknown tool {tool_name}"
            
        tool_results.append({"tool": tool_name, "payload": payload, "result": result})
        messages_out.append(SystemMessage(content=f"Tool Execution ({tool_name}):\n{log_output}"))
        
    except Exception as e:
        logger.error(f"Dispatcher failed to execute tool: {e}")
        messages_out.append(SystemMessage(content=f"Dispatcher Action Error (Invalid Schema?): {str(e)}"))
        
    return {
        "messages": messages_out,
        "tool_results": tool_results,
        "iteration_count": iteration_count + 1
    }
