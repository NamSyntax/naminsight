import uuid
import time
import chainlit as cl
from langchain_core.messages import HumanMessage
import logging
from langgraph.checkpoint.memory import MemorySaver
from src.core.workflow import app_workflow

# Setup unified logging stream
logging.basicConfig(level=logging.INFO)

@cl.on_chat_start
async def on_chat_start():
    # Initialize a unique threaded session state
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    
    await cl.Message(
        content="**NAMINSIGHT AGENT**\n\nThe Controlled Autonomous Agent is locked to this thread. It will remember previous conversational turns dynamically. Submit a task to begin interactive execution."
    ).send()

import os

@cl.on_message
async def handle_message(message: cl.Message):
    """Sync Chainlit UI to Graph runtime."""
    thread_id = cl.user_session.get("thread_id")
    
    last_request_time = cl.user_session.get("last_request_time", 0)
    if time.time() - last_request_time < 10:
        await cl.Message(content="⏳ **Rate Limit Exceeded:** Please wait a few seconds before sending another request.").send()
        return
    cl.user_session.set("last_request_time", time.time())
    
    config = {"configurable": {"thread_id": thread_id}}
    
    ui_step = cl.Step(name="Technical Execution Trace")
    ui_step.collapsed = True
    await ui_step.send()
    final_answer = ""
    
    try:
        # reset loop counter for new inputs
        inputs = {"messages": [HumanMessage(content=message.content)], "iteration_count": 0}
        
        while True:
            # stream graph states
            async for output in app_workflow.astream(inputs, config=config):
                for node_name, state_update in output.items():
                    
                    # skip graph-internal signals
                    if not isinstance(state_update, dict):
                        continue
                        
                    # Render the Recovery Matrix visual cue
                    if node_name == "retry_manager":
                        retry_count = state_update.get("retry_count")
                        if retry_count is not None and retry_count > 0:
                            await ui_step.stream_token(f"\n\n⚠️ **Error detected. Self-debugging attempt {retry_count}/3...**\n")
                        continue
                    
                    if node_name in ["governor_proxy", "finalize"]:
                        if node_name == "finalize" and "final_response" in state_update:
                            final_answer = state_update.get("final_response", "")
                        continue
                        
                    await ui_step.stream_token(f"\n\n--- 🔄 **{node_name.upper()} ACTIVITY** ---\n")
                    
                    messages = state_update.get("messages", [])
                    for msg in messages:
                        if getattr(msg, "type", "") == "ai":
                            content = msg.content
                            # trace tool plans only, hide final answers
                            if '"tool"' in content and '"payload"' in content:
                                await ui_step.stream_token(f"\n**{node_name.capitalize()} Output (Tool Plan):**\n{content}\n")
                        elif getattr(msg, "type", "") == "system":
                            if "Tool Execution" in getattr(msg, "content", "") or "Sandbox Execution" in getattr(msg, "content", ""):
                                await ui_step.stream_token(f"\n**Tool Evaluation Results:**\n```text\n{msg.content}\n```\n")
                            elif "Critic Evaluation" in getattr(msg, "content", ""):
                                await ui_step.stream_token(f"\n**{msg.content}**\n")
                                
            await ui_step.update()
            
            # check HITL pause
            state = app_workflow.get_state(config)
            if state.next and "dispatcher" in state.next:
                
                # auto-approve if we're in a self-correction retry loop
                if state.values.get("retry_count", 0) > 0:
                    inputs = None
                    await ui_step.stream_token("\n*--- ⚙️ Auto-Approved System Retry ---*\n")
                    continue
                    
                # prompt secure execution
                res = await cl.AskActionMessage(
                    content="⚠️ **Security Check:** Architect wants to execute a tool. Do you approve?",
                    actions=[
                        cl.Action(name="approve", payload={"value": "yes"}, label="✅ Approve & Execute"),
                        cl.Action(name="reject", payload={"value": "no"}, label="❌ Reject Plan")
                    ],
                    timeout=300
                ).send()
                
                # handle approval
                if res and res.get("payload", {}).get("value") == "yes":
                    inputs = None
                    await ui_step.stream_token("\n\n*--- 🟢 User Approved Execution ---*\n")
                    continue
                else:
                    # handle rejection & abort
                    await ui_step.stream_token("\n\n*--- 🔴 User Rejected Execution ---*\n")
                    await ui_step.update()
                    await cl.Message(content="🚫 **CANCELED.** Plan rejected. Please provide more details to recreate the plan.").send()
                    break # Break out of the while loop to stop LangGraph progression
            else:
                # graph complete
                break
        
        # Output the final answer outside the trace
        if final_answer and final_answer != "✅ Workflow Execution Complete.":
            await cl.Message(content=final_answer).send()
        elif "dispatcher" not in state.next: # Only show completion if not interrupted
            await cl.Message(content="✅ **Workflow Execution Complete**.").send()
            
        # Expose generated assets
        if os.path.exists(os.path.join("exports", "output.png")):
            elements = [
                cl.Image(name="output.png", display="inline", path=os.path.join("exports", "output.png")),
                cl.File(name="output.png", path=os.path.join("exports", "output.png"), display="inline")
            ]
            await cl.Message(content="**Generated Output / Assets:**", elements=elements).send()
            
            try:
                os.remove(os.path.join("exports", "output.png"))
            except Exception:
                pass
        
    except Exception as e:
        await ui_step.stream_token(f"\n\n⚠️ **Critical Execution Exception:**\n{str(e)}\n")
        await ui_step.update()
        await cl.Message(content="❌ **Agent Pipeline failed. Consult the technical breakdown above.**").send()
