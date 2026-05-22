import os
import json
import re
from types import SimpleNamespace
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agent.tools import ALL_TOOLS, TOOL_NAMES
from src.agent.prompts import build_example_selector, build_prompt
from src.agent.memory_manager import memory_manager
from src.core.schemas import LoanDecision, Citation
from src.core.logger_config import app_logger, log_interaction

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 1: LLM INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
PRIMARY_PROVIDER = os.getenv("PRIMARY_PROVIDER", "anthropic").lower()

def get_primary_llm():
    """Initialize the primary model based on provider preference."""
    if PRIMARY_PROVIDER == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2048,
            temperature=0.1,
        )
    else:
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=2048,
            temperature=0.1,
        )

def get_fallback_llm():
    """
    Fallback model to ensure resilience.
    Uses the opposite of the primary provider.
    """
    if PRIMARY_PROVIDER == "openai":
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=1500,
            temperature=0.1,
        )
    else:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1500,
            temperature=0.1,
        )

# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 4: Build few-shot example selector (done once at startup)
# ─────────────────────────────────────────────────────────────────────────────
app_logger.info("Building semantic example selector...")
try:
    example_selector = build_example_selector(k=3)
    app_logger.info("Semantic example selector ready.")
except Exception as e:
    app_logger.warning(f"Failed to build example selector, running without few-shot: {e}")
    example_selector = None


# ─────────────────────────────────────────────────────────────────────────────
# GUARDRAIL CHAIN: Topic filter before main processing
# ─────────────────────────────────────────────────────────────────────────────
from src.core.schemas import LoanDecision
from src.hitl.manager import hitl_manager
from src.prompt_manager.loader import prompt_manager

def check_guardrail(user_text: str) -> bool:
    """
    Quick check to ensure the query is banking/loan related.
    Includes fallback logic if the primary model fails.
    
    Returns:
        True if the query is safe (on-topic), False if off-topic.
    """
    # OBJECTIVE 8: LLM Guardrails (Fail-fast mechanism)
    guardrail_prompt = prompt_manager.get_chat_prompt("guardrail_prompt", is_system=True)

    try:
        # 1. Try Primary LLM (Anthropic)
        primary_llm = get_primary_llm()
        guardrail_chain = guardrail_prompt | primary_llm
        result = guardrail_chain.invoke({"input": user_text})
        return "OFF_TOPIC" not in result.content.upper()
        
    except Exception as primary_e:
        app_logger.warning(f"Guardrail primary LLM failed: {primary_e}. Trying fallback...")
        
        try:
            # 2. Try Fallback LLM (OpenAI) if Anthropic fails
            fallback_llm = get_fallback_llm()
            guardrail_chain = guardrail_prompt | fallback_llm
            result = guardrail_chain.invoke({"input": user_text})
            return "OFF_TOPIC" not in result.content.upper()
            
        except Exception as fallback_e:
            app_logger.error(f"Guardrail fallback also failed: {fallback_e}")
            # Fail open ONLY if both models are completely down
            return True 


from src.chains.base import build_lcel_agent_executor
from src.chains.callbacks.logging import StructuredLoggingCallbackHandler
from src.mcp.client import mcp_manager

async def build_agent_executor(llm):
    """
    Build a ToolCallingAgent with tools and few-shot prompt.
    Now uses LCEL AgentExecutor and dynamically includes MCP tools.
    """
    prompt = prompt_manager.get_chat_prompt("system_prompt")
    
    # Get standard tools and dynamic MCP tools
    mcp_tools = await mcp_manager.get_langchain_tools()
    all_combined_tools = ALL_TOOLS + mcp_tools
    
    callbacks = [StructuredLoggingCallbackHandler()]

    return build_lcel_agent_executor(
        llm=llm,
        tools=all_combined_tools,
        prompt=prompt,
        callbacks=callbacks
    )


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 8: RETRY LOGIC — Exponential backoff on transient failures
# ─────────────────────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),                        # Max 3 attempts
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 2s, 4s, 8s backoff
    retry=retry_if_exception_type((Exception,)),       # Retry on any exception
    reraise=True,
)
async def invoke_agent_with_retry(executor, input_data: dict) -> dict:
    """
    Invoke the agent with automatic retry on failure.
    """
    app_logger.info("Invoking agent (with retry logic)...")
    return await executor.ainvoke(input_data)


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 5: STRUCTURED OUTPUT PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_structured_output(raw_response: str) -> LoanDecision | None:
    """
    Try to extract a LoanDecision JSON from the agent's response text.
    """
    try:
        if "```json" in raw_response:
            json_str = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            json_str = raw_response.split("```")[1].split("```")[0].strip()
        elif "{" in raw_response and "}" in raw_response:
            start = raw_response.index("{")
            end = raw_response.rindex("}") + 1
            json_str = raw_response[start:end]
        else:
            return None

        parsed = json.loads(json_str)
        return LoanDecision(**parsed)
    except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
        app_logger.debug(f"Could not parse structured output: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT: get_underwriter_response
# Ties everything together: guardrail → agent → retry → fallback → parse
# ─────────────────────────────────────────────────────────────────────────────
async def get_underwriter_response(user_text: str, session_id: str = "default", role_name: str = "junior_analyst") -> dict:
    """
    Main function to process a user loan query.
    Orchestrates the pipeline and parses RAG citations (Week 3).
    """
    # --- Step 1: Guardrail Check ---
    if not check_guardrail(user_text):
        off_topic_msg = (
            "🚫 I am a Loan Underwriting Assistant. I can only help with credit risk, "
            "loan eligibility, document verification, policies, and banking queries.\n\n"
            "Please provide details like your income, debts, or ask a banking-related question."
        )
        log_interaction(session_id, user_text, off_topic_msg, [], guardrail_triggered=True)
        return {
            "response": off_topic_msg,
            "structured_output": None,
            "tools_used": [],
            "citations": []
        }

    # --- Step 2: Retrieve conversation memory (OBJECTIVE 2) ---
    memory = memory_manager.get_memory(session_id)
    chat_history = memory.buffer_as_messages if hasattr(memory, 'buffer_as_messages') else []

    input_data = {
        "input": user_text,
        "chat_history": chat_history,
    }

    # --- Step 3: Try primary agent with retry (OBJECTIVE 1 + 8) ---
    tools_used = []
    citations = []
    raw_response = ""

    try:
        primary_llm = get_primary_llm()
        primary_executor = await build_agent_executor(primary_llm)
        result = await invoke_agent_with_retry(primary_executor, input_data)
        raw_response = result.get("output", "")

        # Check for HITL triggers
        pending_task_id = hitl_manager.check_and_trigger(result)
        if pending_task_id:
            return {
                "session_id": session_id,
                "response": f"Your request has been paused for manual review (Task ID: {pending_task_id}). It requires approval before proceeding.",
                "structured_output": None,
                "tools_used": tools_used,
                "citations": []
            }

        # Format citations from vector search (Week 3)
        retrieved_docs = retrieve_documents(user_text, session_id=session_id, role_name=role_name)
        citations = []
        for step in result.get("intermediate_steps", []):
            action = step[0]
            tool_result = step[1]
            
            if hasattr(action, "tool"):
                tools_used.append(action.tool)
                
                # WEEK 3: Extract Citations if the RAG tool was used
                if action.tool == "knowledge_retrieval":
                    # Parse the formatted string returned by the tool
                    blocks = str(tool_result).split("[CITABLE SOURCE: ")
                    for block in blocks[1:]:
                        try:
                            header, snippet = block.split("]\n", 1)
                            parts = [p.strip() for p in header.split("|")]
                            
                            source_id = parts[0]
                            page = parts[1].split(":")[1].strip() if len(parts) > 1 else None
                            score_str = parts[2].split(":")[1].strip() if len(parts) > 2 else None
                            
                            citations.append({
                                "source_id": source_id,
                                "snippet": snippet.strip(),
                                "page_or_section": page,
                                "relevance_score": float(score_str) if score_str else None
                            })
                        except Exception as e:
                            app_logger.warning(f"Failed to parse citation block: {e}")

        app_logger.info(f"Primary agent succeeded. Tools used: {tools_used}")

    except Exception as primary_error:
        # --- Step 4: Fallback chain (OBJECTIVE 8) ---
        app_logger.warning(f"Primary agent failed: {primary_error}. Trying fallback...")

        try:
            fallback_llm = get_fallback_llm()
            fallback_executor = await build_agent_executor(fallback_llm)
            result = await fallback_executor.ainvoke(input_data)
            raw_response = result.get("output", "")

            for step in result.get("intermediate_steps", []):
                action = step[0]
                tool_result = step[1]
                
                if hasattr(action, "tool"):
                    tools_used.append(action.tool)
                    
                    if action.tool == "knowledge_retrieval":
                        blocks = str(tool_result).split("[CITABLE SOURCE: ")
                        for block in blocks[1:]:
                            try:
                                header, snippet = block.split("]\n", 1)
                                parts = [p.strip() for p in header.split("|")]
                                
                                source_id = parts[0]
                                page = parts[1].split(":")[1].strip() if len(parts) > 1 else None
                                score_str = parts[2].split(":")[1].strip() if len(parts) > 2 else None
                                
                                citations.append({
                                    "source_id": source_id,
                                    "snippet": snippet.strip(),
                                    "page_or_section": page,
                                    "relevance_score": float(score_str) if score_str else None
                                })
                            except Exception as e:
                                app_logger.warning(f"Failed to parse citation block (fallback): {e}")

            app_logger.info("Fallback agent succeeded.")

        except Exception as fallback_error:
            # --- Ultimate fallback: simple error message ---
            app_logger.error(f"Both agents failed. Primary: {primary_error}, Fallback: {fallback_error}")
            raw_response = (
                "⚠️ I'm experiencing technical difficulties processing your request. "
                "Please try again in a moment, or simplify your query.\n\n"
                f"Error details: {str(primary_error)[:200]}"
            )

    # --- Sanitize raw_response (Anthropic returns list of content blocks) ---
    if isinstance(raw_response, list):
        raw_response = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_response
        )
    elif not isinstance(raw_response, str):
        raw_response = str(raw_response)

    # --- Step 5: Parse structured output (OBJECTIVE 5) ---
    structured_output = parse_structured_output(raw_response)

    # --- Step 6: Save to conversation memory (OBJECTIVE 2) ---
    try:
        memory.save_context(
            {"input": user_text},
            {"output": raw_response},
        )
    except Exception as mem_error:
        app_logger.error(f"Failed to save memory: {mem_error}")

    # --- Step 7: Log the interaction (OBJECTIVE 9) ---
    log_interaction(
        session_id=session_id,
        user_input=user_text,
        agent_response=raw_response,
        tools_used=tools_used,
        structured_output=structured_output.model_dump() if structured_output else None,
    )

    return {
        "response": raw_response,
        "structured_output": structured_output,
        "tools_used": tools_used,
        "citations": citations
    }
