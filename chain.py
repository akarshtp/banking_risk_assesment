import os
import json
import re
from types import SimpleNamespace
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from tools import ALL_TOOLS, TOOL_NAMES
from prompts import build_example_selector, build_prompt
from memory_manager import memory_manager
from schemas import LoanDecision, Citation
from logger_config import app_logger, log_interaction

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI  # Add this line

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 1: LLM INITIALIZATION
# Primary model: claude-sonnet for balance of speed + quality
# Fallback model: claude-haiku for resilience (OBJECTIVE 8)
# ─────────────────────────────────────────────────────────────────────────────
# Model name — single source of truth
MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")


def get_primary_llm():
    """Initialize the primary Claude model."""
    return ChatAnthropic(
        model=MODEL_NAME,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2048,
        temperature=0.1,  # Low temperature for consistent underwriting decisions
    )


def get_fallback_llm():
    """
    OBJECTIVE 8: Fallback model — uses OpenAI for resilience 
    if the primary Anthropic model fails.
    """
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), # You can change this to gpt-4o or gpt-3.5-turbo
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
def check_guardrail(user_text: str) -> bool:
    """
    Quick check to ensure the query is banking/loan related.
    
    Returns:
        True if the query is safe (on-topic), False if off-topic.
    """
    guardrail_llm = ChatAnthropic(
        model=MODEL_NAME,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=10,
        temperature=0.0,
    )
    guardrail_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a banking safety filter. If the user query is NOT about banking, "
         "loans, credit, finance, income, policies, or debt, reply ONLY 'OFF_TOPIC'. "
         "Otherwise reply ONLY 'SAFE'. "
         "General greetings (hi, hello, hey, good morning, thanks, etc.) and "
         "conversational pleasantries are SAFE — always allow them through."),
        ("human", "{input}"),
    ])

    # OBJECTIVE 1: Clean chain composition using | pipe operator
    guardrail_chain = guardrail_prompt | guardrail_llm

    try:
        result = guardrail_chain.invoke({"input": user_text})
        return "OFF_TOPIC" not in result.content.upper()
    except Exception as e:
        app_logger.error(f"Guardrail check failed: {e}")
        return True  # Fail open — let the main agent handle it


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE 1: Custom Tool-Calling Agent
# Uses ChatAnthropic.bind_tools() for native Claude tool calling.
# ─────────────────────────────────────────────────────────────────────────────
class ToolCallingAgent:
    """
    Custom tool-calling agent that uses Claude's native tool-calling support.

    OBJECTIVE 1: Clean chain composition
    OBJECTIVE 3: Tools are bound to the agent via bind_tools()
    OBJECTIVE 4: Few-shot prompt with semantic selection
    """

    def __init__(self, llm, tools, prompt, max_iterations=5, verbose=True):
        self.llm = llm.bind_tools(tools)  # Bind tools to the LLM
        self.tools = {t.name: t for t in tools}
        self.prompt = prompt
        self.max_iterations = max_iterations
        self.verbose = verbose

    @staticmethod
    def _extract_text(content) -> str:
        """
        Extract plain text from Claude's response content.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            return "\n".join(text_parts)
        return str(content)

    def invoke(self, input_data: dict) -> dict:
        """
        Run the agent loop: prompt -> LLM -> tool calls -> repeat until done.

        Returns:
            Dict with 'output' (str) and 'intermediate_steps' (list of tuples).
        """
        messages = self.prompt.format_messages(
            input=input_data["input"],
            chat_history=input_data.get("chat_history", []),
            agent_scratchpad=[],
        )

        intermediate_steps = []

        for iteration in range(self.max_iterations):
            response = self.llm.invoke(messages)
            messages.append(response)

            if self.verbose:
                app_logger.info(
                    f"Agent iteration {iteration + 1}: "
                    f"tool_calls={len(response.tool_calls) if response.tool_calls else 0}"
                )

            if not response.tool_calls:
                return {
                    "output": self._extract_text(response.content),
                    "intermediate_steps": intermediate_steps,
                }

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if self.verbose:
                    app_logger.info(f"  Calling tool: {tool_name} with args: {tool_args}")

                if tool_name in self.tools:
                    try:
                        tool_result = self.tools[tool_name].invoke(tool_args)
                    except Exception as e:
                        tool_result = f"Tool error: {str(e)}"
                        app_logger.error(f"  Tool {tool_name} failed: {e}")
                else:
                    tool_result = f"Tool '{tool_name}' not found."

                action = SimpleNamespace(tool=tool_name, tool_input=tool_args)
                intermediate_steps.append((action, tool_result))

                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                ))

        final_content = self._extract_text(
            messages[-1].content if messages else "Max iterations reached."
        )
        return {
            "output": final_content,
            "intermediate_steps": intermediate_steps,
        }


def build_agent_executor(llm):
    """
    Build a ToolCallingAgent with tools and few-shot prompt.
    """
    prompt = build_prompt(example_selector=example_selector)

    return ToolCallingAgent(
        llm=llm,
        tools=ALL_TOOLS,
        prompt=prompt,
        max_iterations=5,
        verbose=True,
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
def invoke_agent_with_retry(executor: ToolCallingAgent, input_data: dict) -> dict:
    """
    Invoke the agent with automatic retry on failure.
    """
    app_logger.info("Invoking agent (with retry logic)...")
    return executor.invoke(input_data)


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
def get_underwriter_response(user_text: str, session_id: str = "default") -> dict:
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
        primary_executor = build_agent_executor(primary_llm)
        result = invoke_agent_with_retry(primary_executor, input_data)
        raw_response = result.get("output", "")

        # Extract tool names and citations from intermediate steps
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
            fallback_executor = build_agent_executor(fallback_llm)
            result = fallback_executor.invoke(input_data)
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