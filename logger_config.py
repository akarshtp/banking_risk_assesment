import os
import logging
import json
from datetime import datetime, timezone
from pythonjsonlogger import jsonlogger


# ─────────────────────────────────────────────────────────────────────────────
# LOG DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM JSON FORMATTER
# Adds timestamp, level, and module info to every log entry.
# ─────────────────────────────────────────────────────────────────────────────
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that adds standard fields to every log record."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        log_record["level"] = record.levelname
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION LOGGER — General system events
# ─────────────────────────────────────────────────────────────────────────────
app_logger = logging.getLogger("loan_underwriter")
app_logger.setLevel(logging.DEBUG)

# Console handler (human-readable for development)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("%(asctime)s | %(levelname)-8s | %(module)s | %(message)s")
console_handler.setFormatter(console_format)
app_logger.addHandler(console_handler)

# File handler (structured JSON for production/audit)
app_file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))
app_file_handler.setLevel(logging.DEBUG)
app_file_handler.setFormatter(CustomJsonFormatter())
app_logger.addHandler(app_file_handler)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTION LOGGER — Dedicated log for all chat interactions
# OBJECTIVE 9: Every user ↔ assistant exchange is logged with full context
# ─────────────────────────────────────────────────────────────────────────────
interaction_logger = logging.getLogger("loan_underwriter.interactions")
interaction_logger.setLevel(logging.INFO)

interaction_file_handler = logging.FileHandler(os.path.join(LOG_DIR, "interactions.log"))
interaction_file_handler.setLevel(logging.INFO)
interaction_file_handler.setFormatter(CustomJsonFormatter())
interaction_logger.addHandler(interaction_file_handler)


def log_interaction(
    session_id: str,
    user_input: str,
    agent_response: str,
    tools_used: list,
    structured_output: dict = None,
    guardrail_triggered: bool = False,
):
    """
    Log a complete interaction in structured JSON format.
    
    OBJECTIVE 9: This is called after every /chat request.
    
    Args:
        session_id: The conversation session identifier.
        user_input: What the user said.
        agent_response: What the agent replied.
        tools_used: List of tool names invoked during this turn.
        structured_output: Parsed LoanDecision dict (if available).
        guardrail_triggered: Whether the guardrail blocked this query.
    """
    interaction_data = {
        "event": "chat_interaction",
        "session_id": session_id,
        "user_input": user_input,
        "agent_response": agent_response[:500],  # Truncate for log readability
        "tools_used": tools_used,
        "tools_count": len(tools_used),
        "structured_output": structured_output,
        "guardrail_triggered": guardrail_triggered,
        "response_length": len(agent_response),
    }

    interaction_logger.info(json.dumps(interaction_data))
