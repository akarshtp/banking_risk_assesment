import logging
from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from src.core.logger_config import app_logger

class StructuredLoggingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for structured logging of LCEL chains."""
    
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        name = serialized.get("name", "UnnamedChain")
        app_logger.info(f"[CHAIN START] {name}")

    def on_chain_end(
        self, outputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        app_logger.info(f"[CHAIN END] Successfully completed")

    def on_chain_error(
        self, error: BaseException, **kwargs: Any
    ) -> None:
        app_logger.error(f"[CHAIN ERROR] {str(error)}")
        
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        name = serialized.get("name", "UnnamedTool")
        app_logger.info(f"[TOOL START] {name} | inputs: {input_str}")

    def on_tool_end(
        self, output: str, **kwargs: Any
    ) -> None:
        # Avoid logging massive text blobs
        trunc_output = output[:200] + "..." if len(output) > 200 else output
        app_logger.info(f"[TOOL END] output: {trunc_output}")

    def on_tool_error(
        self, error: BaseException, **kwargs: Any
    ) -> None:
        app_logger.error(f"[TOOL ERROR] {str(error)}")
