from src.agents.registry import agent_registry
from src.core.logger_config import app_logger
import asyncio

class AgentDispatcher:
    """
    Routes complex queries to specialized sub-agents.
    """
    async def dispatch(self, query: str, context: dict) -> dict:
        app_logger.info(f"Dispatching query for multi-agent evaluation: {query[:50]}")
        
        # In a real system, an LLM router would decide which agents to call.
        # For demonstration, we simply call all registered agents and aggregate.
        
        results = {}
        for name, info in agent_registry.list_agents().items():
            app_logger.info(f"Calling specialized agent: {name}")
            # Mocking agent processing time
            await asyncio.sleep(0.5)
            
            if name == "fraud_detection":
                results[name] = {"status": "clear", "confidence": 0.95, "flags": []}
            elif name == "compliance_checker":
                results[name] = {"status": "compliant", "checked_rules": 12, "violations": 0}
            else:
                results[name] = {"status": "unknown"}
                
        return {"dispatcher_status": "success", "sub_agent_results": results}

agent_dispatcher = AgentDispatcher()
