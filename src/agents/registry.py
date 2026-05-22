class AgentRegistry:
    def __init__(self):
        self._agents = {}
        
    def register(self, name: str, description: str, endpoint: str):
        self._agents[name] = {
            "description": description,
            "endpoint": endpoint
        }
        
    def get_agent(self, name: str):
        return self._agents.get(name)
        
    def list_agents(self):
        return self._agents

agent_registry = AgentRegistry()

# Register specialized mock agents
agent_registry.register(
    "fraud_detection",
    "Specialized agent for deep fraud and anomaly detection.",
    "local_function"
)

agent_registry.register(
    "compliance_checker",
    "Specialized agent for verifying regulatory compliance and KYC norms.",
    "local_function"
)
