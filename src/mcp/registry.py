import yaml
import os

def load_mcp_servers(config_path="config/mcp_servers.yaml"):
    """Load MCP server definitions from YAML registry."""
    if not os.path.exists(config_path):
        return {}
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config.get("mcpServers", {})
