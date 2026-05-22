# MCP Integration

The Model Context Protocol (MCP) allows our AI agents to dynamically discover and invoke external tool servers.

## Configured Servers
Defined in `config/mcp_servers.yaml`:
1. **Credit Bureau**: Mock server that fetches live credit scores for applicants.
2. **Income Verification**: Mock server that verifies stated income against tax records.

## Usage
The LangChain agent automatically loads these tools via the `mcp_manager` at runtime, providing zero-configuration tool usage.
