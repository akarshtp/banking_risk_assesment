import asyncio
import json
from typing import Dict, List, Any
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model, Field
import mcp.client.stdio
import mcp.client.session
from mcp.types import CallToolRequest, ClientCapabilities
from src.mcp.registry import load_mcp_servers
from src.core.logger_config import app_logger

class MCPManager:
    """Manages connections to MCP servers and wraps their tools for Langchain."""
    
    def __init__(self):
        self.servers = load_mcp_servers()
        self.sessions = {}
        self.exit_stacks = {}
        self.tools_cache = []
        self._connected = False
        
    async def connect_all(self):
        """Connects to all configured MCP servers."""
        if self._connected:
            return
            
        import contextlib
        
        for server_name, config in self.servers.items():
            try:
                command = config.get("command")
                args = config.get("args", [])
                env = config.get("env", {})
                
                # Combine current env with server-specific env
                server_env = {**os.environ, **env}
                
                app_logger.info(f"Connecting to MCP server: {server_name}")
                
                server_parameters = mcp.client.stdio.StdioServerParameters(
                    command=command,
                    args=args,
                    env=server_env
                )
                
                exit_stack = contextlib.AsyncExitStack()
                self.exit_stacks[server_name] = exit_stack
                
                stdio_transport = await exit_stack.enter_async_context(
                    mcp.client.stdio.stdio_client(server_parameters)
                )
                read, write = stdio_transport
                
                session = await exit_stack.enter_async_context(
                    mcp.client.session.ClientSession(read, write)
                )
                
                await session.initialize()
                self.sessions[server_name] = session
                
            except Exception as e:
                app_logger.error(f"Failed to connect to MCP server {server_name}: {e}")
                
        self._connected = True
                
    async def get_langchain_tools(self) -> List[StructuredTool]:
        """Discover tools from all MCP servers and wrap them for LangChain."""
        if not self._connected:
            await self.connect_all()
            
        if self.tools_cache:
            return self.tools_cache
            
        all_tools = []
        for server_name, session in self.sessions.items():
            try:
                response = await session.list_tools()
                for tool in response.tools:
                    # Dynamically create Pydantic model for Langchain args schema
                    fields = {}
                    properties = tool.inputSchema.get("properties", {})
                    required = tool.inputSchema.get("required", [])
                    
                    for prop_name, prop_info in properties.items():
                        prop_type = Any
                        if prop_info.get("type") == "string":
                            prop_type = str
                        elif prop_info.get("type") == "number":
                            prop_type = float
                        elif prop_info.get("type") == "integer":
                            prop_type = int
                        elif prop_info.get("type") == "boolean":
                            prop_type = bool
                            
                        if prop_name in required:
                            fields[prop_name] = (prop_type, Field(..., description=prop_info.get("description", "")))
                        else:
                            fields[prop_name] = (prop_type, Field(default=None, description=prop_info.get("description", "")))
                            
                    args_schema = create_model(f"{tool.name}Schema", **fields)
                    
                    # Create the LangChain tool
                    def create_tool_func(s_name, t_name):
                        def tool_func(**kwargs) -> str:
                            # We must run this via asyncio since tool_func is sync in standard langchain
                            # Wait, structured tool can have async coroutine!
                            pass
                            
                        async def atool_func(**kwargs) -> str:
                            target_session = self.sessions[s_name]
                            result = await target_session.call_tool(t_name, arguments=kwargs)
                            # Convert MCP result to string
                            return "\n".join([c.text for c in result.content if c.type == "text"])
                            
                        return atool_func
                        
                    lc_tool = StructuredTool.from_function(
                        coroutine=create_tool_func(server_name, tool.name),
                        name=tool.name,
                        description=tool.description,
                        args_schema=args_schema,
                    )
                    all_tools.append(lc_tool)
                    
            except Exception as e:
                app_logger.error(f"Failed to discover tools from {server_name}: {e}")
                
        self.tools_cache = all_tools
        return all_tools
        
    async def invoke_tool(self, server_name: str, tool_name: str, args: dict) -> str:
        """Raw invocation of an MCP tool by name."""
        if not self._connected:
            await self.connect_all()
            
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not found or not connected.")
            
        session = self.sessions[server_name]
        result = await session.call_tool(tool_name, arguments=args)
        return "\n".join([c.text for c in result.content if c.type == "text"])
        
    async def cleanup(self):
        """Cleanup all connections."""
        for name, stack in self.exit_stacks.items():
            await stack.aclose()
        self.sessions.clear()
        self.exit_stacks.clear()
        self._connected = False

import os
# Global MCP manager
mcp_manager = MCPManager()
