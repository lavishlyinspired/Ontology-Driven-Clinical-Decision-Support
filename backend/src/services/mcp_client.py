"""
MCP Client for Internal Tool Invocation
=========================================

This module provides a client for invoking MCP tools directly within the LCA application,
enabling the chat interface to use all MCP-defined tools without stdio communication.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import asyncio

from ..mcp_server.lca_mcp_server import LCAMCPServer
from ..logging_config import get_logger

logger = get_logger(__name__)


class MCPToolInvoker:
    """
    Internal MCP tool invoker for chat integration

    This class allows the conversation service to invoke MCP tools directly
    without requiring stdio communication. It wraps the LCAMCPServer and
    provides a simple async interface for tool invocation.
    """

    def __init__(self):
        """Initialize the MCP tool invoker"""
        self.mcp_server = LCAMCPServer()
        self._initialized = False
        logger.info("MCP Tool Invoker initialized")

    async def initialize(self):
        """Initialize MCP server components lazily"""
        if not self._initialized:
            # Initialize the MCP server components
            await self.mcp_server._initialize_components()
            self._initialized = True
            logger.info("MCP server components initialized")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available MCP tools

        Returns:
            List of tool definitions with name, description, and input schema
        """
        try:
            tools = self.mcp_server._get_all_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                for tool in tools
            ]
        except Exception as e:
            logger.error(f"Error listing MCP tools: {e}")
            return []

    async def invoke_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invoke an MCP tool by name with arguments

        Args:
            tool_name: Name of the tool to invoke
            arguments: Tool arguments as a dictionary

        Returns:
            Tool execution result as a dictionary
        """
        try:
            # Ensure components are initialized
            await self.initialize()

            logger.info(f"Invoking MCP tool: {tool_name} with args: {arguments}")

            # Get the tool handler
            handler = self.mcp_server._get_tool_handler(tool_name)

            if handler is None:
                return {
                    "status": "error",
                    "error": f"Tool '{tool_name}' not found"
                }

            # Execute the handler
            result = await handler(arguments)

            # Parse result if it's a TextContent list (MCP format)
            if isinstance(result, list) and len(result) > 0:
                if hasattr(result[0], 'text'):
                    result_text = result[0].text
                    try:
                        parsed_result = json.loads(result_text)
                        return {
                            "status": "success",
                            "tool": tool_name,
                            "result": parsed_result
                        }
                    except json.JSONDecodeError:
                        return {
                            "status": "success",
                            "tool": tool_name,
                            "result": result_text
                        }

            # Return raw result if not in MCP format
            return {
                "status": "success",
                "tool": tool_name,
                "result": result
            }

        except Exception as e:
            logger.error(f"Error invoking MCP tool '{tool_name}': {e}", exc_info=True)
            return {
                "status": "error",
                "tool": tool_name,
                "error": str(e),
                "error_type": type(e).__name__
            }

    async def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for tools matching a query

        Args:
            query: Search query (tool name or description keywords)

        Returns:
            List of matching tools
        """
        try:
            all_tools = await self.list_tools()
            query_lower = query.lower()

            matching_tools = [
                tool for tool in all_tools
                if query_lower in tool["name"].lower()
                or query_lower in tool["description"].lower()
            ]

            return matching_tools
        except Exception as e:
            logger.error(f"Error searching MCP tools: {e}")
            return []

    async def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific tool

        Args:
            tool_name: Name of the tool

        Returns:
            Tool information or None if not found
        """
        try:
            all_tools = await self.list_tools()
            for tool in all_tools:
                if tool["name"] == tool_name:
                    return tool
            return None
        except Exception as e:
            logger.error(f"Error getting tool info for '{tool_name}': {e}")
            return None


# Global singleton instance
_mcp_invoker: Optional[MCPToolInvoker] = None


def get_mcp_invoker() -> MCPToolInvoker:
    """
    Get or create the global MCP tool invoker instance

    Returns:
        MCPToolInvoker singleton instance
    """
    global _mcp_invoker
    if _mcp_invoker is None:
        _mcp_invoker = MCPToolInvoker()
    return _mcp_invoker
