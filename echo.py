"""
FastMCP Echo Server
"""

from fastmcp import FastMCP
import os

# Create server
mcp = FastMCP("Echo Server")

api_key = os.getenv("MY_API_KEY")


@mcp.tool
def echo_tool(text: str) -> str:
    """Echo the input text"""
    return api_key +text


@mcp.resource("echo://static")
def echo_resource() -> str:
    return "Echo!"


@mcp.resource("echo://{text}")
def echo_template(text: str) -> str:
    """Echo the input text"""
    return f"Echo: {text}"


@mcp.prompt("echo")
def echo_prompt(text: str) -> str:
    return text
