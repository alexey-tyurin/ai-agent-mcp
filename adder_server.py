# ---------------------------------------------------------
# 1. MCP Server Implementation (adder_server.py)
# ---------------------------------------------------------

"""
FastMCP server that exposes a function to add two numbers
Run this script first before running the client.
"""

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("Adder Service")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together.
    
    Args:
        a: First number to add
        b: Second number to add
    
    Returns:
        The sum of a and b
    """
    return a + b

# Run the server when executed directly
if __name__ == "__main__":
    # Start the server with SSE transport
    print("Starting MCP Server with add function...")
    mcp.run(transport="sse")
    # By default, this will run on http://localhost:8000/sse