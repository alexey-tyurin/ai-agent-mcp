# ---------------------------------------------------------
# 2. ADK Client Implementation (adder_agent.py)
# ---------------------------------------------------------

"""
Google ADK agent that connects to the MCP server
Run this after starting the server.
"""

import os
import asyncio
from google.adk.agents import Agent, LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import SseServerParams

# MCP server URL (the server must be running)
MCP_SERVER_URL = "http://localhost:8000/sse"

# Set Ollama environment variables
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
# Alternatively, you can use OpenAI format:
# os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
# os.environ["OPENAI_API_KEY"] = "anything"

async def init_mcp_tools():
    """Initialize connection to the MCP server and get tools."""
    print(f"Connecting to MCP server at {MCP_SERVER_URL}...")

    # Connect to the MCP server using SSE transport with SseServerParams
    tools, exit_stack = await MCPToolset.from_server(
        connection_params=SseServerParams(url=MCP_SERVER_URL)
    )

    # Return both the tools and the exit_stack for cleanup
    return tools, exit_stack

# Define the agent
def create_agent():
    """Create an ADK agent that uses MCP tools with Llama 3.2 via Ollama."""
    # Use LiteLlm wrapper with Ollama's Llama 3.2 model
    agent = LlmAgent(
        name="calculator_agent",
        # Use the ollama_chat provider to access Llama 3.2 in Ollama
        model=LiteLlm(model="ollama_chat/llama3.2"),
        description="An agent that can add two numbers using an MCP server.",
        instruction="""
        You are a helpful assistant that can add numbers.
        When a user asks you to add two numbers, use the 'add' tool 
        to calculate the result.
        
        Always explain the calculation to the user after you get the result.
        """,
        # Tools will be added dynamically after MCP connection
    )
    return agent

async def main():
    """Main function to set up and run the agent."""
    try:
        # Initialize MCP tools
        mcp_tools, exit_stack = await init_mcp_tools()

        # Check if we got the add tool
        tool_names = [tool.name for tool in mcp_tools]
        print(f"Available MCP tools: {', '.join(tool_names)}")

        if "add" not in tool_names:
            print("ERROR: 'add' tool not found in MCP server!")
            await exit_stack.aclose()
            return

        # Create agent with the MCP tools
        agent = create_agent()
        agent.tools = mcp_tools

        print("Agent ready! Try asking it to add two numbers.")
        print("Type 'exit' to quit.")

        # Simple interaction loop
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            # Process the request with the agent
            response = await agent.process_async(user_input)
            print(f"Agent: {response.text}")

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Clean up MCP connection
        if 'exit_stack' in locals():
            await exit_stack.aclose()
            print("MCP connection closed.")

# Run the client when executed directly
if __name__ == "__main__":
    asyncio.run(main())
