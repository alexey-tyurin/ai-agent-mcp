# ai-agent-mcp

# MCP Adder Service with Google ADK Client using Llama 3.2

This project demonstrates how to build an MCP server that exposes a simple
addition function and a Google ADK agent that calls this function using
Meta's Llama 3.2 model via Ollama and LiteLLM.

## Prerequisites

1. Install Ollama from https://ollama.com/
2. Pull the Llama 3.2 model:
   ```
   ollama pull llama3.2
   ```
3. Start the Ollama service (it should be running in the background)

## Setup

1. Install dependencies:
   ```
   python install_dependencies.py
   ```

2. Start the MCP server:
   ```
   python adder_server.py
   ```
   
   The server will run on http://localhost:8000/sse

3. In a new terminal, run the ADK client:
   ```
   python adder_agent.py
   ```

4. Interact with the agent by asking it to add numbers:
   ```
   You: Can you add 42 and 24?
   ```

5. Type 'exit' to quit the client.

## Project Structure

- `adder_server.py`: MCP server with addition function
- `adder_agent.py`: Google ADK agent with Llama 3.2 that connects to the MCP server
- `install_dependencies.py`: Script to install required dependencies

## Notes on LiteLLM and Ollama Integration

- The agent uses LiteLLM to connect to locally running Llama 3.2 model via Ollama
- It uses the 'ollama_chat' provider format for proper Ollama integration
- Environment variables are set up to point to the Ollama API endpoint

