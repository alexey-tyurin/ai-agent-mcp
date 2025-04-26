# Content Moderation Agent with MCP and LiteLLM

This project demonstrates a content moderation workflow using Google's Agent Development Kit (ADK) and the Model Context Protocol (MCP). It consists of:

1. A moderation MCP server that proxies requests to OpenAI's moderation API
2. An ADK agent that uses LiteLLM with Ollama to process user queries after moderation

## Prerequisites

- Python 3.9 or higher
- Ollama installed and running locally with Llama 3 model
- OpenAI API key

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install Ollama from https://ollama.com/ if not already installed
4. Pull the Llama 3 model in Ollama:
   ```
   ollama pull llama3
   ```

## Setup

1. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

1. Start the Ollama server:
   ```
   ollama serve
   ```

2. Start the MCP moderation server in a new terminal:
   ```
   python moderation_server.py
   ```

3. Start the moderation agent in another terminal:
   ```
   python moderation_agent.py
   ```

## Usage

Once both the server and agent are running, you can interact with the agent by typing questions. The agent will:

1. Send each query to the MCP server for moderation using OpenAI's moderation API
2. If the content is flagged as inappropriate, explain which categories were flagged
3. If the content is safe, process it with Llama 3 via Ollama and return the response

To exit the application, type `exit`.

## Implementation Details

- `moderation_server.py`: MCP server that connects to OpenAI's moderation API
- `moderation_agent.py`: Google ADK agent that uses MCP tools and LiteLLM with Ollama

## Flow Diagram

```
User Query → Moderation Agent → MCP Server → OpenAI Moderation API
                      ↓
               Content Safe?
                ↙     ↘
          YES         NO
           ↓           ↓
        Process      Return
        with LLM     Warning
```

## License

See the LICENSE file for details. 