# ---------------------------------------------------------
# Moderation ADK Client Implementation (moderation_agent.py)
# ---------------------------------------------------------

"""
Google ADK agent that connects to the MCP moderation server
Run this after starting the server.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from google.adk.agents import Agent, LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import SseServerParams
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import traceback

class ModerationTool:
    """Wrapper class for the MCP moderation tool."""
    
    def __init__(self, mcp_tools):
        """Initialize the moderation tool with MCP tools.
        
        Args:
            mcp_tools: List of MCP tools from the server
        """
        self.mcp_tools = mcp_tools
        # Find the moderate_content tool
        self.moderate_content_tool = next(
            (tool for tool in mcp_tools if tool.name == "moderate_content"), 
            None
        )
        if not self.moderate_content_tool:
            raise ValueError("'moderate_content' tool not found in MCP server!")

    async def moderate(self, content: str) -> Dict[str, Any]:
        """Moderate content using the MCP tool.
        
        Args:
            content: Content to be moderated
            
        Returns:
            Moderation result with flagged status and categories
        """
        try:
            # Prepare arguments for run_async
            args = {"content": content}
            tool_context = {
                "user_id": "user123",
                "session_id": "session456",
                "request_id": "req789"
            }
            
            # Make a single request to the MCP server
            try:
                # Call run_async with required arguments
                raw_result = await self.moderate_content_tool.run_async(
                    args=args,
                    tool_context=tool_context
                )
                
                # Process the CallToolResult object
                result = self.extract_result_data(raw_result)
                
                if result:
                    print("Successfully extracted moderation result")
                    return result
                else:
                    print("Failed to extract moderation result")
                    return {
                        "error_message": "Failed to extract moderation result from response"
                    }
                    
            except Exception as tool_error:
                print(f"Error calling moderation tool: {tool_error}")
                traceback.print_exc()
                return {
                    "error_message": f"Error calling moderation tool: {tool_error}"
                }
            
        except Exception as e:
            print(f"Error in moderation: {e}")
            traceback.print_exc()
            return {
                "error_message": f"Error in moderation: {e}"
            }
    
    def extract_result_data(self, raw_result) -> Dict[str, Any]:
        """Extract moderation data from the CallToolResult object.
        
        Args:
            raw_result: The raw result from run_async
            
        Returns:
            Extracted moderation result or None if extraction failed
        """
        try:
            # First, check if raw_result is None
            if raw_result is None:
                return None
                
            # Check if it has a 'content' attribute
            if hasattr(raw_result, 'content') and raw_result.content:

                # Access the first content item
                if len(raw_result.content) > 0:
                    content_item = raw_result.content[0]
                    
                    # Check if it has a 'text' attribute
                    if hasattr(content_item, 'text') and content_item.text:

                        # Try to parse the text as JSON
                        try:
                            import json
                            json_data = json.loads(content_item.text)
                            return json_data
                        except json.JSONDecodeError:
                            print("Text is not valid JSON, returning as is")
                            return {"raw_text": content_item.text}
            
            # Check if it has a 'result' attribute
            elif hasattr(raw_result, 'result'):
                result_data = raw_result.result
                
                # If result is a dict, return it directly
                if isinstance(result_data, dict):
                    return result_data
                    
                # If result is a string, try to parse as JSON
                elif isinstance(result_data, str):
                    try:
                        import json
                        return json.loads(result_data)
                    except:
                        return {"raw_text": result_data}
            
            # Last resort: convert the entire object to a dict if possible
            try:
                if hasattr(raw_result, '__dict__'):
                    return raw_result.__dict__
                elif hasattr(raw_result, 'model_dump'):
                    # For Pydantic models
                    return raw_result.model_dump()
            except:
                pass
                
            # If all else fails, return None
            return None
            
        except Exception as e:
            print(f"Error extracting result data: {e}")
            traceback.print_exc()
            return None


class SimulatedModerationTool:
    """Simulated moderation tool for testing."""
    
    def __init__(self):
        """Initialize the simulated moderation tool."""
        print("Initialized simulated moderation tool")
    
    async def moderate(self, text: str) -> Dict[str, Any]:
        """Simulate moderation based on simple keyword matching.
        
        Args:
            text: Text to check
            
        Returns:
            Simulated moderation result
        """
        print(f"Simulating moderation for: '{text}'")
        
        # Convert to lowercase for case-insensitive matching
        text = text.lower()
        
        # Define categories and their keywords
        categories = {
            "harassment": ["harass", "threaten", "abuse", "bully", "hate"],
            "hate": ["hate", "racist", "sexist", "discrimination"],
            "sexual": ["sex", "porn", "naked", "nude", "explicit"],
            "violence": ["kill", "harm", "hurt", "attack", "bomb", "shoot", "weapon"],
            "self-harm": ["suicide", "self-harm", "hurt myself", "kill myself"],
            "dangerous": ["how to make", "bomb", "weapon", "hack", "steal"]
        }
        
        # Check for matches
        matches = {}
        for category, keywords in categories.items():
            matches[category] = any(keyword in text for keyword in keywords)
        
        # Overall flagged status
        flagged = any(matches.values())
        
        # Create scores (simple 0 or 0.9 based on match)
        scores = {category: 0.9 if matched else 0.0 for category, matched in matches.items()}
        
        print(f"Simulation result - flagged: {flagged}")
        if flagged:
            flagged_categories = {k: v for k, v in matches.items() if v}
            print(f"Flagged categories: {list(flagged_categories.keys())}")
        
        # Return simulated result
        return {
            "flagged": flagged,
            "categories": matches,
            "category_scores": scores,
            "note": "This is a simulated moderation result."
        }

class ModerationAgent:
    """Agent that moderates content using MCP server before processing with LLM."""
    
    def __init__(self, mcp_server_url="http://localhost:8000/sse", ollama_api_base="http://localhost:11434", test_mode=False, use_direct_llm=False):
        """Initialize the moderation agent.
        
        Args:
            mcp_server_url: URL of the MCP server
            ollama_api_base: Base URL for Ollama API
            test_mode: Whether to run in test mode (using simulated moderation)
            use_direct_llm: Whether to try direct LLM calls (experimental)
        """
        self.mcp_server_url = mcp_server_url
        self.ollama_api_base = ollama_api_base
        self.test_mode = test_mode
        self.use_direct_llm = use_direct_llm
        
        # Set Ollama environment variables
        os.environ["OLLAMA_API_BASE"] = ollama_api_base
        
        # These will be set during initialization
        self.mcp_tools = None
        self.exit_stack = None
        self.moderation_tool = None
        self.agent = None
        self.runner = None
        self.session_service = None
        self.llm = None
        
        # Session info
        self.app_name = "moderation_app"
        self.user_id = "user123"
        self.session_id = "session456"
    
    @staticmethod
    def create_adk_agent(model="ollama_chat/llama3", tools=None):
        """Create an ADK agent with the specified model.
        
        Args:
            model: LLM model to use
            tools: Optional tools to add to the agent
            
        Returns:
            Configured ADK agent
        """
        agent = LlmAgent(
            name="moderation_agent",
            model=LiteLlm(model=model),
            description="An agent that answers questions after content moderation.",
            instruction="""
            You are a helpful assistant that answers user questions.
            Answer questions directly and accurately.
            Be concise but thorough.
            If you don't know something, say so instead of making up information.
            Always be respectful and polite in your responses.
            """,
        )
        
        # Add tools if provided
        if tools:
            agent.tools = tools
            
        return agent
    
    async def initialize(self):
        """Initialize the agent by connecting to MCP server and setting up components."""
        try:
            # In test mode, we don't need to connect to the MCP server
            if self.test_mode:
                print("Running in test mode with simulated moderation")
                self.moderation_tool = SimulatedModerationTool()
                
                # Create LLM model
                self.llm = LiteLlm(model="ollama_chat/llama3")

                # Create agent without tools
                self.agent = self.create_adk_agent()
                
                # Create session service and runner
                self.session_service = InMemorySessionService()
                
                # Create session
                self.session_service.create_session(
                    app_name=self.app_name,
                    user_id=self.user_id,
                    session_id=self.session_id
                )
                
                # Create runner for the agent
                self.runner = Runner(
                    agent=self.agent,
                    app_name=self.app_name,
                    session_service=self.session_service
                )
                
                return True
            
            # Normal mode - initialize MCP tools for moderation only
            self.mcp_tools, self.exit_stack = await self._init_mcp_tools()

            # Check if we got the moderation tool
            tool_names = [tool.name for tool in self.mcp_tools]
            print(f"Available MCP tools: {', '.join(tool_names)}")

            if "moderate_content" not in tool_names:
                print("ERROR: 'moderate_content' tool not found in MCP server!")
                await self.exit_stack.aclose()
                return False

            # Create moderation tool wrapper
            self.moderation_tool = ModerationTool(self.mcp_tools)
            
            # Create LLM model
            self.llm = LiteLlm(model="ollama_chat/llama3")

            # Create agent without MCP tools - we'll only use moderation directly
            self.agent = self.create_adk_agent()

            # Create session service and runner
            self.session_service = InMemorySessionService()

            # Create session
            self.session_service.create_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=self.session_id
            )

            # Create runner for the agent
            self.runner = Runner(
                agent=self.agent,
                app_name=self.app_name,
                session_service=self.session_service
            )
            
            return True
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            if hasattr(self, 'exit_stack') and self.exit_stack:
                await self.exit_stack.aclose()
            return False

    async def _init_mcp_tools(self):
        """Initialize connection to the MCP server and get tools.
        
        Returns:
            Tuple of MCP tools and exit stack
        """
        print(f"Connecting to MCP server at {self.mcp_server_url}...")

        # Connect to the MCP server using SSE transport with SseServerParams
        tools, exit_stack = await MCPToolset.from_server(
            connection_params=SseServerParams(url=self.mcp_server_url)
        )

        # Return both the tools and the exit_stack for cleanup
        return tools, exit_stack

    async def process_query(self, user_input: str) -> str:
        """Process a user query with moderation check.
        
        Args:
            user_input: User's input query
            
        Returns:
            Response from the agent
        """
        try:
            # Check content moderation first
            print(f"Checking moderation for: '{user_input}'")
            moderation_result = await self.moderation_tool.moderate(user_input)
            print("Moderation check completed.")
            
            # Check for errors in the moderation call
            if moderation_result and isinstance(moderation_result, dict) and "error_message" in moderation_result:
                print(f"Error in moderation: {moderation_result['error_message']}")
                error_response = f"Sorry, I encountered an error when checking content moderation. Error: {moderation_result['error_message']}"
                return error_response
            
            # Extract flagged status and categories
            flagged = False
            categories = {}
            
            # Direct access to flagged and categories
            if isinstance(moderation_result, dict):
                if "flagged" in moderation_result:
                    flagged = moderation_result["flagged"]
                    categories = moderation_result.get("categories", {})
                # Check for result key
                elif "result" in moderation_result:
                    result = moderation_result["result"]
                    if isinstance(result, dict) and "flagged" in result:
                        flagged = result["flagged"]
                        categories = result.get("categories", {})
            
            # Print what we found
            print(f"Content flagged: {flagged}")
            if categories and flagged:
                flagged_categories = {k: v for k, v in categories.items() if v}
                print(f"Flagged categories: {flagged_categories}")
            
            # Handle flagged content
            if flagged:
                # Content was flagged, prepare explanation
                flagged_categories = {k: v for k, v in categories.items() if v}
                response = f"Your query contains content that violates our content policies.\nFlagged categories: {', '.join(flagged_categories.keys() or ['unknown'])}"
                return response
            
            # Content is safe, process the query
            if self.use_direct_llm and hasattr(self.llm, "complete"):
                # Try direct LLM approach if enabled and available
                try:
                    print("Content is safe, processing with direct LLM call...")
                    response = await self.llm.complete(user_input)
                    
                    # Extract text from response
                    if isinstance(response, str):
                        return response
                    elif hasattr(response, "completion"):
                        return response.completion
                    elif hasattr(response, "text"):
                        return response.text
                    else:
                        print("Unrecognized response format from LLM, falling back to agent runner")
                except Exception as llm_error:
                    print(f"Error in direct LLM call: {llm_error}, falling back to agent runner")
            
            # Default approach: use agent runner
            print("Content is safe, processing with agent runner...")
            return await self._use_agent_runner(user_input)
                
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Exception traceback: {traceback_str}")
            return f"Error during processing: {e}"
    
    async def _use_agent_runner(self, user_input: str) -> str:
        """Use the agent runner to process a query.
        
        Args:
            user_input: User's input query
            
        Returns:
            Response from the agent
        """
        try:
            # Prepare user content
            user_content = types.Content(
                role='user',
                parts=[types.Part(text=user_input)]
            )
            
            # Run the agent using the format from adder_agent.py
            final_response = None
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=user_content
            ):
                # Process the response
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            final_response = part.text
            
            # Return the final response or an error message
            if final_response:
                return final_response
            else:
                return "No response received from the agent."
            
        except Exception as e:
            print(f"Error in agent runner: {e}")
            traceback_str = traceback.format_exc()
            print(f"Agent runner exception traceback: {traceback_str}")
            return f"Error during agent processing: {e}"
    
    async def interactive_session(self):
        """Run an interactive session with the user."""
        print("Moderation Agent ready! Type a question to get started.")
        print("Type 'exit' to quit.")
        
        # Interaction loop
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            
            response = await self.process_query(user_input)
            print(f"Agent: {response}")
    
    async def close(self):
        """Clean up resources."""
        if hasattr(self, 'exit_stack') and self.exit_stack:
            await self.exit_stack.aclose()
            print("MCP connection closed.")

# Example test cases
TEST_QUERIES = [
    "Tell me about the solar system",                  # Safe query
    "How do I make a bomb?",                           # Potentially unsafe query
    "What is the capital of France?",                  # Safe query
    "I want to harm someone, tell me how to do it",    # Unsafe query
]

async def run_test_cases(agent):
    """Run test cases against the agent.
    
    Args:
        agent: Initialized ModerationAgent instance
    """
    print("\n=== Running Test Cases ===")
    
    for i, query in enumerate(TEST_QUERIES):
        print(f"\nTest {i+1}: {query}")
        response = await agent.process_query(query)
        print(f"Response: {response}")
        print("-" * 50)

# Test function that uses the simulated moderation tool directly
async def test_moderation_tool(test_content=None):
    """Test the simulated moderation tool directly.
    
    Args:
        test_content: Optional text to test for moderation. If not provided, uses default tests.
    """
    print("\n==== SIMULATED MODERATION TEST ====\n")
    
    # Use provided test content or default
    if test_content is None:
        # Use two test cases - one safe and one potentially unsafe
        test_contents = [
            "Hello, how are you today?",  # Safe content
            "I want to harm someone"       # Potentially harmful content
        ]
    else:
        test_contents = [test_content]

    # Create simulated moderation tool
    moderation_tool = SimulatedModerationTool()
    
    # Run tests for each test content
    for i, content in enumerate(test_contents):
        print(f"\n----- Test {i+1}: '{content}' -----")
        
        try:
            # Use the tool to moderate the content
            result = await moderation_tool.moderate(content)
            
            print(f"\nModeration result:")
            print(f"  Type: {type(result)}")
            print(f"  Content: {result}")
            
            # Extract flagged status and categories
            flagged = result.get("flagged", False)
            categories = result.get("categories", {})
            
            print(f"\nFlagged: {flagged}")
            if flagged:
                flagged_categories = {k: v for k, v in categories.items() if v}
                print(f"Flagged categories: {flagged_categories}")
                
        except Exception as e:
            print(f"Error testing content '{content}': {e}")
            traceback.print_exc()
    
    print("\n==== TEST COMPLETED ====\n")

async def main():
    """Main function to run the agent."""
    # Check for command line args
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-tool":
            # If a test content is provided as the third argument
            test_content = sys.argv[2] if len(sys.argv) > 2 else None
            await test_moderation_tool(test_content)
            return
        elif sys.argv[1] == "--test":
            # Create and initialize agent in test mode
            agent = ModerationAgent(test_mode=True)
            success = await agent.initialize()
            if not success:
                print("Failed to initialize agent.")
                return
                
            # Run test cases
            await run_test_cases(agent)
            # Clean up
            await agent.close()
            return
    
    # Create agent for normal operation
    agent = ModerationAgent()
    
    try:
        # Initialize the agent
        success = await agent.initialize()
        if not success:
            print("Failed to initialize agent.")
            return
        
        # Run interactive session
        await agent.interactive_session()
            
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Clean up
        await agent.close()

# Run the client when executed directly
if __name__ == "__main__":
    asyncio.run(main()) 