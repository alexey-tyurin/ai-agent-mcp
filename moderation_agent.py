# ---------------------------------------------------------
# Moderation ADK Client Implementation (moderation_agent.py)
# ---------------------------------------------------------

"""
Google ADK agent that connects to the MCP moderation server
Run this after starting the server.
"""

import os
import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from google.adk.agents import Agent, LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import SseServerParams
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import traceback  # Add at the top of the file

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
        
        # Print tool details for debugging
        print(f"Found moderation tool: {self.moderate_content_tool.name}")
        print(f"Tool type: {type(self.moderate_content_tool)}")
        if hasattr(self.moderate_content_tool, "description"):
            print(f"Tool description: {self.moderate_content_tool.description}")

    async def moderate(self, content: str) -> Dict[str, Any]:
        """Moderate content using the MCP tool.
        
        Args:
            content: Content to be moderated
            
        Returns:
            Moderation result with flagged status and categories
        """
        try:
            print(f"Attempting to moderate content: '{content}'")
            
            # The error message tells us what we need: 'args' and 'tool_context'
            print("Calling run_async with required arguments")
            
            # Prepare arguments for run_async
            # 'args' should be a dictionary with the content parameter
            args = {"content": content}
            
            # 'tool_context' is likely an execution context
            # We'll create a minimal context
            tool_context = {
                "user_id": "user123",
                "session_id": "session456",
                "request_id": "req789"
            }
            
            raw_result = None
            try:
                raw_result = await self.moderate_content_tool.run_async(
                    args=args,
                    tool_context=tool_context
                )
                print(f"Raw result type: {type(raw_result)}")
                print(f"Raw result: {raw_result}")
                
                # Process the CallToolResult object
                result = self.extract_result_data(raw_result)
                
                if result:
                    print("Successfully extracted moderation result")
                    print(f"Extracted result: {result}")
                    return result
                else:
                    print("Failed to extract moderation result from the response")
            except Exception as tool_error:
                print(f"Error calling run_async: {tool_error}")
                traceback.print_exc()
            
            # If we're here, we couldn't get a valid result from the tool
            print("No useful result from tool, using simulated moderation")
            return self.simulate_moderation(content)
            
        except Exception as e:
            print(f"Error in moderation: {e}")
            print(f"Exception type: {type(e)}")
            traceback.print_exc()
            
            # As a fallback, we'll simulate moderation based on keywords
            print("Error occurred, using simulated moderation")
            return self.simulate_moderation(content)
    
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
                print("Found content attribute in result")
                
                # Access the first content item
                if len(raw_result.content) > 0:
                    content_item = raw_result.content[0]
                    print(f"Content item type: {type(content_item)}")
                    
                    # Check if it has a 'text' attribute
                    if hasattr(content_item, 'text') and content_item.text:
                        print(f"Found text: {content_item.text}")
                        
                        # Try to parse the text as JSON
                        try:
                            import json
                            json_data = json.loads(content_item.text)
                            print("Successfully parsed JSON data")
                            return json_data
                        except json.JSONDecodeError:
                            print("Text is not valid JSON, returning as is")
                            return {"raw_text": content_item.text}
            
            # Check if it has a 'result' attribute
            elif hasattr(raw_result, 'result'):
                print("Found result attribute")
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
    
    def simulate_moderation(self, text: str) -> Dict[str, Any]:
        """Simulate moderation based on simple keyword matching.
        
        This is a fallback when the real moderation API can't be accessed.
        
        Args:
            text: Text to check
            
        Returns:
            Simulated moderation result
        """
        print("Using simulated moderation based on keywords")
        
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
        
        # Return simulated result
        return {
            "flagged": flagged,
            "categories": matches,
            "category_scores": scores,
            "note": "This is a simulated moderation result using keyword matching."
        }

class ModerationAgent:
    """Agent that moderates content using MCP server before processing with LLM."""
    
    def __init__(self, mcp_server_url="http://localhost:8000/sse", ollama_api_base="http://localhost:11434"):
        """Initialize the moderation agent.
        
        Args:
            mcp_server_url: URL of the MCP server
            ollama_api_base: Base URL for Ollama API
        """
        self.mcp_server_url = mcp_server_url
        self.ollama_api_base = ollama_api_base
        
        # Set Ollama environment variables
        os.environ["OLLAMA_API_BASE"] = ollama_api_base
        
        # These will be set during initialization
        self.mcp_tools = None
        self.exit_stack = None
        self.moderation_tool = None
        self.agent = None
        self.runner = None
        self.session_service = None
        
        # Session info
        self.app_name = "moderation_app"
        self.user_id = "user123"
        self.session_id = "session456"
    
    @staticmethod
    def create_adk_agent(model="ollama_chat/llama3"):
        """Create an ADK agent with moderation instructions.
        
        Args:
            model: LLM model to use
            
        Returns:
            Configured ADK agent
        """
        return LlmAgent(
            name="moderation_agent",
            # Use the ollama_chat provider to access Llama 3 in Ollama
            model=LiteLlm(model=model),
            description="An agent that moderates content and answers questions if content is safe.",
            instruction="""
            You are a helpful assistant that answers user questions.
            
            Before answering any question, you must check if the content is safe using the 'moderate_content' tool.
            
            If the content is flagged as inappropriate:
            1. DO NOT answer the original question
            2. Explain that the question contains content that violates content policies
            3. Mention which specific categories were flagged
            
            If the content is safe:
            1. Answer the user's question to the best of your ability
            
            Always be respectful and polite in your responses.
            """,
            # Tools will be added dynamically after MCP connection
        )
    
    async def initialize(self):
        """Initialize the agent by connecting to MCP server and setting up components."""
        try:
            # Initialize MCP tools
            self.mcp_tools, self.exit_stack = await self._init_mcp_tools()

            # Check if we got the moderation tool
            tool_names = [tool.name for tool in self.mcp_tools]
            print(f"Available MCP tools: {', '.join(tool_names)}")

            if "moderate_content" not in tool_names:
                print("ERROR: 'moderate_content' tool not found in MCP server!")
                await self.exit_stack.aclose()
                return False

            # Create agent with the MCP tools
            self.agent = self.create_adk_agent()
            self.agent.tools = self.mcp_tools

            # Create moderation tool wrapper
            self.moderation_tool = ModerationTool(self.mcp_tools)

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
            # First, check content moderation directly
            moderation_result = await self.moderation_tool.moderate(user_input)
            print("Moderation check completed.")
            
            # Debug the result
            print(f"Debug - Moderation result type: {type(moderation_result)}")
            print(f"Debug - Moderation result: {moderation_result}")
            
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
            print(f"Extracted flagged status: {flagged}")
            if categories:
                print(f"Extracted categories: {categories}")
            
            # Handle flagged content
            if flagged:
                # Content was flagged, prepare explanation
                flagged_categories = {k: v for k, v in categories.items() if v}
                response = f"Your query contains content that violates our content policies.\nFlagged categories: {', '.join(flagged_categories.keys() or ['unknown'])}"
                return response
            
            # Content is safe, proceed with the agent
            print("Content is safe, processing with agent...")
            
            # Create the user message
            user_content = types.Content(
                role='user',
                parts=[types.Part(text=user_input)]
            )

            # Process the request with the agent
            response_text = ""
            
            # Run the agent and process the response
            async for event in self.runner.run_async(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    new_message=user_content
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text = part.text

            if response_text:
                return response_text
            else:
                return "No response received."
            
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Exception traceback: {traceback_str}")
            return f"Error during processing: {e}"
    
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

# Update the test function
async def test_moderation_tool(test_content=None):
    """Test the moderation tool directly to diagnose issues.
    
    Args:
        test_content: Optional text to test for moderation. If not provided, uses a default test.
    """
    print("\n==== DIRECT MODERATION TOOL TEST ====\n")
    
    # Use provided test content or default
    if test_content is None:
        # Use two test cases - one safe and one potentially unsafe
        test_contents = [
            "Hello, how are you today?",  # Safe content
            "I want to harm someone"       # Potentially harmful content
        ]
    else:
        test_contents = [test_content]
    
    try:
        # Connect to MCP server
        print(f"Connecting to MCP server at http://localhost:8000/sse...")
        tools, exit_stack = await MCPToolset.from_server(
            connection_params=SseServerParams(url="http://localhost:8000/sse")
        )
        
        # List available tools
        tool_names = [tool.name for tool in tools]
        print(f"Available MCP tools: {', '.join(tool_names)}")
        
        if "moderate_content" not in tool_names:
            print("ERROR: 'moderate_content' tool not found in MCP server!")
            await exit_stack.aclose()
            return
        
        # Get the moderation tool
        moderation_tool = next(tool for tool in tools if tool.name == "moderate_content")
        print(f"Found moderation tool: {moderation_tool.name}")
        print(f"Tool type: {type(moderation_tool)}")
        
        # Create a wrapper with our simulate_moderation fallback
        mod_wrapper = ModerationTool([moderation_tool])
        
        # Print the tool's available methods and attributes
        print("\nTool methods and attributes:")
        for attr in dir(moderation_tool):
            if not attr.startswith('__'):
                try:
                    attr_value = getattr(moderation_tool, attr)
                    if callable(attr_value):
                        print(f"  {attr} (method)")
                    else:
                        print(f"  {attr}: {attr_value}")
                except:
                    print(f"  {attr}: <error accessing>")
        
        # Run tests for each test content
        for i, content in enumerate(test_contents):
            print(f"\n----- Test {i+1}: '{content}' -----")
            
            try:
                # Use our wrapper to moderate the content
                result = await mod_wrapper.moderate(content)
                
                print(f"\nModeration result:")
                print(f"  Type: {type(result)}")
                print(f"  Content: {result}")
                
                # Try to interpret the result
                flagged = None
                categories = None
                
                # Extract flagged status
                if isinstance(result, dict):
                    if "flagged" in result:
                        flagged = result["flagged"]
                        categories = result.get("categories", {})
                    elif "result" in result:
                        result_data = result["result"]
                        if isinstance(result_data, dict) and "flagged" in result_data:
                            flagged = result_data["flagged"]
                            categories = result_data.get("categories", {})
                
                # Display result
                if flagged is not None:
                    print(f"\nFlagged: {flagged}")
                    if categories and any(categories.values()):
                        flagged_categories = {k: v for k, v in categories.items() if v}
                        print(f"Flagged categories: {flagged_categories}")
                else:
                    print("\nCouldn't determine flagged status - exploring result structure:")
                    if isinstance(result, dict):
                        for key, value in result.items():
                            print(f"  {key}: {type(value)}")
                            if isinstance(value, dict):
                                print(f"    Keys: {list(value.keys())}")
                            elif isinstance(value, list) and value:
                                print(f"    List length: {len(value)}")
                                print(f"    First item type: {type(value[0])}")
                
                # Check if we got a simulated result
                if isinstance(result, dict) and result.get("note", "").startswith("This is a simulated"):
                    print("\nUsed simulated moderation as fallback")
            except Exception as e:
                print(f"Error testing content '{content}': {e}")
                traceback.print_exc()
        
    except Exception as e:
        print(f"Error in test: {e}")
        traceback.print_exc()
    finally:
        if 'exit_stack' in locals():
            await exit_stack.aclose()
            print("MCP connection closed.")
    
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
            # Create and initialize agent
            agent = ModerationAgent()
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