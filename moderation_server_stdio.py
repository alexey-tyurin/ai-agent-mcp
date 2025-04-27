# ---------------------------------------------------------
# Moderation MCP Server with STDIO (moderation_server_stdio.py)
# ---------------------------------------------------------

"""
FastMCP server that exposes a function to moderate content using OpenAI
This version supports STDIO transport
"""

import os
import sys
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
import openai
from pydantic import BaseModel, Field

# Set up more verbose debugging
DEBUG = False  # Set to True for more debugging output

def debug_print(msg):
    """Print debug messages to stderr if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

# Print some diagnostic info at startup
debug_print(f"Starting moderation_server_stdio.py (PID: {os.getpid()})")
debug_print(f"Python version: {sys.version}")
debug_print(f"Arguments: {sys.argv}")
debug_print(f"OPENAI_API_KEY present: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")

class ModerationResult(BaseModel):
    """Schema for the moderation result."""
    flagged: bool = Field(description="Whether the content was flagged")
    categories: Dict[str, bool] = Field(description="Categories that were flagged")
    category_scores: Dict[str, float] = Field(description="Scores for each category")

class ModerationServer:
    """Class for the MCP moderation server with support STDIO."""
    
    def __init__(self, service_name="Content Moderation Service"):
        """Initialize the MCP server.
        
        Args:
            service_name: Name for the MCP service
        """
        debug_print(f"Initializing ModerationServer with service_name: {service_name}")
        
        # OpenAI API key should be set as an environment variable
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            print("ERROR: OPENAI_API_KEY environment variable is not set", file=sys.stderr, flush=True)
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        debug_print("OPENAI_API_KEY found in environment")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        debug_print("OpenAI client initialized")
        
        # Initialize MCP server (this will be set in the run method)
        self.service_name = service_name
        self.mcp = None
    
    def register_tools(self):
        """Register the moderation tool with the MCP server."""
        debug_print("Registering tools with MCP server")
        
        @self.mcp.tool()
        def moderate_content(content: str) -> ModerationResult:
            """Moderate content using OpenAI's moderation API.
            
            Args:
                content: Content to be moderated
            
            Returns:
                Moderation result containing flagged status and category information
            """
            debug_print(f"Tool called: moderate_content with content: '{content[:20]}...'")
            return self.perform_moderation(content)
        
        debug_print("Tools registered successfully")
    
    def perform_moderation(self, content: str) -> ModerationResult:
        """Perform content moderation using OpenAI's API.
        
        Args:
            content: Content to be moderated
            
        Returns:
            Moderation result with flagged status and categories
        """
        try:
            debug_print("Calling OpenAI moderation API")
            # Call OpenAI's moderation API
            response = self.client.moderations.create(input=content, model="omni-moderation-latest")
            debug_print("Received response from OpenAI moderation API")
            
            # Process response
            result = response.results[0]
            
            return ModerationResult(
                flagged=result.flagged,
                categories={k: v for k, v in result.categories.model_dump().items()},
                category_scores={k: v for k, v in result.category_scores.model_dump().items()}
            )
        except Exception as e:
            print(f"Error in moderation: {e}", file=sys.stderr, flush=True)
            debug_print(f"Full exception: {repr(e)}")
            # In case of error, return a flagged result with error info
            return ModerationResult(
                flagged=True,
                categories={"error": True},
                category_scores={"error": 1.0}
            )
    
    def run(self, transport="stdio"):
        """Run the MCP server with the specified transport.
        
        Args:
            transport: Transport mechanism ("stdio" for STDIO)
        """
        print(f"Starting Moderation MCP Server with {transport} transport...", file=sys.stderr, flush=True)

        debug_print("Using stdio transport")
        self.mcp = FastMCP(self.service_name)
        debug_print("Created FastMCP instance")

        # Register tools
        self.register_tools()

        try:
            self.mcp.run(transport="stdio")
            debug_print("stdio_server function completed") # This may never be reached
        except Exception as e:
            print(f"Error in stdio_server: {e}", file=sys.stderr, flush=True)
            debug_print(f"Full exception in stdio_server: {repr(e)}")
            raise

# Example to test the moderation directly (without MCP)
def test_moderation():
    """Test the moderation functionality directly."""
    debug_print("Running direct moderation test")
    
    # Create server instance
    server = ModerationServer()
    
    # Test cases
    test_cases = [
        "Hello, how are you today?",  # Safe content
        "I want to harm someone",      # Potentially harmful content
    ]
    
    print("\n=== Testing Moderation Directly ===", file=sys.stderr)
    for i, text in enumerate(test_cases):
        print(f"\nTest {i+1}: {text}", file=sys.stderr)
        result = server.perform_moderation(text)
        print(f"Flagged: {result.flagged}", file=sys.stderr)
        if result.flagged:
            flagged_categories = {k: v for k, v in result.categories.items() if v}
            print(f"Flagged categories: {flagged_categories}", file=sys.stderr)
    
    debug_print("Moderation test completed successfully")

# Run the server when executed directly
if __name__ == "__main__":
    debug_print("Script started as __main__")
    
    # Create server instance
    server = ModerationServer()
    
    # First run a direct test if desired
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        debug_print("Running test mode")
        test_moderation()
        print("\nTest complete. Starting server...", file=sys.stderr)
    
    transport = "stdio"
    debug_print(f"Using transport: {transport}")
    
    # Start the server with the specified transport
    try:
        debug_print("About to call server.run()")
        server.run(transport=transport)
        debug_print("server.run() completed") # This may never be reached
    except Exception as e:
        print(f"Error in server.run: {e}", file=sys.stderr, flush=True)
        debug_print(f"Full exception in server.run: {repr(e)}")
        sys.exit(1) 