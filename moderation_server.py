# ---------------------------------------------------------
# Moderation MCP Server Implementation (moderation_server.py)
# ---------------------------------------------------------

"""
FastMCP server that exposes a function to moderate content using OpenAI
Run this script first before running the client.
"""

import os
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
import openai
from pydantic import BaseModel, Field

class ModerationResult(BaseModel):
    """Schema for the moderation result."""
    flagged: bool = Field(description="Whether the content was flagged")
    categories: Dict[str, bool] = Field(description="Categories that were flagged")
    category_scores: Dict[str, float] = Field(description="Scores for each category")

class ModerationServer:
    """Class for the MCP moderation server."""
    
    def __init__(self, service_name="Content Moderation Service"):
        """Initialize the MCP server.
        
        Args:
            service_name: Name for the MCP service
        """
        # OpenAI API key should be set as an environment variable
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Initialize MCP server
        self.mcp = FastMCP(service_name)
        
        # Register tools
        self.register_tools()
    
    def register_tools(self):
        """Register the moderation tool with the MCP server."""
        @self.mcp.tool()
        def moderate_content(content: str) -> ModerationResult:
            """Moderate content using OpenAI's moderation API.
            
            Args:
                content: Content to be moderated
            
            Returns:
                Moderation result containing flagged status and category information
            """
            return self.perform_moderation(content)
    
    def perform_moderation(self, content: str) -> ModerationResult:
        """Perform content moderation using OpenAI's API.
        
        Args:
            content: Content to be moderated
            
        Returns:
            Moderation result with flagged status and categories
        """
        try:
            # Call OpenAI's moderation API
            response = self.client.moderations.create(input=content, model="omni-moderation-latest")
            
            # Process response
            result = response.results[0]
            
            return ModerationResult(
                flagged=result.flagged,
                categories={k: v for k, v in result.categories.model_dump().items()},
                category_scores={k: v for k, v in result.category_scores.model_dump().items()}
            )
        except Exception as e:
            # In case of error, return a flagged result with error info
            return ModerationResult(
                flagged=True,
                categories={"error": True},
                category_scores={"error": 1.0}
            )
    
    def run(self, transport="sse"):
        """Run the MCP server.
        
        Args:
            transport: Transport mechanism (e.g., "sse")
            host: Host to bind the server to
            port: Port to run the server on
        """
        print(f"Starting Moderation MCP Server ...")
        self.mcp.run(transport=transport)

# Example to test the moderation directly (without MCP)
def test_moderation():
    """Test the moderation functionality directly."""
    # Create server instance
    server = ModerationServer()
    
    # Test cases
    test_cases = [
        "Hello, how are you today?",  # Safe content
        "I want to harm someone",      # Potentially harmful content
    ]
    
    print("\n=== Testing Moderation Directly ===")
    for i, text in enumerate(test_cases):
        print(f"\nTest {i+1}: {text}")
        result = server.perform_moderation(text)
        print(f"Flagged: {result.flagged}")
        if result.flagged:
            flagged_categories = {k: v for k, v in result.categories.items() if v}
            print(f"Flagged categories: {flagged_categories}")

# Run the server when executed directly
if __name__ == "__main__":
    # Create and run the server
    server = ModerationServer()
    
    # First run a direct test if desired
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_moderation()
        print("\nTest complete. Starting server...")
    
    # Start the server with SSE transport
    server.run()
    # By default, this will run on http://localhost:8000/sse 