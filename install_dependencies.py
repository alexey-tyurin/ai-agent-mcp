# ---------------------------------------------------------
# 3. Setup Script (install_dependencies.py)
# ---------------------------------------------------------

"""
Script to install required dependencies
Run this first before running the server or client.
"""

import subprocess
import sys

def install_dependencies():
    print("Installing dependencies...")
    
    # Required packages
    packages = [
        "google-adk",    # Google Agent Development Kit
        "mcp",           # Model Context Protocol SDK
        "litellm>=1.41.27",  # LiteLLM for model integration (version with Ollama support)
    ]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_dependencies()