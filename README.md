# AI Content Moderation System

## The Problem: When AI Conversations Go Wrong

Imagine this: You've just launched your exciting new AI chatbot. Users are flooding in, conversations are happening, and everything seems perfect—until it's not.

A user asks your AI how to build an explosive device. Another probes for inappropriate content. Someone else tests the boundaries with harmful language. Suddenly, your innovative AI tool has become a potential liability.

**This is the reality many AI developers face today.**

Without proper content moderation, AI systems can:
- Generate harmful, dangerous, or illegal content
- Expose organizations to legal and reputational risks
- Create unsafe experiences for users
- Turn promising applications into PR nightmares

## The Solution: Integrated Content Moderation

This is where our AI Content Moderation System comes in. We've built a production-ready solution that:

1. Seamlessly checks every user input through OpenAI's moderation API
2. Filters out harmful content before it reaches your AI
3. Provides detailed feedback on why content was blocked
4. Integrates smoothly with the Google ADK framework

The best part? OpenAI's moderation API is **free** to use, making robust content safety accessible to projects of all sizes.

## Real-World Applications

This system isn't just a technical demo—it's designed for real production use cases:

- **Educational Platforms**: Ensure student interactions with AI tutors remain appropriate and safe
- **Customer Service Bots**: Prevent abuse while maintaining helpful service
- **Content Generation Tools**: Filter requests for inappropriate creative content
- **Internal Enterprise Tools**: Maintain professional standards in workplace AI usage
- **Healthcare Chatbots**: Prevent requests for harmful medical advice
- **Social Applications**: Screen user-generated prompts in AI-powered social features

One of our beta testers, a developer of an educational AI platform, shared: *"Before implementing this system, we had to manually review flagged interactions. Now, inappropriate content is caught automatically, saving us countless hours and significantly reducing our risk."*

A production-ready content moderation system that leverages OpenAI's moderation API through Google's Agent Development Kit (ADK) and Machine Conversation Protocol (MCP).

![Project Banner](https://i.imgur.com/QObdWb5.png)

## Business Value

- **Free Content Moderation**: Utilize OpenAI's moderation API at no additional cost to filter harmful content.
- **Protect Users & Platforms**: Automatically identify and block dangerous, harmful, or inappropriate content.
- **Regulatory Compliance**: Help meet legal requirements for content monitoring on digital platforms.
- **Scale with Confidence**: Handle moderation at scale without manual review bottlenecks.

## Technical Value

- **Cross-Vendor Implementation**: Practical demonstration of Google ADK agents communicating with OpenAI services.
- **Dual Transport Options**: Support for both Server-Sent Events (SSE) and Standard I/O (STDIO) transports.
- **Modular Architecture**: Easily extensible to support additional moderation providers or capabilities.
- **Production-Ready Design**: Error handling, logging, and separation of concerns for real-world use.

## Architecture

This project implements two variants of a content moderation flow:

### SSE Transport Architecture

```
┌────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  User Query │──→│ ADK Agent    │──→│ MCP Client    │──→│ MCP Server  │
│             │    │ (SSE Client) │    │ (Moderation) │    │ (SSE)       │
└────────────┘    └──────────────┘    └──────────────┘    └──────┬──────┘
                          ↑                                      │
                          │                                      ↓
                          │                              ┌─────────────┐
                          │                              │ OpenAI      │
                          └──────────────────────────────┤ Moderation  │
                                Response                  │ API         │
                                                         └─────────────┘
```

### STDIO Transport Architecture

```
┌────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  User Query │──→│ ADK Agent    │──→│ MCP Client    │──→│ MCP Server  │
│             │    │ (STDIO)      │    │ (Moderation) │    │ (STDIO)     │
└────────────┘    └──────────────┘    └──────────────┘    └──────┬──────┘
                          ↑                                      │
                          │                                      ↓
                          │                              ┌─────────────┐
                          │                              │ OpenAI      │
                          └──────────────────────────────┤ Moderation  │
                                Response                  │ API         │
                                                         └─────────────┘
```

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- [Ollama](https://ollama.ai/) (for local LLM serving, optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-content-moderation.git
cd ai-content-moderation
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"  # On Windows, use: set OPENAI_API_KEY=your-api-key-here
```

### Running the System

#### SSE Transport Version

1. Start the moderation server:
```bash
python moderation_server_sse.py
```

2. In a new terminal, start the moderation agent:
```bash
python moderation_agent_sse.py
```

#### STDIO Transport Version

1. Run the moderation agent (it will start the server automatically):
```bash
python moderation_agent_stdio.py
```

### Testing

Run the comprehensive test script to check all components:
```bash
python test_moderation_system.py
```

Or test individual components:

```bash
# Test server in standalone mode
python moderation_server_sse.py --test

# Test agent with simulated moderation (no server needed)
python moderation_agent_sse.py --test

# Test the moderation tool directly
python moderation_agent_sse.py --test-tool "Test content to moderate"
```

## Making it Production Ready

1. **Authentication**: Add proper authentication for the MCP server.
2. **Containerization**: Package applications as Docker containers for consistent deployment.
3. **Monitoring**: Implement health checks, metrics collection, and alerting.
4. **Logging**: Enhance logging for better observability and troubleshooting.
5. **Rate Limiting**: Implement rate limiting to prevent API abuse.
6. **Caching**: Add caching for repetitive moderation requests.
7. **Failover Mechanisms**: Implement retry logic and fallback options.
8. **Security Hardening**: Review and address security concerns.

## Deployment Options

### Cloud Deployment

- **AWS**: Deploy as Lambda functions or ECS containers
- **Google Cloud**: Use Cloud Run or GKE
- **Azure**: Deploy as Azure Functions or AKS

### Self-Hosted

- **Docker Compose**: For simple multi-container deployments
- **Kubernetes**: For large-scale, resilient deployments
- **Serverless Frameworks**: For event-driven architectures

## Comparison of Transport Methods

| Feature | SSE | STDIO |
|---------|-----|-------|
| **Setup Complexity** | Requires separate server process | Server embedded in client process |
| **Network Requirements** | Needs HTTP connection | Works within a single host |
| **Scalability** | Better for distributed systems | Better for single-host deployment |
| **Debugging** | Easier to monitor network traffic | Harder to inspect communication |
| **Resource Usage** | Higher overhead | Lower overhead |
| **Security** | Needs network security considerations | More contained security boundary |

## Follow-up Ideas

1. **Support Multiple Moderation Providers**: Add support for alternative moderation APIs.
2. **Custom Moderation Rules**: Allow configuration of custom moderation policies.
3. **Content Categorization**: Expand beyond binary moderation to content classification.
4. **Explanation Generation**: Provide human-readable explanations for moderation decisions.
5. **Web UI**: Add a web interface for testing and monitoring moderation.
6. **Feedback Loop**: Implement a system to learn from false positives/negatives.
7. **Batch Processing**: Add support for moderating batches of content efficiently.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [OpenAI](https://openai.com/) for providing the moderation API
- [Google ADK](https://github.com/google-ai/google-adk) for the agent framework
- [MCP](https://github.com/google-ai/mcp) for the Machine Conversation Protocol implementation 