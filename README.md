# AI Content Moderation System with Google AI agent and Model Context Protocol (MCP)

A content moderation system that leverages OpenAI's moderation API through Google's Agent Development Kit (ADK) and Model Context Protocol (MCP).

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

## Business Value

- **Free Content Moderation**: Utilize OpenAI's moderation API at no additional cost to filter harmful content.
- **Protect Users & Platforms**: Automatically identify and block dangerous, harmful, or inappropriate content.
- **Regulatory Compliance**: Help meet legal requirements for content monitoring on digital platforms.
- **Scale with Confidence**: Handle moderation at scale without manual review bottlenecks.

## Technical Value

- **Cross-Vendor Implementation**: Practical demonstration of Google ADK agents communicating with OpenAI services via MCP and use Llama in the same flow.
- **Dual Transport Options**: Support for both Server-Sent Events (SSE) and Standard I/O (STDIO) transports.
- **Modular Architecture**: Easily extensible to support additional moderation providers or capabilities.
- **Production-Ready Design**: Error handling, logging, and separation of concerns for real-world use.

## Architecture

This project implements two variants of a content moderation flow:

### SSE Transport Architecture

```
┌────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│ User Query │───→│ ADK Agent    │───→│ MCP Client   │───→│ MCP Server  │
│            │←───│ (SSE Client) │←───│ (Moderation) │←───│ (SSE)       │
└────────────┘    └──────────────┘    └──────────────┘    └────────┬────┘
                          ↑                                    ↑   │ 
                          │              Moderation Response   │   ↓
                          │                              ┌─────────────┐
                          │                              │ OpenAI      │
                          │                              │             │
                          │                              │ API         │
                          │                              └─────────────┘
                          │
                          ↓
               ┌────────────────────┐
               │ Llama LLM          │
               │ (Content Response) │
               └────────────────────┘
```

### STDIO Transport Architecture

```
┌────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│ User Query │───→│ ADK Agent    │───→│ MCP Client   │───→│ MCP Server  │
│            │←───│ (STDIO)      │←───│ (Moderation) │←───│ (STDIO)     │
└────────────┘    └──────────────┘    └──────────────┘    └────────┬────┘
                          ↑                                   ↑    │ 
                          │              Moderation Response  │    ↓
                          │                              ┌─────────────┐
                          │                              │ OpenAI      │
                          │                              │             │
                          │                              │ API         │
                          │                              └─────────────┘
                          │
                          ↓
               ┌────────────────────┐
               │ Llama LLM          │
               │ (Content Response) │
               └────────────────────┘
```

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- [Ollama](https://ollama.ai/) (for local LLM serving, optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/alexey-tyurin/ai-agent-mcp.git
cd ai-agent-mcp
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

## Deployment Options for Model Context Protocol (MCP) Server

### Cloud Deployment Options

#### 1. Containerized Services
Containerization using Docker and Kubernetes is a popular approach for deploying MCP servers in cloud environments. This provides:

* Scalability: Easily scale up/down based on demand
* Portability: Deploy across different cloud providers
* Isolation: Each MCP server runs in its own container
* Orchestration: Kubernetes handles service discovery, load balancing, and failover

#### 2. Serverless Deployments
Serverless options like AWS Lambda, Google Cloud Functions, or Azure Functions can be used for MCP servers that don't require persistent connections. Benefits include:

* Pay-per-use pricing model
* Automatic scaling
* No infrastructure management
* Ideal for event-driven MCP implementations

#### 3. Cloud-Managed Hosting
Major cloud providers offer managed services that can host MCP servers:

* AWS App Runner or Elastic Beanstalk
* Google Cloud Run
* Azure App Service
* Digital Ocean App Platform

These services handle infrastructure scaling while you focus on the application logic.

### Self-Hosted Deployment Options

#### 1. On-Premises Servers
Traditional on-premises deployment involves installing the MCP server directly on physical servers in your data center. This offers:

* Complete control over hardware
* Data residency and compliance benefits
* No dependency on external cloud services
* Can be optimized for specific workloads

#### 2. Private Cloud
Deploy MCP servers on private cloud infrastructure using technologies like OpenStack or VMware vSphere. This provides:

* Virtualization benefits while keeping data in-house
* Resource pooling and automated provisioning
* Enhanced security and privacy controls
* Better isolation than public cloud for sensitive implementations

#### 3. Edge Deployment
For latency-sensitive applications, MCP servers can be deployed at the edge, closer to where data is generated and consumed. Benefits include:

* Reduced latency for real-time applications
* Bandwidth optimization
* Works well for distributed architectures
* Can operate in environments with limited connectivity

### Hybrid Approaches
Many organizations opt for hybrid deployment strategies:

* Primary MCP servers in the cloud with edge deployments for latency-sensitive operations
* Critical components self-hosted with auxiliary services in the cloud
* Development and testing in cloud, production in self-hosted environments

### Implementation Considerations
When choosing a deployment option, consider these factors:

* Scalability Requirements: How many concurrent connections will your MCP server need to handle?
* Data Privacy Concerns: Are there regulatory requirements for data handling?
* Latency Requirements: Is real-time communication critical for your application?
* Budget Constraints: Cloud services offer flexibility but costs can scale with usage
* Team Expertise: Self-hosted options require more infrastructure expertise

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

## Screenshots

### SSE Transport Version

<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent-mcp/blob/main/screenshots/sse_test.png?raw=true" width="auto" height="auto"></img> 
</p>

### STDIO Transport Version
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent-mcp/blob/main/screenshots/stdio_test.png?raw=true" width="auto" height="auto"></img> 
</p>

### Testing

<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent-mcp/blob/main/screenshots/tests1.png?raw=true" width="auto" height="auto"></img> 
</p>
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent-mcp/blob/main/screenshots/tests2.png?raw=true" width="auto" height="auto"></img> 
</p>
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent-mcp/blob/main/screenshots/tests3.png?raw=true" width="auto" height="auto"></img> 
</p>
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent-mcp/blob/main/screenshots/tests4.png?raw=true" width="auto" height="auto"></img> 
</p>
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent-mcp/blob/main/screenshots/tests5.png?raw=true" width="auto" height="auto"></img> 
</p>
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent-mcp/blob/main/screenshots/tests6.png?raw=true" width="auto" height="auto"></img> 
</p>
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent-mcp/blob/main/screenshots/tests7.png?raw=true" width="auto" height="auto"></img> 
</p>
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent-mcp/blob/main/screenshots/tests8.png?raw=true" width="auto" height="auto"></img> 
</p>

## Acknowledgements

- [OpenAI](https://openai.com/) for providing the moderation API
- [Google ADK](https://github.com/google-ai/google-adk) for the agent framework
- [MCP](https://docs.anthropic.com/en/docs/agents-and-tools/mcp) for the Model Context Protocol

## Contact Information

For any questions or feedback, please contact Alexey Tyurin at altyurin3@gmail.com.

## License

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project is licensed under the MIT License - see the [LICENSE](https://github.com/alexey-tyurin/ai-agent-mcp/blob/main/LICENSE) file for details.