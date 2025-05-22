# AI News Researcher: Multi-Agent System with SmolAgents and OpenRouter

This project implements a sophisticated multi-agent system for AI-powered research, analysis, and validation using SmolAgents and OpenRouter API.

## System Architecture

The system consists of four specialized agents:
1. **Research Agent**: Gathers information from the web
2. **Analysis Agent**: Processes and analyzes the data
3. **Validation Agent**: Fact-checks and verifies information
4. **Coordinator Agent**: Orchestrates the workflow between agents

## Setup Instructions

### Prerequisites

- Python 3.11+
- OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai))
- UV package manager (for dependency management)

### Installation

1. Clone this repository
2. Install UV if you don't have it:
   ```bash
   curl -sSf https://astral.sh/uv/install.sh | bash
   ```
3. Install required packages with UV:
   ```bash
   uv pip install smolagents markdownify duckduckgo-search
   ```
4. Set up your OpenRouter API key:
   ```bash
   export OPENROUTER_API_KEY="your-openrouter-api-key"
   ```

## Usage

Run the main application:
```bash
python smolagents_multiagent_app.py
```

This will run demonstrations of:
- Custom tools
- Individual agent capabilities
- Coordinated multi-agent projects

## Rules for Using This Multi-Agent System with OpenRouter

1. **API Configuration**:
   - Always set the OPENROUTER_API_KEY environment variable before running
   - The system uses OpenRouter's API endpoint: https://openrouter.ai/api/v1

2. **Model Selection**:
   - Default model is set to "anthropic/claude-3-5-sonnet"
   - You can specify other OpenRouter-compatible models when initializing agents
   - Multiple LLM models can be used simultaneously for different agents
   - Mix and match models based on their strengths (e.g., Claude for reasoning, GPT-4 for creativity)
   - See [OpenRouter's model list](https://openrouter.ai/docs/models) for available options

3. **Agent Initialization**:
   - All agents use the `create_openrouter_model()` function to create model instances
   - Model instantiation follows OpenRouter's OpenAI-compatible interface

4. **Rate Limiting and Tokens**:
   - Be aware of OpenRouter's rate limits and token quotas
   - Consider implementing retry logic for rate limit errors
   - Monitor token usage to manage costs

5. **Environment Variables**:
   - Keep API keys in environment variables, never hardcode them
   - Check for API key presence before running to avoid runtime errors

6. **Error Handling**:
   - Handle API connection errors gracefully
   - Implement appropriate timeout settings for API calls

7. **Model Compatibility**:
   - Ensure your chosen model supports tool calling/function calling
   - Not all models available on OpenRouter support tool calling

8. **Customization**:
   - Custom agents should follow the same pattern for OpenRouter integration
   - When adding new tools, ensure they're compatible with your chosen model

## Advanced Usage

For more advanced usage patterns, see the `AdvancedMultiAgentOrchestrator` class in the source code, which demonstrates:
- Task queuing
- Result caching
- Collaborative research
- Error handling and retry logic

## Resources

- [SmolAgents Documentation](https://github.com/huggingface/smolagents)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [UV Package Manager](https://github.com/astral-sh/uv) 
