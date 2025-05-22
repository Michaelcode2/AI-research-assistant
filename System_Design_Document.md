# Technical Requirements Document
## Multi-Agent Research Assistant with Smolagents

### Project Overview
This document provides comprehensive technical requirements for developing a multi-agent research assistant using the Smolagents framework. It's designed to assist Cursor IDE with intelligent code completion, understanding project structure, and providing relevant suggestions.

---

## 1. Project Architecture

### 1.1 Core Components
```
Multi-Agent System
├── Custom Tools (@tool decorators)
├── Specialized Agents (Research, Analysis, Validation)
├── Coordination Layer (Orchestrator)
└── Advanced Patterns (Task Queue, Error Handling)
```

### 1.2 Technology Stack
- **Framework**: Smolagents (Hugging Face)
- **Language**: Python 3.11+
- **Package Manager**: uv
- **AI Models**: OpenAI GPT-4, Hugging Face models
- **Dependencies**: openai, requests, python-dotenv

---

## 2. Development Environment Setup

### 2.1 Cursor IDE Configuration

#### 2.1.1 Required Extensions
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.flake8",
    "ms-python.mypy-type-checker",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "ms-python.debugpy"
  ]
}
```

#### 2.1.2 Workspace Settings (.vscode/settings.json)
```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.autoImportCompletions": true,
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"],
  "files.associations": {
    "*.py": "python",
    ".env*": "dotenv"
  },
  "python.analysis.extraPaths": [
    "./src",
    "./.venv/lib/python3.11/site-packages"
  ]
}
```

### 2.2 Type Hints and Annotations
All functions must include comprehensive type hints for better IDE support:

```python
from typing import List, Dict, Any, Optional, Union, Callable
from smolagents import ToolCallingAgent, tool
from smolagents.tools import DuckDuckGoSearchTool, PythonInterpreterTool

# Example type annotations
@tool
def save_research_note(content: str, filename: str = "research_notes.txt") -> str:
    """Type hints help Cursor understand function signatures"""
    pass

class ResearchAgent:
    def __init__(self, model_name: str = "openai:gpt-4") -> None:
        self.agent: ToolCallingAgent = ToolCallingAgent(...)
    
    def research_topic(self, topic: str) -> str:
        """Returns research results as string"""
        pass
```

---

## 3. Code Structure and Patterns

### 3.1 File Organization
```
smolagents-multiagent/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── research_agent.py
│   │   ├── analysis_agent.py
│   │   ├── validation_agent.py
│   │   └── coordinator_agent.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── custom_tools.py
│   │   └── validation_tools.py
│   ├── orchestrators/
│   │   ├── __init__.py
│   │   └── advanced_orchestrator.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logging_utils.py
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_tools.py
│   └── test_orchestrators.py
├── examples/
│   ├── basic_demo.py
│   ├── advanced_demo.py
│   └── custom_workflow.py
├── docs/
│   ├── api_reference.md
│   ├── user_guide.md
│   └── development_guide.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── main.py
```

### 3.2 Design Patterns

#### 3.2.1 Agent Factory Pattern
```python
from abc import ABC, abstractmethod
from typing import Protocol

class AgentProtocol(Protocol):
    """Protocol for agent interface - helps Cursor with type checking"""
    def execute_task(self, task: str) -> str: ...

class AgentFactory:
    """Factory for creating different types of agents"""
    
    @staticmethod
    def create_agent(agent_type: str, **kwargs) -> AgentProtocol:
        if agent_type == "research":
            return ResearchAgent(**kwargs)
        elif agent_type == "analysis":
            return AnalysisAgent(**kwargs)
        # ... other agent types
        raise ValueError(f"Unknown agent type: {agent_type}")
```

#### 3.2.2 Tool Registry Pattern
```python
from typing import Dict, Callable, Any

class ToolRegistry:
    """Registry for managing custom tools"""
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
    
    def register(self, name: str, tool_func: Callable) -> None:
        """Register a tool with the registry"""
        self._tools[name] = tool_func
    
    def get_tool(self, name: str) -> Callable:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools"""
        return list(self._tools.keys())
```

---

## 4. API Interfaces and Contracts

### 4.1 Agent Interface
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseAgent(ABC):
    """Base class for all agents - provides interface contract"""
    
    def __init__(self, model_name: str, system_prompt: str):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.agent: ToolCallingAgent = None
    
    @abstractmethod
    def execute_task(self, task_data: Dict[str, Any]) -> str:
        """Execute a specific task - must be implemented by subclasses"""
        pass
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return []
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data before processing"""
        return True
```

### 4.2 Tool Interface
```python
from typing import Any, Dict
from smolagents import tool

class ToolMetadata:
    """Metadata for tools to help with documentation"""
    
    def __init__(self, name: str, description: str, 
                 parameters: Dict[str, Any], returns: str):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.returns = returns

# Custom tool with metadata
@tool
def enhanced_save_research_note(
    content: str, 
    filename: str = "research_notes.txt",
    append: bool = True
) -> Dict[str, Any]:
    """
    Enhanced research note saving with metadata
    
    Args:
        content: The research content to save
        filename: Name of the file to save to
        append: Whether to append to existing file
    
    Returns:
        Dict with success status and metadata
    """
    try:
        mode = "a" if append else "w"
        with open(filename, mode, encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Research Note - {filename}\n")
            f.write(f"{'='*50}\n")
            f.write(content)
            f.write(f"\n{'='*50}\n\n")
        
        return {
            "success": True,
            "filename": filename,
            "content_length": len(content),
            "mode": mode
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "filename": filename
        }
```

---

## 5. Configuration Management

### 5.1 Environment Configuration
```python
# src/utils/config.py
import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class Config:
    """Configuration class for the application"""
    
    # API Keys
    openai_api_key: str
    huggingface_api_key: Optional[str] = None
    
    # Model Configuration
    default_model: str = "openai:gpt-4"
    fallback_model: str = "openai:gpt-3.5-turbo"
    
    # Agent Configuration
    max_retries: int = 3
    timeout: int = 30
    
    # File Configuration
    output_directory: str = "./outputs"
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables"""
        load_dotenv()
        
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY"),
            default_model=os.getenv("DEFAULT_MODEL", "openai:gpt-4"),
            fallback_model=os.getenv("FALLBACK_MODEL", "openai:gpt-3.5-turbo"),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            timeout=int(os.getenv("TIMEOUT", "30")),
            output_directory=os.getenv("OUTPUT_DIR", "./outputs"),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        return True

# Usage in other modules
config = Config.from_env()
```

### 5.2 Logging Configuration
```python
# src/utils/logging_utils.py
import logging
import sys
from typing import Optional

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    
    logger = logging.getLogger("smolagents_multiagent")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger
```

---

## 6. Error Handling and Validation

### 6.1 Custom Exceptions
```python
# src/utils/exceptions.py
class SmolagentsError(Exception):
    """Base exception for smolagents application"""
    pass

class AgentError(SmolagentsError):
    """Exception raised by agents"""
    
    def __init__(self, agent_name: str, message: str):
        self.agent_name = agent_name
        super().__init__(f"Agent '{agent_name}': {message}")

class ToolError(SmolagentsError):
    """Exception raised by tools"""
    
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}': {message}")

class ValidationError(SmolagentsError):
    """Exception raised during validation"""
    pass

class ConfigurationError(SmolagentsError):
    """Exception raised for configuration issues"""
    pass
```

### 6.2 Validation Utilities
```python
# src/utils/validation.py
from typing import Any, Dict, List
import re

def validate_topic(topic: str) -> bool:
    """Validate research topic"""
    if not topic or len(topic.strip()) < 3:
        raise ValidationError("Topic must be at least 3 characters long")
    
    if len(topic) > 200:
        raise ValidationError("Topic must be less than 200 characters")
    
    return True

def validate_agent_config(config: Dict[str, Any]) -> bool:
    """Validate agent configuration"""
    required_fields = ["model_name", "system_prompt"]
    
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Missing required field: {field}")
    
    return True

def validate_file_path(file_path: str) -> bool:
    """Validate file path for security"""
    # Prevent directory traversal
    if ".." in file_path or file_path.startswith("/"):
        raise ValidationError("Invalid file path")
    
    # Check file extension
    allowed_extensions = [".txt", ".md", ".json", ".csv"]
    if not any(file_path.endswith(ext) for ext in allowed_extensions):
        raise ValidationError("File extension not allowed")
    
    return True
```

---

## 7. Testing Framework

### 7.1 Test Configuration
```python
# tests/conftest.py
import pytest
from unittest.mock import Mock
from src.utils.config import Config

@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Mock(spec=Config)
    config.openai_api_key = "test-key"
    config.default_model = "openai:gpt-4"
    config.max_retries = 3
    return config

@pytest.fixture
def sample_research_data():
    """Sample research data for testing"""
    return """
    Artificial Intelligence has seen significant growth in 2024.
    Key developments include improvements in large language models,
    computer vision systems, and robotics applications.
    """

@pytest.fixture
def mock_agent():
    """Mock agent for testing"""
    agent = Mock()
    agent.run.return_value = "Mock agent response"
    return agent
```

### 7.2 Test Examples
```python
# tests/test_agents.py
import pytest
from unittest.mock import Mock, patch
from src.agents.research_agent import ResearchAgent
from src.utils.exceptions import AgentError

class TestResearchAgent:
    """Test cases for ResearchAgent"""
    
    def test_research_agent_initialization(self, mock_config):
        """Test agent initialization"""
        agent = ResearchAgent(model_name="test-model")
        assert agent.agent is not None
    
    @patch('src.agents.research_agent.ToolCallingAgent')
    def test_research_topic_success(self, mock_tool_calling_agent, sample_research_data):
        """Test successful research topic execution"""
        # Setup
        mock_agent_instance = Mock()
        mock_agent_instance.run.return_value = sample_research_data
        mock_tool_calling_agent.return_value = mock_agent_instance
        
        # Execute
        research_agent = ResearchAgent()
        result = research_agent.research_topic("AI trends")
        
        # Assert
        assert result == sample_research_data
        mock_agent_instance.run.assert_called_once()
    
    def test_research_topic_validation(self):
        """Test input validation for research topic"""
        research_agent = ResearchAgent()
        
        with pytest.raises(ValidationError):
            research_agent.research_topic("")  # Empty topic
        
        with pytest.raises(ValidationError):
            research_agent.research_topic("x" * 300)  # Too long topic
```

---

## 8. Performance and Monitoring

### 8.1 Performance Metrics
```python
# src/utils/metrics.py
import time
from typing import Dict, Any, Callable
from functools import wraps

def measure_execution_time(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper

class MetricsCollector:
    """Collect and store performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
    
    def record_metric(self, name: str, value: Any):
        """Record a metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            "value": value,
            "timestamp": time.time()
        })
    
    def get_average(self, metric_name: str) -> float:
        """Get average value for a metric"""
        if metric_name not in self.metrics:
            return 0.0
        
        values = [m["value"] for m in self.metrics[metric_name]]
        return sum(values) / len(values)
```

---

## 9. Cursor IDE Specific Features

### 9.1 Code Completion Hints
```python
# Type hints for better IDE support
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents import ToolCallingAgent
    from smolagents.tools import DuckDuckGoSearchTool

# Use string literals for forward references
def create_agent(model_name: str) -> 'ToolCallingAgent':
    """Function signature helps Cursor with type inference"""
    pass

# Detailed docstrings for better context
class ResearchAgent:
    """
    Research Agent for conducting information gathering tasks.
    
    This agent specializes in:
    - Web searches using DuckDuckGo
    - Information synthesis and summarization
    - Source citation and validation
    
    Example:
        >>> agent = ResearchAgent(model_name="openai:gpt-4")
        >>> result = agent.research_topic("climate change")
        >>> print(result)
    
    Attributes:
        agent (ToolCallingAgent): The underlying Smolagents agent
        model_name (str): Name of the model being used
    """
    pass
```

### 9.2 Debug Configuration
```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Main Application",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env",
            "args": []
        },
        {
            "name": "Debug Individual Agent",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/examples/basic_demo.py",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        },
        {
            "name": "Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${workspaceFolder}/tests"],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        }
    ]
}
```

---

## 10. Development Guidelines

### 10.1 Code Style
- Use Black formatter with 88 character line length
- Follow PEP 8 naming conventions
- Use type hints for all function parameters and return values
- Write comprehensive docstrings using Google style
- Use meaningful variable and function names

### 10.2 Git Workflow
```bash
# Feature development workflow
git checkout -b feature/new-agent-type
git add .
git commit -m "feat: Add new agent type for data processing"
git push origin feature/new-agent-type
# Create pull request
```

### 10.3 Documentation Standards
- Every class and function must have docstrings
- Include usage examples in docstrings
- Document parameter types and return values
- Maintain README.md with current setup instructions

---

## 11. Deployment Considerations

### 11.1 Environment Variables
```bash
# Production environment variables
ENVIRONMENT=production
OPENAI_API_KEY=your_production_key
LOG_LEVEL=WARNING
MAX_RETRIES=5
TIMEOUT=60
OUTPUT_DIR=/app/outputs
```

### 11.2 Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/src

# Run application
CMD ["uv", "run", "python", "main.py"]
```

---

## 12. Security Considerations

### 12.1 API Key Management
- Never commit API keys to version control
- Use environment variables or secure vaults
- Rotate API keys regularly
- Implement rate limiting for API calls

### 12.2 Input Validation
- Sanitize all user inputs
- Validate file paths to prevent directory traversal
- Limit file sizes and types
- Implement request timeouts

This technical requirements document provides Cursor IDE with comprehensive context about the project structure, patterns, and requirements for intelligent code completion and development assistance.