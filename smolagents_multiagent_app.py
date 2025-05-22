# Multi-Agent Research Assistant with Smolagents
# This example demonstrates key concepts for building multi-agent applications

import os
import time
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from smolagents import (
    ToolCallingAgent, 
    OpenAIServerModel,
    tool
)

# Load environment variables from .env file
load_dotenv()
print("Environment variables loaded from .env file")

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.environ.get("RETRY_DELAY", "2"))
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "deepseek/deepseek-r1")

# List of models known to support tool/function calling
TOOL_CALLING_MODELS = [
    "anthropic/claude-3-5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-sonnet",
    "google/gemini-1.5-pro",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "mistral/mixtral-8x7b-instruct",
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "openai/gpt-4",
    "openai/gpt-3.5-turbo",
    "deepseek/deepseek-r1"
]

def check_model_compatibility(model_id):
    """Check if the selected model is known to support tool/function calling."""
    for supported_model in TOOL_CALLING_MODELS:
        if model_id.lower().startswith(supported_model.lower()):
            return True
    
    print(f"⚠️  WARNING: Model '{model_id}' may not support tool calling.")
    print(f"Known compatible models: {', '.join(TOOL_CALLING_MODELS)}")
    print("The application may not work as expected.")
    return False

# Function to create models with OpenRouter
def create_openrouter_model(model_id=None):
    """Create an OpenRouter-compatible model instance."""
    if model_id is None:
        model_id = DEFAULT_MODEL
    
    # Check model compatibility
    check_model_compatibility(model_id)
        
    return OpenAIServerModel(
        model_id=model_id,
        api_base=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        max_completion_tokens=8192,
        timeout=60  # 60 second timeout for API calls
    )

# Retry decorator for API calls
def retry_on_error(max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Decorator to retry functions on failure with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        print(f"❌ Failed after {max_retries} retries: {str(e)}")
                        raise
                    wait_time = delay * (2 ** (retries - 1))  # Exponential backoff
                    print(f"⚠️ API call failed: {str(e)}. Retrying in {wait_time}s... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

# ============================================================================
# CUSTOM TOOLS DEFINITION
# ============================================================================

@tool
def save_research_note(content: str, filename: str = "research_notes.txt") -> str:
    """
    Save research findings to a file.
    
    Args:
        content: The research content to save
        filename: Name of the file to save to
    
    Returns:
        Success message
    """
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Research Note - {filename}\n")
            f.write(f"{'='*50}\n")
            f.write(content)
            f.write(f"\n{'='*50}\n\n")
        return f"Successfully saved research note to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

@tool  
def summarize_text(text: str, max_length: int = 200) -> str:
    """
    Create a concise summary of the given text.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary
        
    Returns:
        Summarized text
    """
    # Simple summarization (in practice, you might use a more sophisticated method)
    sentences = text.split('. ')
    if len(text) <= max_length:
        return text
    
    # Take first few sentences that fit within max_length
    summary = ""
    for sentence in sentences:
        if len(summary + sentence) <= max_length:
            summary += sentence + ". "
        else:
            break
    
    return summary.strip() or text[:max_length] + "..."

@tool
def validate_information(claim: str, source: str) -> str:
    """
    Validate information by checking source credibility and claim consistency.
    
    Args:
        claim: The claim to validate
        source: The source of the information
        
    Returns:
        Validation assessment
    """
    # Simple validation logic (in practice, this would be more sophisticated)
    credible_domains = ['wikipedia.org', 'reuters.com', 'bbc.com', '.edu', '.gov']
    
    credibility_score = 0
    for domain in credible_domains:
        if domain in source.lower():
            credibility_score += 1
    
    if credibility_score > 0:
        return f"✅ VALIDATED: Claim appears credible based on source analysis. Source: {source}"
    else:
        return f"⚠️  CAUTION: Source credibility unclear. Requires additional verification. Source: {source}"

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web and return results.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 5)
        
    Returns:
        Search results as text
    """
    try:
        # Try to use the duckduckgo_search library if available
        from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            for r in list(ddgs.text(query, max_results=num_results)):
                results.append(f"Title: {r['title']}\nURL: {r['href']}\nBody: {r['body']}\n")
        
        if not results:
            return "No results found."
        
        return "\n---\n".join(results)
    except ImportError:
        return "Error: duckduckgo_search library is not installed. Install with: pip install duckduckgo-search"
    except Exception as e:
        return f"Error performing search: {str(e)}"

@tool
def python_interpreter(code: str) -> str:
    """
    Execute Python code and return the result.
    
    Args:
        code: Python code to execute
        
    Returns:
        Result of execution
    """
    import io
    import sys
    import traceback
    
    # Redirect stdout to capture print statements
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    result = ""
    
    try:
        # Execute the code
        exec_globals = {}
        exec(code, exec_globals)
        result = new_stdout.getvalue()
        
        # If no output from print statements, check for a result variable
        if not result and 'result' in exec_globals:
            result = str(exec_globals['result'])
    except Exception as e:
        result = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
    finally:
        # Restore stdout
        sys.stdout = old_stdout
    
    return result

# ============================================================================
# AGENT CLASSES
# ============================================================================

class ResearchAgent:
    """Agent specialized in conducting research and gathering information."""
    
    def __init__(self, model_name: str = None):
        model = create_openrouter_model(model_name)
        
        # Simplified initialization with only essential parameters
        self.agent = ToolCallingAgent(
            tools=[web_search, save_research_note],
            model=model
        )
    
    @retry_on_error()
    def research_topic(self, topic: str) -> str:
        """Research a specific topic and save findings."""
        prompt = f"""
        You are a professional research assistant. Your role is to:
        1. Conduct thorough research on given topics
        2. Find reliable and credible sources
        3. Save important findings for later reference
        4. Provide comprehensive information with proper citations
        
        Always aim for accuracy and cite your sources clearly.
        
        Research the topic: "{topic}"
        
        Please:
        1. Search for current, reliable information using the web_search tool
        2. Summarize key findings in simple text format
        3. Save important information using the save_research_note tool
        4. List your sources
        
        IMPORTANT: Always format your responses as simple text. Do not use markdown or special formatting.
        """
        # Handle errors without using timeout
        try:
            return self.agent.run(prompt)
        except Exception as e:
            print(f"Error in research_topic: {str(e)}")
            # If we get a JSON parsing error, try to save something useful
            fallback_response = f"Research on {topic} could not be completed due to a technical error: {str(e)}"
            try:
                # Try to save at least something to the research notes
                save_research_note(fallback_response, f"{topic.replace(' ', '_')}_error_note.txt")
            except:
                pass
            return fallback_response

class AnalysisAgent:
    """Agent specialized in analyzing and processing information."""
    
    def __init__(self, model_name: str = None):
        model = create_openrouter_model(model_name)
        
        # Simplified initialization with only essential parameters
        self.agent = ToolCallingAgent(
            tools=[python_interpreter, summarize_text],
            model=model
        )
    
    @retry_on_error()
    def analyze_research(self, research_data: str) -> str:
        """Analyze research data and provide insights."""
        prompt = f"""
        You are an expert data analyst and information processor. Your role is to:
        1. Analyze research data and findings
        2. Create summaries and insights
        3. Perform calculations and statistical analysis when needed
        4. Present information in a clear, structured format
        
        Use Python for any calculations or data processing tasks.
        
        Analyze the following research data and provide insights:
        
        {research_data}
        
        Please:
        1. Identify key themes and patterns
        2. Summarize main points using the summarize_text tool
        3. Highlight any important statistics or data points
        4. Provide actionable insights or recommendations
        
        IMPORTANT: Always format your responses as simple text. Do not use markdown or special formatting.
        """
        # Handle errors without using timeout
        try:
            return self.agent.run(prompt)
        except Exception as e:
            print(f"Error in analyze_research: {str(e)}")
            fallback_response = f"Analysis could not be completed due to a technical error: {str(e)}"
            return fallback_response

class ValidationAgent:
    """Agent specialized in fact-checking and validation."""
    
    def __init__(self, model_name: str = None):
        model = create_openrouter_model(model_name)
        
        # Simplified initialization with only essential parameters
        self.agent = ToolCallingAgent(
            tools=[validate_information, web_search],
            model=model
        )
    
    @retry_on_error()
    def validate_claims(self, claims: List[str], sources: List[str]) -> str:
        """Validate multiple claims against their sources."""
        validations = []
        
        for claim, source in zip(claims, sources):
            prompt = f"""
            You are a fact-checker and validation specialist. Your role is to:
            1. Verify claims and statements for accuracy
            2. Check source credibility and reliability
            3. Cross-reference information with multiple sources
            4. Flag potential misinformation or inconsistencies
            
            Always be thorough and objective in your validation process.
            
            Validate this claim: "{claim}"
            From source: "{source}"
            
            Use the validate_information tool and conduct additional searches if needed.
            
            IMPORTANT: Always format your responses as simple text. Do not use markdown or special formatting.
            """
            try:
                validation = self.agent.run(prompt)
                validations.append(validation)
            except Exception as e:
                print(f"Error in validate_claims: {str(e)}")
                validations.append(f"Validation of claim '{claim}' could not be completed due to a technical error: {str(e)}")
        
        return "\n\n".join(validations)

class CoordinatorAgent:
    """Master agent that coordinates other agents and manages the workflow."""
    
    def __init__(self, model_name: str = None):
        model = create_openrouter_model(model_name)
        
        # Simplified initialization with only essential parameters
        self.agent = ToolCallingAgent(
            tools=[save_research_note],
            model=model
        )
        
        # Initialize specialist agents
        self.research_agent = ResearchAgent(model_name)
        self.analysis_agent = AnalysisAgent(model_name)
        self.validation_agent = ValidationAgent(model_name)
    
    @retry_on_error()
    def comprehensive_research_project(self, topic: str) -> Dict[str, Any]:
        """Coordinate a comprehensive research project across multiple agents."""
        
        print(f"🚀 Starting comprehensive research project on: {topic}")
        results = {}
        
        # Step 1: Research Phase
        print("📚 Phase 1: Conducting research...")
        try:
            research_results = self.research_agent.research_topic(topic)
            results['research'] = research_results
            print("✅ Research phase completed")
        except Exception as e:
            print(f"❌ Research phase failed: {str(e)}")
            results['research'] = f"Research failed: {str(e)}"
        
        # Step 2: Analysis Phase
        print("🔍 Phase 2: Analyzing findings...")
        try:
            analysis_results = self.analysis_agent.analyze_research(results.get('research', 'No research data available'))
            results['analysis'] = analysis_results
            print("✅ Analysis phase completed")
        except Exception as e:
            print(f"❌ Analysis phase failed: {str(e)}")
            results['analysis'] = f"Analysis failed: {str(e)}"
        
        # Step 3: Validation Phase
        print("✅ Phase 3: Validating information...")
        try:
            # Extract claims for validation (simplified approach)
            claims = [f"Key finding about {topic}"]  # In practice, extract from research_results
            sources = ["Research findings"]  # In practice, extract actual sources
            
            validation_results = self.validation_agent.validate_claims(claims, sources)
            results['validation'] = validation_results
            print("✅ Validation phase completed")
        except Exception as e:
            print(f"❌ Validation phase failed: {str(e)}")
            results['validation'] = f"Validation failed: {str(e)}"
        
        # Step 4: Final Synthesis
        print("📋 Phase 4: Creating final report...")
        try:
            final_report = self.create_final_report(topic, results)
            results['final_report'] = final_report
            
            # Save comprehensive report
            save_research_note(final_report, f"{topic.replace(' ', '_')}_comprehensive_report.txt")
            
            print("🎉 Comprehensive research project completed!")
        except Exception as e:
            print(f"❌ Final report creation failed: {str(e)}")
            results['final_report'] = f"Final report creation failed: {str(e)}"
            
            # Try to save an error report
            try:
                error_report = f"""
                Error Report for {topic} Research Project
                
                Research: {results.get('research', 'Not available')}
                
                Analysis: {results.get('analysis', 'Not available')}
                
                Validation: {results.get('validation', 'Not available')}
                
                Error: {str(e)}
                """
                save_research_note(error_report, f"{topic.replace(' ', '_')}_error_report.txt")
            except:
                pass
        
        return results
    
    @retry_on_error()
    def create_final_report(self, topic: str, results: Dict[str, Any]) -> str:
        """Create a final comprehensive report."""
        prompt = f"""
        You are a project coordinator managing a team of AI agents. Your role is to:
        1. Break down complex tasks into subtasks for specialist agents
        2. Coordinate workflow between different agents
        3. Synthesize results from multiple agents
        4. Ensure quality and completeness of final outputs
        5. Make decisions about next steps based on intermediate results
        
        You work with: Research Agent, Analysis Agent, and Validation Agent.
        
        Create a comprehensive final report for the research project on "{topic}".
        
        Combine and synthesize the following results:
        
        RESEARCH FINDINGS:
        {results.get('research', 'No research data available')}
        
        ANALYSIS RESULTS:
        {results.get('analysis', 'No analysis data available')}
        
        VALIDATION RESULTS:
        {results.get('validation', 'No validation data available')}
        
        Please create a well-structured final report that includes:
        1. Executive Summary
        2. Key Findings
        3. Analysis and Insights
        4. Validation Status
        5. Recommendations or Conclusions
        6. Areas for further research
        
        Save this report using the save_research_note tool.
        
        IMPORTANT: Always format your responses as simple text. Do not use markdown or special formatting.
        """
        
        # Handle errors without using timeout
        try:
            return self.agent.run(prompt)
        except Exception as e:
            print(f"Error in create_final_report: {str(e)}")
            fallback_response = f"Final report for {topic} could not be created due to a technical error: {str(e)}"
            return fallback_response

# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATIONS
# ============================================================================

def demo_individual_agents():
    """Demonstrate individual agent capabilities."""
    print("\n" + "="*60)
    print("DEMO: Individual Agent Capabilities")
    print("="*60)
    
    try:
        # Demo Research Agent
        print("\n🔬 Testing Research Agent:")
        research_agent = ResearchAgent()
        research_result = research_agent.research_topic("artificial intelligence trends 2024")
        print("Research completed!")
        
        # Demo Analysis Agent
        print("\n📊 Testing Analysis Agent:")
        analysis_agent = AnalysisAgent()
        analysis_result = analysis_agent.analyze_research("AI is growing rapidly with new developments in LLMs, computer vision, and robotics.")
        print("Analysis completed!")
        
        # Demo Validation Agent
        print("\n✅ Testing Validation Agent:")
        validation_agent = ValidationAgent()
        validation_result = validation_agent.validate_claims(
            ["AI market is growing rapidly"], 
            ["https://example.com/ai-report"]
        )
        print("Validation completed!")
    except Exception as e:
        print(f"❌ Error in demo_individual_agents: {str(e)}")
        # Continue with the program even if this demo fails
        pass

def demo_coordinated_project():
    """Demonstrate coordinated multi-agent project."""
    print("\n" + "="*60)
    print("DEMO: Coordinated Multi-Agent Project")
    print("="*60)
    
    try:
        coordinator = CoordinatorAgent()
        
        # Use a simpler topic to reduce chances of errors
        topic = "electric cars"
        print(f"Using simplified topic for demo: '{topic}'")
        
        # Set a short timeout for the demo
        start_time = time.time()
        max_time = 5 * 60  # 5 minutes max for demo
        
        # Run with time limit
        results = {}
        try:
            results = coordinator.comprehensive_research_project(topic)
        except Exception as e:
            print(f"❌ Comprehensive project failed: {str(e)}")
        
        # Check if we're exceeding time limit
        if time.time() - start_time > max_time:
            print("⚠️ Demo taking too long - proceeding with partial results")
        
        print(f"\n📋 Final Results Summary:")
        print(f"Research completed: {'✅' if results.get('research') and 'failed' not in results.get('research', '') else '❌'}")
        print(f"Analysis completed: {'✅' if results.get('analysis') and 'failed' not in results.get('analysis', '') else '❌'}")
        print(f"Validation completed: {'✅' if results.get('validation') and 'failed' not in results.get('validation', '') else '❌'}")
        print(f"Final report created: {'✅' if results.get('final_report') and 'failed' not in results.get('final_report', '') else '❌'}")
    except Exception as e:
        print(f"❌ Error in demo_coordinated_project: {str(e)}")
        # Continue with the program even if this demo fails
        pass

def demo_custom_tools():
    """Demonstrate custom tool usage."""
    print("\n" + "="*60)
    print("DEMO: Custom Tools")
    print("="*60)
    
    # Test custom tools directly
    print("💾 Testing save_research_note tool:")
    result1 = save_research_note("This is a test research note.", "demo_notes.txt")
    print(result1)
    
    print("\n📝 Testing summarize_text tool:")
    long_text = "Artificial intelligence is rapidly evolving. Machine learning algorithms are becoming more sophisticated. Deep learning has revolutionized computer vision and natural language processing. The future holds many possibilities for AI applications in various industries."
    summary = summarize_text(long_text, 100)
    print(f"Summary: {summary}")
    
    print("\n🔍 Testing validate_information tool:")
    validation = validate_information("AI is transforming industries", "https://wikipedia.org/ai-impact")
    print(validation)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# if __name__ == "__main__":
#     print("🤖 Multi-Agent Application with Smolagents using OpenRouter API")
#     print("=" * 50)
#     print(f"Using model: {DEFAULT_MODEL}")
#     print(f"API endpoint: {OPENROUTER_BASE_URL}")
    
#     # Check for API key
#     if not OPENROUTER_API_KEY:
#         print("❌ Error: OPENROUTER_API_KEY environment variable not set.")
#         print("Please set your OpenRouter API key using:")
#         print("export OPENROUTER_API_KEY='your-openrouter-api-key'")
#         exit(1)
    
#     # Check model compatibility
#     if not check_model_compatibility(DEFAULT_MODEL):
#         user_input = input("Continue anyway? (y/n): ")
#         if user_input.lower() != 'y':
#             print("Exiting application.")
#             sys.exit(0)
    
#     try:
#         print("Starting demo_custom_tools()...")
#         # Run demonstrations
#         demo_custom_tools()
#         print("Completed demo_custom_tools()")
        
#         print("Starting demo_individual_agents()...")
#         demo_individual_agents()
#         print("Completed demo_individual_agents()")
        
#         print("Starting demo_coordinated_project()...")
#         demo_coordinated_project()
#         print("Completed demo_coordinated_project()")
        
#         print("\n🎉 All demos completed successfully!")
#         print("\n📚 Learning Points:")
#         print("1. Custom tools extend agent capabilities")
#         print("2. Specialized agents handle specific tasks better")
#         print("3. Coordination agents manage complex workflows")
#         print("4. Multi-agent systems can tackle complex problems")
        
#     except Exception as e:
#         print(f"❌ Error during execution: {str(e)}")
#         print("Make sure you have:")
#         print("1. Installed smolagents: pip install smolagents")
#         print("2. Set up your OpenRouter API key properly")
#         print("3. All required dependencies installed")
#     finally:
#         print("Program finished. Exiting...")
#         # Force exit to ensure no hanging threads
#         sys.exit(0)

# ============================================================================
# ADVANCED PATTERNS AND EXTENSIONS
# ============================================================================

class AdvancedMultiAgentOrchestrator:
    """
    Advanced orchestrator demonstrating more complex patterns:
    - Agent communication
    - Task queuing
    - Result caching
    - Error handling and retry logic
    """
    
    def __init__(self):
        self.agents = {
            'research': ResearchAgent(),
            'analysis': AnalysisAgent(),
            'validation': ValidationAgent(),
            'coordinator': CoordinatorAgent()
        }
        self.task_queue = []
        self.results_cache = {}
    
    def add_task(self, task_type: str, task_data: Dict[str, Any]):
        """Add a task to the processing queue."""
        self.task_queue.append({
            'type': task_type,
            'data': task_data,
            'status': 'pending'
        })
    
    def process_task_queue(self):
        """Process all tasks in the queue."""
        for task in self.task_queue:
            if task['status'] == 'pending':
                try:
                    result = self.execute_task(task)
                    task['status'] = 'completed'
                    task['result'] = result
                except Exception as e:
                    task['status'] = 'failed'
                    task['error'] = str(e)
    
    def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a specific task based on its type."""
        task_type = task['type']
        task_data = task['data']
        
        if task_type == 'research':
            return self.agents['research'].research_topic(task_data['topic'])
        elif task_type == 'analysis':
            return self.agents['analysis'].analyze_research(task_data['data'])
        elif task_type == 'validation':
            return self.agents['validation'].validate_claims(
                task_data['claims'], 
                task_data['sources']
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def collaborative_research(self, topics: List[str]) -> Dict[str, Any]:
        """
        Demonstrate collaborative research across multiple topics.
        """
        print(f"🔬 Starting collaborative research on {len(topics)} topics...")
        
        # Add research tasks for each topic
        for topic in topics:
            self.add_task('research', {'topic': topic})
        
        # Process all research tasks
        self.process_task_queue()
        
        # Collect all research results
        research_results = []
        for task in self.task_queue:
            if task['type'] == 'research' and task['status'] == 'completed':
                research_results.append(task['result'])
        
        # Analyze combined results
        combined_research = "\n\n".join(research_results)
        analysis_result = self.agents['analysis'].analyze_research(combined_research)
        
        return {
            'individual_research': research_results,
            'combined_analysis': analysis_result,
            'task_summary': {
                'total_tasks': len(self.task_queue),
                'completed': len([t for t in self.task_queue if t['status'] == 'completed']),
                'failed': len([t for t in self.task_queue if t['status'] == 'failed'])
            }
        }

# Example of using the advanced orchestrator
def demo_advanced_patterns():
    """Demonstrate advanced multi-agent patterns."""
    print("\n" + "="*60)
    print("DEMO: Advanced Multi-Agent Patterns")
    print("="*60)

    orchestrator = AdvancedMultiAgentOrchestrator()
    
    # Collaborative research on related topics
    topics = [
        "new battery technologies in automotive industry"
    ]
    
    results = orchestrator.collaborative_research(topics)
    
    print(f"📊 Research Summary:")
    print(f"Topics researched: {len(topics)}")
    print(f"Tasks completed: {results['task_summary']['completed']}")
    print(f"Tasks failed: {results['task_summary']['failed']}")
    
    return results

#Uncomment to run advanced demo
if __name__ == "__main__":
    demo_advanced_patterns()