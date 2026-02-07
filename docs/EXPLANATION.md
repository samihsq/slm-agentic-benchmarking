# Technical Explanation: SLM Agentic Benchmarking Framework

This document explains how the benchmarking framework works, its architecture, and how to extend it.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [CrewAI Integration](#crewai-integration)
3. [Agent Architectures](#agent-architectures)
4. [Benchmark Integration](#benchmark-integration)
5. [Cost Tracking System](#cost-tracking-system)
6. [Azure Deployment](#azure-deployment)
7. [Extending the Framework](#extending-the-framework)

---

## Architecture Overview

### High-Level Flow

```
User Request → CLI (run_benchmark.py)
    ↓
Agent Selection (OneShot/Sequential/Concurrent/GroupChat/Baseline)
    ↓
Model Configuration (Azure LLM via LiteLLM)
    ↓
Benchmark Runner (MedAgentBench/MedQA/MCP-Bench/BFCL)
    ↓
Cost Tracker (Real-time monitoring)
    ↓
Results Storage (JSON files in results/)
```

### Component Responsibilities

**`src/agents/`** - Agent architectures that orchestrate LLM calls
- `BaseAgent` - Abstract interface all agents implement
- `OneShotAgent` - Direct LLM call (non-agentic baseline)
- `SequentialAgent` - Multi-stage pipeline (Analyze → Evaluate → Respond)
- `ConcurrentAgent` - Parallel specialized agents → Synthesizer
- `GroupChatAgent` - Collaborative discussion → Moderator
- `BaselineAgent` - SOTA frontier model (GPT-4o) for comparison

**`src/config/`** - Model configuration and Azure integration
- `azure_llm_config.py` - Maps model names to Azure endpoints, handles credentials

**`src/benchmarks/`** - Benchmark-specific runners
- `medical/` - MedAgentBench, MedQA, MedMCQA
- `tool_calling/` - MCP-Bench, BFCL v3

**`src/evaluation/`** - Metrics and cost tracking
- `cost_tracker.py` - Budget monitoring, usage logging
- `metrics.py` - Performance metrics calculation

**`run_benchmark.py`** - Unified CLI entry point

---

## CrewAI Integration

### What is CrewAI?

CrewAI is a framework for orchestrating multiple AI agents to work together. In this project, we use it to create **agentic architectures** where multiple specialized agents collaborate to solve tasks.

### How We Use CrewAI

#### 1. Agent Definition

Each agent in CrewAI has:
- **Role**: What the agent does (e.g., "Task Analyzer")
- **Goal**: What it's trying to achieve
- **Backstory**: Context about its expertise
- **LLM**: The language model it uses

Example from `SequentialAgent`:

```python
self.analyzer = Agent(
    role="Task Analyzer",
    goal="Analyze the nature and requirements of tasks",
    backstory="""You are an expert at understanding task requirements.
    Your job is to identify:
    - What is the core objective of the task
    - What domain or topic does this relate to""",
    llm=self.llm,  # Shared LLM instance
    verbose=self.verbose,
)
```

#### 2. Task Definition

Tasks define what agents should do:

```python
analyze_task = Task(
    description="Analyze this task: {task}",
    expected_output="Analysis of the task's nature",
    agent=self.analyzer,  # Which agent does this
)
```

#### 3. Crew Orchestration

A `Crew` coordinates agents and tasks:

```python
crew = Crew(
    agents=[self.analyzer, self.evaluator, self.responder],
    tasks=[analyze_task, evaluate_task, respond_task],
    process=Process.sequential,  # Run tasks in order
    verbose=self.verbose,
)

result = crew.kickoff()  # Execute the crew
```

### Process Types

- **`Process.sequential`**: Tasks run one after another (used in SequentialAgent)
- **`Process.hierarchical`**: Tasks organized in a hierarchy (not used here)
- **Concurrent execution**: Achieved via task `context` dependencies (used in ConcurrentAgent)

### Why CrewAI?

1. **Modularity**: Easy to swap agents or add new stages
2. **Reusability**: Same agent can be used in different crews
3. **Transparency**: Each agent's reasoning is visible
4. **Standardization**: Consistent interface across architectures

---

## Agent Architectures

### 1. OneShotAgent (Non-Agentic Baseline)

**Purpose**: Direct LLM call without any orchestration. This is the **true baseline** - 1 task = 1 API call.

**How it works**:
```python
# Direct LiteLLM call, bypasses CrewAI
response = litellm.completion(
    model=self.model_id,
    messages=[system_prompt, user_message],
    ...
)
```

**Use case**: Compare agentic architectures against simple LLM calls.

**Token usage**: Minimal (just the task + response)

---

### 2. SequentialAgent (Pipeline Pattern)

**Purpose**: Multi-stage processing where each stage builds on the previous.

**Architecture**:
```
Task → [Analyzer] → [Evaluator] → [Responder] → Response
```

**How it works**:
1. **Analyzer** receives the task, identifies requirements
2. **Evaluator** receives analyzer output, determines approach
3. **Responder** receives both, generates final response

**CrewAI Implementation**:
```python
# Tasks have context dependencies
evaluate_task = Task(
    description="...",
    agent=self.evaluator,
    context=[analyze_task],  # Can see analyzer's output
)

respond_task = Task(
    description="...",
    agent=self.responder,
    context=[analyze_task, evaluate_task],  # Can see both
)
```

**Token usage**: ~3x OneShot (each agent processes the task)

**When to use**: Tasks that benefit from structured reasoning stages.

---

### 3. ConcurrentAgent (Parallel Processing)

**Purpose**: Multiple specialized agents work in parallel, then synthesize.

**Architecture**:
```
Task → [Analyst] ┐
      [Researcher] → [Synthesizer] → Response
      [Critic] ┘
```

**How it works**:
1. Three agents analyze the task from different angles simultaneously
2. Synthesizer combines their outputs into final response

**CrewAI Implementation**:
```python
# All three tasks run independently
analyze = Task(agent=self.analyst, ...)
research = Task(agent=self.researcher, ...)
critique = Task(agent=self.critic, ...)

# Synthesizer depends on all three
synthesize = Task(
    agent=self.synthesizer,
    context=[analyze, research, critique],  # Sees all outputs
)
```

**Token usage**: ~4x OneShot (3 parallel agents + synthesizer)

**When to use**: Complex tasks needing multiple perspectives.

---

### 4. GroupChatAgent (Collaborative Discussion)

**Purpose**: Agents discuss and refine ideas together.

**Architecture**:
```
Task → [Proposer] ←→ [Critic] ←→ [Advisor]
              ↓           ↓           ↓
         [Moderator] → Response
```

**How it works**:
1. **Proposer** suggests initial solution
2. **Critic** challenges and identifies weaknesses
3. **Advisor** provides expert guidance
4. **Moderator** synthesizes discussion into final answer

**CrewAI Implementation**:
```python
# Sequential discussion
propose = Task(agent=self.proposer, ...)
critique = Task(agent=self.critic, context=[propose], ...)
advise = Task(agent=self.advisor, context=[propose, critique], ...)
moderate = Task(agent=self.moderator, context=[propose, critique, advise], ...)
```

**Token usage**: ~4-5x OneShot (multiple discussion rounds)

**When to use**: Tasks requiring careful deliberation and refinement.

---

### 5. BaselineAgent (SOTA Comparison)

**Purpose**: Run frontier models (GPT-4o) in non-agentic mode for upper-bound comparison.

**How it works**: Identical to OneShotAgent but defaults to GPT-4o/GPT-4o-mini.

**Use case**: "What's the best possible performance?" baseline.

---

## Benchmark Integration

### Benchmark Runner Pattern

All benchmarks follow the same pattern:

```python
class BenchmarkRunner:
    def __init__(self, agent, cost_tracker, verbose):
        self.agent = agent
        self.cost_tracker = cost_tracker
    
    def load_tasks(self, limit=None):
        """Load benchmark tasks"""
        # Returns list of task dicts
    
    def run(self, limit=None, save_results=True):
        """Run benchmark"""
        tasks = self.load_tasks(limit)
        results = []
        
        for task in tasks:
            # Run agent
            response = self.agent.respond_to_task(task["task"], context)
            
            # Track cost
            cost = self.cost_tracker.log_usage(...)
            
            # Evaluate result
            result = EvaluationResult(...)
            results.append(result)
        
        return results
```

### Medical Benchmarks

#### MedAgentBench

**What it tests**: Agentic capabilities in clinical scenarios
- Multi-step workflows (e.g., "Review patient history, order labs, interpret results")
- Tool use (FHIR-compliant EMR APIs)
- Iterative refinement

**Integration**:
```python
runner = MedAgentBenchRunner(agent, cost_tracker)
results = runner.run(limit=100)  # 100 clinical tasks
```

**Task format**:
```python
{
    "task_id": "medagent_001",
    "category": "diagnosis",
    "task": "A 45-year-old patient presents with...",
    "patient_data": {...},  # FHIR-compliant data
    "ground_truth": "acute myocardial infarction"
}
```

#### MedQA/MedMCQA

**What it tests**: Medical knowledge (not agentic capabilities)
- Multiple-choice questions
- USMLE-style (MedQA) or Indian medical exams (MedMCQA)

**Integration**:
```python
runner = MedQARunner(agent, cost_tracker, dataset="medqa")
results = runner.run(limit=100)
```

**Evaluation**: Checks if agent selects correct answer (A/B/C/D)

---

### Tool Calling Benchmarks

#### MCP-Bench

**What it tests**: Tool-use capabilities
- Tool schema understanding
- Multi-hop planning (use tool A, then tool B)
- Parameter precision

**Integration**:
```python
runner = MCPBenchRunner(agent, cost_tracker)
results = runner.run(limit=250)
```

**Task format**:
```python
{
    "task_id": "mcp_001",
    "description": "Search for papers and get top 3 results",
    "tools": [
        {"name": "search_papers", "parameters": {...}},
        {"name": "get_paper_details", "parameters": {...}}
    ],
    "expected_tool_calls": ["search_papers", "get_paper_details"]
}
```

**Evaluation**: Checks if agent calls expected tools with correct parameters

#### BFCL v3

**What it tests**: Function calling fundamentals
- Simple calls (single function)
- Parallel calls (multiple functions simultaneously)
- Nested calls (function A → function B)

**Integration**:
```python
runner = BFCLRunner(agent, cost_tracker)
results = runner.run(limit=2000)
```

---

## Cost Tracking System

### How It Works

The `CostTracker` monitors spending in real-time:

```python
tracker = CostTracker(budget_limit=10000.0)

# After each API call
cost = tracker.log_usage(
    model="phi-4",
    prompt_tokens=1000,
    completion_tokens=500,
    task_id="task_001",
    benchmark="medqa",
    agent_type="SequentialAgent",
)
```

### Cost Calculation

```python
def _calculate_cost(model, prompt_tokens, completion_tokens):
    pricing = PRICING_PER_1M[model]
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
```

### Budget Alerts

Alerts trigger at configurable thresholds (default: 30%, 60%, 90%):

```python
if percent_used >= threshold:
    print(f"⚠️  BUDGET ALERT: {threshold * 100}% used")
```

### Pre-Run Estimation

Before running experiments, estimate costs:

```python
estimated = tracker.estimate_cost(
    model="phi-4",
    num_tasks=100,
    avg_prompt_tokens=3000,
    avg_completion_tokens=2000,
)
```

### Usage Logging

All usage is logged to `cost_tracking.json`:

```json
{
  "total_spent": 45.23,
  "total_tokens": 1234567,
  "records": [
    {
      "timestamp": "2026-01-15T10:30:00",
      "model": "phi-4",
      "task_id": "medqa_001",
      "prompt_tokens": 1000,
      "completion_tokens": 500,
      "cost": 0.14,
      "benchmark": "medqa",
      "agent_type": "SequentialAgent"
    }
  ]
}
```

---

## Azure Deployment

### Model Types

#### 1. Serverless Models (Azure OpenAI / Azure AI Foundry)

**No infrastructure needed** - just API calls:

```python
# Azure OpenAI (GPT-4o)
llm_kwargs = {
    "model": "azure/gpt-4o",
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "api_base": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_version": "2024-08-01-preview",
}

# Azure AI Foundry (Phi-4, Llama, etc.)
llm_kwargs = {
    "model": "azure_ai/Phi-4",
    "api_key": os.getenv("AZURE_AI_API_KEY"),
}
```

**Pricing**: Pay-per-token, no fixed costs

#### 2. Azure ML Endpoints (Custom Models)

**Requires deployment** - models hosted on Azure ML:

```python
llm_kwargs = {
    "model": "azure_ml/glm-4-7-flash",
    "api_base": os.getenv("AZURE_ML_ENDPOINT"),
}
```

**Pricing**: Pay-per-token + endpoint hosting (~$0.50-2/hr)

### LiteLLM Integration

LiteLLM provides a unified interface for all providers:

```python
import litellm

# Works with Azure OpenAI, Azure AI Foundry, Azure ML, etc.
response = litellm.completion(
    model="azure/gpt-4o",  # LiteLLM routes automatically
    messages=[...],
    api_key=api_key,
    api_base=api_base,
)
```

**Why LiteLLM?**
- Single API for all providers
- Automatic retries and error handling
- Token usage tracking
- Cost estimation

### CrewAI LLM Wrapper

CrewAI uses LiteLLM under the hood:

```python
from crewai import LLM

llm = LLM(
    model="azure/gpt-4o",
    api_key=api_key,
    api_base=api_base,
)

agent = Agent(llm=llm, ...)  # CrewAI handles LLM calls
```

---

## Extending the Framework

### Adding a New Agent Architecture

1. **Create agent class** in `src/agents/`:

```python
from .base_agent import BaseAgent, BenchmarkResponse
from crewai import Agent, Task, Crew, Process
from ..config import get_llm

class MyCustomAgent(BaseAgent):
    def __init__(self, model="phi-4", verbose=False):
        super().__init__(model, verbose)
        self.llm = get_llm(model)
        self._setup_agents()
    
    def _setup_agents(self):
        self.agent1 = Agent(
            role="Role 1",
            goal="Goal 1",
            backstory="...",
            llm=self.llm,
        )
        # ... more agents
    
    def respond_to_task(self, task, context=None):
        # Define tasks
        task1 = Task(description=task, agent=self.agent1, ...)
        
        # Create crew
        crew = Crew(
            agents=[self.agent1, ...],
            tasks=[task1, ...],
            process=Process.sequential,
        )
        
        # Execute
        result = crew.kickoff()
        return self.parse_json_response(str(result))
```

2. **Register in `src/agents/__init__.py`**

3. **Add to CLI** in `run_benchmark.py`:

```python
agents = {
    "my_custom": lambda: MyCustomAgent(model=model, verbose=verbose),
    ...
}
```

---

### Adding a New Benchmark

1. **Create runner class** in `src/benchmarks/`:

```python
from ...agents.base_agent import BaseAgent, EvaluationResult
from ...evaluation.cost_tracker import CostTracker

class MyBenchmarkRunner:
    def __init__(self, agent, cost_tracker, verbose=False):
        self.agent = agent
        self.cost_tracker = cost_tracker
    
    def load_tasks(self, limit=None):
        # Load benchmark tasks
        return tasks
    
    def run(self, limit=None, save_results=True):
        # Run benchmark
        tasks = self.load_tasks(limit)
        results = []
        
        for task in tasks:
            response = self.agent.respond_to_task(task["task"])
            # Evaluate, track cost, etc.
            results.append(result)
        
        return results
```

2. **Register in `src/benchmarks/__init__.py`**

3. **Add to CLI** in `run_benchmark.py`

---

### Adding a New Model

1. **Add to `src/config/azure_llm_config.py`**:

```python
AVAILABLE_MODELS = {
    "my-model": {
        "model": "azure_ai/My-Model",
        "description": "My Model Description",
        "context_window": 128000,
        "cost_per_1m_input": 0.10,
        "cost_per_1m_output": 0.20,
        "provider": "my_provider",
        "serverless": True,  # or False if Azure ML
    },
}
```

2. **Add pricing to `CostTracker.PRICING_PER_1M`**

---

### Customizing Cost Tracking

Modify `src/evaluation/cost_tracker.py`:

```python
# Add custom pricing
PRICING_PER_1M["my-model"] = {"input": 0.10, "output": 0.20}

# Custom alert thresholds
tracker = CostTracker(
    budget_limit=5000.0,
    alert_thresholds=[0.25, 0.50, 0.75, 0.90],  # Custom thresholds
)
```

---

## Data Flow Example

### Running a Benchmark

1. **CLI invocation**:
   ```bash
   python run_benchmark.py --model phi-4 --agent sequential --benchmark medqa
   ```

2. **Agent creation**:
   ```python
   agent = SequentialAgent(model="phi-4", verbose=False)
   # Sets up 3 CrewAI agents (analyzer, evaluator, responder)
   ```

3. **Benchmark runner**:
   ```python
   runner = MedQARunner(agent, cost_tracker)
   tasks = runner.load_tasks(limit=10)
   ```

4. **For each task**:
   - Runner calls `agent.respond_to_task(task, context)`
   - Agent creates CrewAI crew and executes
   - CrewAI makes LLM calls via LiteLLM → Azure
   - Response parsed and returned
   - Cost tracked
   - Result evaluated

5. **Results saved**:
   - JSON file in `results/medqa/`
   - Cost summary printed

---

## Key Design Decisions

### Why CrewAI?

- **Standardization**: Consistent agent interface
- **Modularity**: Easy to swap components
- **Transparency**: See each agent's reasoning
- **Extensibility**: Add new agents easily

### Why LiteLLM?

- **Unified API**: Same code for all providers
- **Automatic routing**: Handles Azure endpoints
- **Token tracking**: Built-in usage monitoring
- **Error handling**: Retries and fallbacks

### Why Separate Benchmark Runners?

- **Isolation**: Each benchmark has unique evaluation logic
- **Reusability**: Same runner works with any agent
- **Extensibility**: Easy to add new benchmarks
- **Maintainability**: Changes to one benchmark don't affect others

### Why Cost Tracking Module?

- **Budget compliance**: Stay under $10k limit
- **Transparency**: Know exactly what costs what
- **Optimization**: Identify expensive operations
- **Planning**: Estimate before running

---

## Troubleshooting

### Common Issues

**"Unknown model" error**:
- Check model name matches `AVAILABLE_MODELS` keys
- Verify Azure credentials are set

**"AZURE_OPENAI_API_KEY not set"**:
- Set environment variables before running
- Check `.env` file if using python-dotenv

**CrewAI errors**:
- Ensure CrewAI version is compatible (`^0.80.0`)
- Check LLM configuration is valid

**Cost tracking not working**:
- Verify `cost_tracking.json` is writable
- Check token counts are being logged in response metadata

---

## Performance Considerations

### Token Usage by Architecture

| Architecture | Token Multiplier | Use Case |
|--------------|-----------------|----------|
| OneShot | 1x | Baseline, simple tasks |
| Sequential | ~3x | Structured reasoning |
| Concurrent | ~4x | Multi-perspective analysis |
| GroupChat | ~4-5x | Deliberation needed |

### Optimization Tips

1. **Use OneShot for simple tasks** - Don't over-engineer
2. **Limit task count during development** - Use `--limit 10`
3. **Use cost-efficient models** - `gpt-4o-mini` for baseline, `phi-4` for SLMs
4. **Monitor costs in real-time** - Check `cost_tracking.json` frequently
5. **Run estimates first** - Use `--estimate` before full runs

---

## Next Steps

1. **Set up Azure credentials** (see README.md)
2. **Run cost estimation**: `python run_benchmark.py --estimate`
3. **Test with small limit**: `python run_benchmark.py --model phi-4 --agent one_shot --benchmark medqa --limit 5`
4. **Compare architectures**: Use `--compare-baseline` flag
5. **Scale up**: Remove `--limit` for full benchmarks

For questions or issues, refer to the code comments or the plan in `docs/PLAN.md`.
