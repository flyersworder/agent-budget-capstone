# Part 1 Execution Plan: Single-Agent Strategic Allocation with Tools

**Created**: November 16, 2025
**Status**: Planning Complete, Ready for Implementation
**Revised**: November 16, 2025 - Using ADK Agents with Tools
**Technical Verification**: November 16, 2025 - Confirmed model & tools

## Technical Verification Summary

✅ **Confirmed Working Configuration:**
- **Model**: `gemini-2.5-flash-lite` - Fast, cost-effective, supports thinking budget
- **Google Search Tool**: `from google.adk.tools import google_search`
- **Thinking Control**: `ThinkingConfig(thinking_budget=<tokens>, include_thoughts=True)`
- **Output Control**: `GenerateContentConfig(max_output_tokens=<tokens>)`

This configuration allows us to:
1. Control reasoning vs output token allocation
2. Use Google Search for realistic research tasks
3. Track tool usage patterns across strategies
4. Analyze thinking process (via `include_thoughts=True`)

## Executive Summary

This plan outlines the implementation of Part 1 from the capstone project, focusing on demonstrating how different token allocations between reasoning and output affect **agent behavior with tool use**. This is much more interesting than simple LLM calls because:

1. **Tool selection decisions** require reasoning tokens
2. **Tool orchestration** (which tool? when? how many times?) varies by strategy
3. **Synthesis** of tool results shows quality differences
4. **Real-world relevance**: Production agents use tools extensively

## Key Insight: Why Agents + Tools Matter

**Different token allocation strategies will behave differently:**

- **Deep Thinker (80% reasoning / 20% output)**:
  - More deliberate tool selection
  - Better query formulation
  - Fewer unnecessary tool calls
  - Concise but insightful synthesis

- **Balanced (50/50)**:
  - Moderate planning
  - Reasonable tool usage
  - Good synthesis quality

- **Verbose (20% reasoning / 80% output)**:
  - Quick, possibly inefficient tool calls
  - More tool invocations
  - Long explanations but potentially shallow insights

## Understanding from Google ADK

### Key Capabilities

1. **Agent Creation with Budget Control**:
   ```python
   from google.adk.agents import Agent
   from google.adk.planners import BuiltInPlanner
   from google.genai import types

   agent = Agent(
       model="gemini-2.5-flash-lite",  # Supports thinking budget
       planner=BuiltInPlanner(
           thinking_config=types.ThinkingConfig(
               thinking_budget=8000,  # Reasoning tokens
               include_thoughts=True  # See thinking process
           )
       ),
       generate_content_config=types.GenerateContentConfig(
           max_output_tokens=2000  # Output tokens
       ),
       tools=[google_search]  # Built-in Google Search tool
   )
   ```

2. **Usage Tracking** (via Runner):
   ```python
   from google.adk.runners import Runner

   runner = Runner(agent=agent, app_name="experiment")
   # Usage tracked in session events
   ```

3. **Tool Integration**:
   - Google Search
   - Web browsing
   - Custom tools

### Implementation Approach

**Choice: Google ADK Agents with Tools** ✅

**Rationale**:
- Shows realistic agent behavior
- Tool usage patterns reveal reasoning quality
- More interesting than simple text generation
- Directly applicable to production scenarios

**Code Pattern**:
```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search  # ✅ Direct import of google_search
from google.genai import types

# Create agent with specific budget
def create_agent(strategy: AllocationStrategy, total_budget: int = 10000):
    budget = strategy.create_budget(total_budget)

    return Agent(
        model="gemini-2.5-flash-lite",  # ✅ Supports thinking, fast & cost-effective
        name=f"{strategy.value}_agent",
        instruction=f"You are a research assistant with {strategy.value} allocation.",
        tools=[google_search],  # ✅ Built-in Google Search tool
        planner=BuiltInPlanner(
            thinking_config=types.ThinkingConfig(
                thinking_budget=budget.reasoning_tokens,
                include_thoughts=True  # So we can analyze reasoning
            )
        ),
        generate_content_config=types.GenerateContentConfig(
            max_output_tokens=budget.output_tokens,
            temperature=0.0
        )
    )
```

## Project Structure

```
agent-budget-capstone/
├── .env                        # API keys (not committed)
├── pyproject.toml              # Dependencies
├── agent_budget/               # Core framework
│   ├── __init__.py
│   ├── core.py                 # TokenBudget, AllocationStrategy
│   ├── agent_factory.py        # Creates agents with different budgets
│   └── monitor.py              # Usage tracking and analysis
├── experiments/                # Experiment infrastructure
│   ├── __init__.py
│   ├── tasks/                  # Demo tasks requiring tool use
│   │   ├── __init__.py
│   │   ├── research_tasks.py   # Research questions
│   │   ├── fact_check_tasks.py # Fact-checking tasks
│   │   └── comparison_tasks.py # Comparison analysis
│   ├── run_single_agent.py     # Main experiment runner
│   ├── evaluator.py            # LLM judge for quality scoring
│   └── results/                # Output data (CSV, JSON)
├── notebooks/                  # Jupyter demos
│   └── part1_demo.ipynb        # Interactive demonstration
└── docs/                       # Documentation
    └── RESULTS.md              # Findings and insights
```

## Implementation Timeline (7 Days)

### Day 1-2: Core Framework

**Files to Create**:
1. `agent_budget/core.py`
2. `agent_budget/agent_factory.py`
3. `agent_budget/monitor.py`

**Key Classes**:

```python
# agent_budget/core.py
from dataclasses import dataclass
from enum import Enum

@dataclass
class TokenBudget:
    reasoning_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.reasoning_tokens + self.output_tokens

    def validate(self) -> bool:
        return self.reasoning_tokens >= 0 and self.output_tokens >= 0

class AllocationStrategy(Enum):
    DEEP_THINKER = "deep"     # 80/20 - More reasoning, less output
    BALANCED = "balanced"      # 50/50 - Equal allocation
    VERBOSE = "verbose"        # 20/80 - Less reasoning, more output

    def create_budget(self, total_tokens: int) -> TokenBudget:
        """Create budget for this strategy."""
        ratios = {
            "deep": (0.8, 0.2),
            "balanced": (0.5, 0.5),
            "verbose": (0.2, 0.8)
        }
        r_ratio, o_ratio = ratios[self.value]
        return TokenBudget(
            reasoning_tokens=int(total_tokens * r_ratio),
            output_tokens=int(total_tokens * o_ratio)
        )
```

```python
# agent_budget/agent_factory.py
from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search
from google.genai import types
from .core import AllocationStrategy, TokenBudget

class AgentFactory:
    """Factory for creating agents with different token allocations."""

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self.model = model

    def create_agent(
        self,
        strategy: AllocationStrategy,
        total_budget: int = 10000,
        tools: list | None = None
    ) -> Agent:
        """Create an agent with specific budget allocation."""

        budget = strategy.create_budget(total_budget)

        if tools is None:
            tools = [google_search]  # Default to built-in Google Search

        instructions = {
            "deep": (
                "You are a research assistant that thinks deeply before acting. "
                "Carefully plan your tool usage. Use tools only when necessary. "
                "Provide concise, insightful responses."
            ),
            "balanced": (
                "You are a research assistant with balanced thinking and output. "
                "Use tools when helpful. Provide clear, well-reasoned responses."
            ),
            "verbose": (
                "You are a research assistant that provides detailed explanations. "
                "Use tools to gather comprehensive information. "
                "Provide thorough, detailed responses."
            )
        }

        return Agent(
            model=self.model,
            name=f"{strategy.value}_agent",
            instruction=instructions[strategy.value],
            description=f"Agent with {strategy.value} token allocation strategy",
            tools=tools,
            planner=BuiltInPlanner(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=budget.reasoning_tokens,
                    include_thoughts=True  # So we can analyze reasoning
                )
            ),
            generate_content_config=types.GenerateContentConfig(
                max_output_tokens=budget.output_tokens,
                temperature=0.0  # Deterministic for fair comparison
            )
        )
```

```python
# agent_budget/monitor.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ToolUsageMetrics:
    """Track tool usage patterns."""
    tool_name: str
    call_count: int
    total_tokens: int

@dataclass
class AgentMetrics:
    """Comprehensive agent execution metrics."""
    strategy: str
    reasoning_tokens_used: int
    output_tokens_used: int
    total_tokens_used: int
    tool_usage: List[ToolUsageMetrics]
    duration_seconds: float

    @property
    def total_tool_calls(self) -> int:
        return sum(t.call_count for t in self.tool_usage)

    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy,
            "reasoning_tokens_used": self.reasoning_tokens_used,
            "output_tokens_used": self.output_tokens_used,
            "total_tokens_used": self.total_tokens_used,
            "total_tool_calls": self.total_tool_calls,
            "tool_usage_details": [
                {"tool": t.tool_name, "calls": t.call_count, "tokens": t.total_tokens}
                for t in self.tool_usage
            ],
            "duration_seconds": self.duration_seconds
        }

class UsageMonitor:
    """Monitor and analyze agent usage from session events."""

    def extract_metrics_from_events(
        self,
        events: list,
        strategy: str,
        duration: float
    ) -> AgentMetrics:
        """Extract usage metrics from runner events."""

        reasoning_tokens = 0
        output_tokens = 0
        tool_usage = {}

        for event in events:
            # Extract token usage
            if hasattr(event, 'usage_metadata') and event.usage_metadata:
                reasoning_tokens += getattr(event.usage_metadata, 'thinking_tokens', 0)
                output_tokens += getattr(event.usage_metadata, 'candidates_token_count', 0)

            # Track tool calls
            if hasattr(event, 'tool_name') and event.tool_name:
                if event.tool_name not in tool_usage:
                    tool_usage[event.tool_name] = {"count": 0, "tokens": 0}
                tool_usage[event.tool_name]["count"] += 1

        tool_metrics = [
            ToolUsageMetrics(
                tool_name=name,
                call_count=data["count"],
                total_tokens=data["tokens"]
            )
            for name, data in tool_usage.items()
        ]

        return AgentMetrics(
            strategy=strategy,
            reasoning_tokens_used=reasoning_tokens,
            output_tokens_used=output_tokens,
            total_tokens_used=reasoning_tokens + output_tokens,
            tool_usage=tool_metrics,
            duration_seconds=duration
        )
```

**Testing**:
- Unit tests for TokenBudget calculations
- Test agent creation with different strategies
- Verify tool integration works
- Test usage tracking from events

---

### Day 3-4: Experiment Infrastructure

**Files to Create**:
1. `experiments/tasks/research_tasks.py`
2. `experiments/tasks/fact_check_tasks.py`
3. `experiments/tasks/comparison_tasks.py`
4. `experiments/evaluator.py`
5. `experiments/run_single_agent.py`

**Task Definitions** (Requiring Tool Use):

```python
# experiments/tasks/research_tasks.py

RESEARCH_TASKS = [
    {
        "id": "research_1",
        "title": "AI Breakthroughs 2024",
        "task": (
            "What were the major AI breakthroughs in 2024? "
            "Provide a summary with specific examples and sources. "
            "Focus on breakthroughs in large language models, computer vision, "
            "and robotics."
        ),
        "expected_tool_usage": "Should search for recent AI news and developments"
    },
    {
        "id": "research_2",
        "title": "Climate Policy Updates",
        "task": (
            "What are the latest international climate policy agreements as of 2024? "
            "Include key commitments and participating countries."
        ),
        "expected_tool_usage": "Should search for current climate agreements"
    },
    {
        "id": "research_3",
        "title": "Quantum Computing Progress",
        "task": (
            "What is the current state of quantum computing in 2024? "
            "Include recent milestones, leading companies, and practical applications."
        ),
        "expected_tool_usage": "Should search for quantum computing news"
    }
]
```

```python
# experiments/tasks/fact_check_tasks.py

FACT_CHECK_TASKS = [
    {
        "id": "fact_1",
        "title": "GPT-4 Parameters",
        "task": (
            "Verify the claim: 'GPT-4 has 1 trillion parameters.' "
            "Is this accurate? Provide evidence from reliable sources."
        ),
        "expected_tool_usage": "Should search for GPT-4 specifications"
    },
    {
        "id": "fact_2",
        "title": "Mars Mission Date",
        "task": (
            "Verify: 'NASA's first crewed mission to Mars is scheduled for 2030.' "
            "Check if this is accurate and provide the actual timeline."
        ),
        "expected_tool_usage": "Should search for NASA Mars mission plans"
    }
]
```

```python
# experiments/tasks/comparison_tasks.py

COMPARISON_TASKS = [
    {
        "id": "compare_1",
        "title": "LLM Comparison",
        "task": (
            "Compare the capabilities of Claude Sonnet 4 and GPT-4. "
            "Include: model size, key features, strengths, and limitations. "
            "Use current information from 2024-2025."
        ),
        "expected_tool_usage": "Should search for both models and compare"
    },
    {
        "id": "compare_2",
        "title": "EV Comparison",
        "task": (
            "Compare Tesla Model 3 and Nissan Leaf (2024 models). "
            "Include: range, price, charging time, and key features."
        ),
        "expected_tool_usage": "Should search for both vehicle specs"
    }
]
```

**Evaluator** (Updated for Tool Usage):

```python
# experiments/evaluator.py

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

class QualityEvaluator:
    """Evaluate agent output quality using LLM judge."""

    def __init__(self):
        # Create a judge agent (no budget constraints for evaluation)
        self.judge_agent = Agent(
            model="gemini-2.0-flash",
            name="judge_agent",
            instruction="You are an expert evaluator of research quality."
        )

        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.judge_agent,
            app_name="evaluator",
            session_service=self.session_service
        )

    async def evaluate_quality(
        self,
        task: str,
        output: str,
        tool_usage: list
    ) -> dict:
        """Evaluate output quality (0-100 scale)."""

        judge_prompt = f"""
        Evaluate this research agent's response on a scale of 0-100.

        Original Task:
        {task}

        Agent Output:
        {output}

        Tool Usage:
        {self._format_tool_usage(tool_usage)}

        Evaluation Criteria:
        1. Depth of Analysis (30 points): Thorough, insightful analysis
        2. Accuracy (30 points): Factually correct, well-researched
        3. Source Quality (20 points): Uses reliable, current sources
        4. Tool Efficiency (20 points): Appropriate tool usage, not excessive

        Provide scores as JSON:
        {{
            "overall_score": <0-100>,
            "depth_score": <0-30>,
            "accuracy_score": <0-30>,
            "source_quality_score": <0-20>,
            "tool_efficiency_score": <0-20>,
            "justification": "<brief explanation>"
        }}
        """

        # Run judge agent
        session = await self.session_service.create_session(
            app_name="evaluator",
            user_id="system",
            session_id=f"eval_{hash(task)}"
        )

        from google.genai import types
        content = types.Content(
            role='user',
            parts=[types.Part(text=judge_prompt)]
        )

        response_text = ""
        async for event in self.runner.run_async(
            user_id="system",
            session_id=session.session_id,
            new_message=content
        ):
            if event.is_final_response() and event.content:
                response_text = event.content.parts[0].text

        # Parse JSON from response
        import json
        import re

        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group(1))
        else:
            scores = json.loads(response_text)

        return scores

    def _format_tool_usage(self, tool_usage: list) -> str:
        """Format tool usage for display."""
        if not tool_usage:
            return "No tools used"

        return "\n".join([
            f"- {t['tool']}: {t['calls']} call(s)"
            for t in tool_usage
        ])
```

**Experiment Runner**:

```python
# experiments/run_single_agent.py

import asyncio
import time
import pandas as pd
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent_budget.core import AllocationStrategy
from agent_budget.agent_factory import AgentFactory
from agent_budget.monitor import UsageMonitor
from experiments.evaluator import QualityEvaluator
from experiments.tasks.research_tasks import RESEARCH_TASKS
from experiments.tasks.fact_check_tasks import FACT_CHECK_TASKS
from experiments.tasks.comparison_tasks import COMPARISON_TASKS

async def run_single_agent_experiments(
    total_budget: int = 10000,
    output_dir: str = "experiments/results"
):
    """Run Part 1 experiments: Single-agent strategic allocation with tools."""

    factory = AgentFactory()
    monitor = UsageMonitor()
    evaluator = QualityEvaluator()
    session_service = InMemorySessionService()

    strategies = [
        AllocationStrategy.DEEP_THINKER,
        AllocationStrategy.BALANCED,
        AllocationStrategy.VERBOSE
    ]

    # Combine all tasks
    all_tasks = RESEARCH_TASKS + FACT_CHECK_TASKS + COMPARISON_TASKS

    results = []

    for strategy in strategies:
        # Create agent for this strategy
        agent = factory.create_agent(strategy, total_budget)
        runner = Runner(
            agent=agent,
            app_name="experiment",
            session_service=session_service
        )

        for task_data in all_tasks:
            print(f"\n{'='*60}")
            print(f"Strategy: {strategy.value}")
            print(f"Task: {task_data['title']}")
            print(f"{'='*60}")

            # Create session
            session = await session_service.create_session(
                app_name="experiment",
                user_id="researcher",
                session_id=f"{strategy.value}_{task_data['id']}"
            )

            # Execute task with timing
            start_time = time.time()

            content = types.Content(
                role='user',
                parts=[types.Part(text=task_data['task'])]
            )

            events = []
            final_output = ""

            async for event in runner.run_async(
                user_id="researcher",
                session_id=session.session_id,
                new_message=content
            ):
                events.append(event)
                if event.is_final_response() and event.content:
                    final_output = event.content.parts[0].text

            duration = time.time() - start_time

            # Extract usage metrics
            metrics = monitor.extract_metrics_from_events(
                events, strategy.value, duration
            )

            # Evaluate quality
            quality_scores = await evaluator.evaluate_quality(
                task_data['task'],
                final_output,
                metrics.to_dict()['tool_usage_details']
            )

            # Calculate efficiency
            efficiency = quality_scores['overall_score'] / (metrics.total_tokens_used / 1000)

            # Store results
            result = {
                "strategy": strategy.value,
                "task_id": task_data['id'],
                "task_title": task_data['title'],
                "task_category": task_data['id'].split('_')[0],
                **metrics.to_dict(),
                "quality_score": quality_scores['overall_score'],
                "depth_score": quality_scores['depth_score'],
                "accuracy_score": quality_scores['accuracy_score'],
                "source_quality_score": quality_scores['source_quality_score'],
                "tool_efficiency_score": quality_scores['tool_efficiency_score'],
                "efficiency": efficiency,
                "output_text": final_output[:500]  # Truncate for CSV
            }

            results.append(result)

            print(f"\n✅ Completed in {duration:.2f}s")
            print(f"   Quality: {quality_scores['overall_score']}/100")
            print(f"   Tool Calls: {metrics.total_tool_calls}")
            print(f"   Tokens: {metrics.total_tokens_used}/{total_budget}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/part1_results.csv", index=False)
    df.to_json(f"{output_dir}/part1_results.json", orient="records", indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_dir}/")
    print(f"{'='*60}")

    return df

# CLI entry point
if __name__ == "__main__":
    import os
    os.makedirs("experiments/results", exist_ok=True)
    asyncio.run(run_single_agent_experiments())
```

---

### Day 5-6: Execution & Visualization

**Run Experiments**:
```bash
python -m experiments.run_single_agent
```

**Visualization** (Jupyter Notebook):
```python
# notebooks/part1_demo.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv("experiments/results/part1_results.csv")

# 1. Pareto Frontier: Quality vs Speed
fig, ax = plt.subplots(figsize=(10, 6))

for strategy in df["strategy"].unique():
    strategy_data = df[df["strategy"] == strategy]
    ax.scatter(
        strategy_data["duration_seconds"],
        strategy_data["quality_score"],
        label=strategy.capitalize(),
        s=150,
        alpha=0.7
    )

ax.set_xlabel("Speed (seconds)", fontsize=12)
ax.set_ylabel("Quality Score (0-100)", fontsize=12)
ax.set_title("Pareto Frontier: Quality vs Speed", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiments/results/pareto_frontier.png", dpi=300)
plt.show()

# 2. Tool Usage Patterns
fig, ax = plt.subplots(figsize=(10, 6))

tool_usage = df.groupby("strategy")["total_tool_calls"].mean()
tool_usage.plot(kind="bar", ax=ax, color=["steelblue", "orange", "green"])
ax.set_ylabel("Average Tool Calls per Task", fontsize=12)
ax.set_title("Tool Usage by Strategy", fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("experiments/results/tool_usage.png", dpi=300)
plt.show()

# 3. Token Allocation Breakdown
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, strategy in enumerate(df["strategy"].unique()):
    strategy_data = df[df["strategy"] == strategy]

    avg_reasoning = strategy_data["reasoning_tokens_used"].mean()
    avg_output = strategy_data["output_tokens_used"].mean()

    axes[idx].pie(
        [avg_reasoning, avg_output],
        labels=["Reasoning", "Output"],
        autopct='%1.1f%%',
        colors=["#FF6B6B", "#4ECDC4"]
    )
    axes[idx].set_title(f"{strategy.capitalize()} Strategy", fontsize=12)

plt.tight_layout()
plt.savefig("experiments/results/token_breakdown.png", dpi=300)
plt.show()

# 4. Efficiency Comparison
fig, ax = plt.subplots(figsize=(10, 6))

efficiency_data = df.groupby("strategy")[["efficiency", "tool_efficiency_score"]].mean()
efficiency_data.plot(kind="bar", ax=ax)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Efficiency Metrics by Strategy", fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(["Overall Efficiency", "Tool Efficiency"])
plt.tight_layout()
plt.savefig("experiments/results/efficiency.png", dpi=300)
plt.show()
```

---

### Day 7: Documentation & Analysis

**Create `docs/RESULTS.md`**:
- Summary of findings
- Pareto frontier analysis
- Tool usage patterns by strategy
- Insights and recommendations
- Production deployment guidelines

**Update `README.md`**:
- Project overview
- Setup instructions
- How to run experiments
- Interpretation of results

---

## Dependencies

Update `pyproject.toml`:

```toml
[project]
name = "agent-budget-capstone"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "google-adk>=1.18.0",
    "pandas>=2.2.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "jupyter>=1.0.0",
    "pytest>=8.0.0",
    "mypy>=1.8.0",
    "ruff>=0.1.0",
]
```

## Key Metrics to Track

### 1. **Quality Metrics**
- Overall quality score (0-100)
- Depth of analysis
- Accuracy
- Source quality

### 2. **Efficiency Metrics**
- Quality per 1K tokens
- Tool efficiency score
- Response time

### 3. **Tool Usage Metrics** (NEW!)
- Number of tool calls
- Tool selection patterns
- Query quality

### 4. **Token Allocation Metrics**
- Actual reasoning tokens used
- Actual output tokens used
- Variance from budget

## Expected Results

### Hypothesis: Tool Usage Patterns

**Deep Thinker (80/20)**:
- **Fewer tool calls** but more thoughtful queries
- **Higher quality synthesis** of tool results
- **Slower** but more accurate
- **Best for**: Complex research requiring deep analysis

**Balanced (50/50)**:
- **Moderate tool usage**
- **Good quality** with reasonable speed
- **Pareto optimal** for most tasks

**Verbose (20/80)**:
- **More tool calls** (less planning = more trial-and-error)
- **Longer outputs** but potentially shallower insights
- **Faster** but lower quality
- **Best for**: Quick information gathering

### Visualization: Expected Pareto Frontier

```
Quality
  ^
  |     Deep ●
  |         / \
  |        /   \
  |       / Bal.\
  |      /   ●   \
  |     /         \
  |    /  Verbose  \
  |   /      ●      \
  +-------------------> Speed
```

## Success Metrics

**Minimum Viable**:
- ✅ Working code that runs all experiments (3 strategies × 7 tasks = 21 runs)
- ✅ CSV output with quality + tool usage metrics
- ✅ At least one visualization
- ✅ Basic documentation

**Target**:
- ✅ All of the above
- ✅ Multiple visualizations (Pareto + tool usage + efficiency)
- ✅ Statistical analysis
- ✅ Jupyter notebook demo
- ✅ Comprehensive RESULTS.md

**Stretch**:
- ✅ All of the above
- ✅ Multiple tool types (Search + Web browsing + Custom tools)
- ✅ Statistical significance testing
- ✅ Interactive visualizations
- ✅ Presentation slides

## Next Steps

1. **Initialize project structure**: Create all directories
2. **Install dependencies**: `uv add google-adk pandas matplotlib seaborn python-dotenv`
3. **Verify API access**: Test basic ADK agent with Google Search
4. **Start Day 1**: Implement core framework
5. **Daily checkpoints**: Review and test each component

## Confirmed Technical Details

1. ✅ **Model**: `gemini-2.5-flash-lite` supports thinking budget
2. ✅ **Google Search**: Available as `google_search` from `google.adk.tools`
3. ✅ **Thinking Control**: Via `ThinkingConfig(thinking_budget=<tokens>)`
4. ✅ **Output Control**: Via `GenerateContentConfig(max_output_tokens=<tokens>)`

## Questions to Address

1. **API Rate Limits**: How many requests can we make? Need retry logic?
2. **Tool Response Handling**: How to track tool outputs in metrics?
3. **Evaluation Consistency**: Multiple runs for statistical validity?
4. **Budget Enforcement**: Does ADK strictly enforce budget limits or are they soft limits?

---

**Status**: Ready to begin implementation
**Estimated Completion**: 7 days
**Key Improvement**: Now uses realistic agent + tool scenarios instead of simple LLM calls
