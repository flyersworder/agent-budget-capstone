# Agent Budget Capstone Project

**Course**: Google 5-Day AI Agents Intensive
**Project Type**: Capstone Demonstration
**Focus**: Strategic Token Allocation for Multi-Agent Systems

## Project Overview

This capstone demonstrates strategic resource allocation for AI agents under budget constraints. The key insight is that **different token types serve different purposes** (reasoning vs output), and **coordination overhead must be justified** by improved allocation efficiency.

### Core Problems Addressed

1. **Single-Agent Strategic Allocation**: How should an agent allocate tokens between reasoning and output for optimal quality-speed-cost tradeoffs?
2. **Multi-Agent Budget Coordination**: How should multiple agents share a budget pool, and when is coordination overhead worth the improved allocation?

### Research Questions

**RQ1**: What is the Pareto frontier for reasoning vs output token allocation?
**RQ2**: When does coordination overhead justify improved multi-agent allocation?
**RQ3**: What allocation strategies work best for different task types?

## Part 1: Single-Agent Strategic Allocation

### Motivation

Modern LLMs (especially Gemini Flash Thinking, GPT-o1) distinguish between:
- **Reasoning tokens**: Internal "thinking" (hidden from user)
- **Output tokens**: Final response (visible to user)

**The Strategic Question**: Given a fixed total budget (e.g., 10K tokens), how do you allocate between reasoning and output?

### Hypothesis

Different allocation strategies create a **Pareto frontier** with distinct quality-speed-cost tradeoffs:

- **Deep Thinker** (high reasoning, low output): Highest quality, slowest
- **Balanced** (equal split): Moderate quality, moderate speed
- **Verbose Explainer** (low reasoning, high output): Lower quality, fastest

**No single strategy dominates** - optimal choice depends on user constraints.

### Experimental Design

**Task**: Research paper analysis (summarize, critique, identify key contributions)

**Three Allocation Strategies**:

```python
# Strategy A: Deep Thinker
TokenBudget(
    reasoning_tokens=8000,  # 80% for deep analysis
    output_tokens=2000,     # 20% for concise summary
    total=10000
)
# Expected: Insightful, succinct, slow

# Strategy B: Balanced
TokenBudget(
    reasoning_tokens=5000,  # 50-50 split
    output_tokens=5000,
    total=10000
)
# Expected: Good quality, moderate detail, moderate speed

# Strategy C: Verbose Explainer
TokenBudget(
    reasoning_tokens=2000,  # 20% quick thinking
    output_tokens=8000,     # 80% detailed explanation
    total=10000
)
# Expected: Surface-level insights, comprehensive coverage, fast
```

### Metrics

1. **Quality**: LLM judge (Gemini 2.5 Flash) scores 0-100
   - Depth of analysis
   - Accuracy of critique
   - Insight originality

2. **Speed**: Time to completion (seconds)

3. **Token Efficiency**: Quality per 1000 tokens

4. **Coverage**: Number of key points identified

### Expected Results

**Pareto Frontier Visualization**:
```
Quality
  ^
  |     A (Deep)
  |       ●
  |      / \
  |     /   \
  |    /  B  \
  |   /   ●   \
  |  /         \
  | /     C     \
  |/      ●      \
  +---------------> Speed
```

**Insights**:
- Strategy A: Best for quality-critical tasks (research, analysis)
- Strategy C: Best for time-critical tasks (customer support, summaries)
- Strategy B: Pareto optimal for most general tasks

### Implementation Notes

**Gemini Integration**:
```python
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-2.0-flash-thinking-exp",
    contents=prompt,
    config={
        "thinking_config": {
            "thinking_budget": reasoning_tokens  # Control reasoning!
        },
        "max_output_tokens": output_tokens,
        "temperature": 0  # Deterministic for fair comparison
    }
)

# Extract usage
reasoning_used = response.usage_metadata.thinking_tokens
output_used = response.usage_metadata.candidates_token_count
```

## Part 2: Multi-Agent Budget Coordination

### Motivation

Production AI systems often involve **multiple agents** collaborating on a task (e.g., research → analyze → synthesize). These agents must share a **fixed budget pool**.

**The Coordination Paradox**: Negotiation itself consumes the resource being negotiated!

### The Core Tradeoff

**Overhead vs Benefit**:
- **No coordination**: Zero overhead, potentially unfair/suboptimal allocation
- **Smart coordination**: Costs tokens, but may achieve better allocation

**When is coordination worth it?** This is the research question!

### Experimental Design

**Task**: Research report generation pipeline

**Three Agents**:
1. **Researcher**: Gathers information from sources
2. **Analyzer**: Synthesizes findings, identifies patterns
3. **Writer**: Produces final polished report

**Total Budget**: 50,000 tokens

**Four Allocation Strategies**:

#### Strategy 1: Equal Split (No Coordination)

```python
allocations = {
    "researcher": 16,666 tokens,  # 33.3%
    "analyzer": 16,666 tokens,    # 33.3%
    "writer": 16,667 tokens       # 33.3%
}
coordination_overhead = 0 tokens
```

**Pros**: Zero overhead
**Cons**: Ignores task complexity differences

#### Strategy 2: Heuristic Pre-allocation (No Coordination)

```python
# Based on task complexity estimates
allocations = {
    "researcher": 25,000 tokens,  # 50% (complex gathering)
    "analyzer": 15,000 tokens,    # 30% (synthesis)
    "writer": 10,000 tokens       # 20% (simpler formatting)
}
coordination_overhead = 0 tokens
```

**Pros**: Zero overhead, task-aware
**Cons**: Fixed heuristic may not fit all scenarios

#### Strategy 3: Coordinator-Based (Single-Round Analysis)

```python
class BudgetCoordinator:
    def allocate(self, agents, total_budget):
        # Analyze task complexity (costs tokens)
        task_analysis = self.analyze_complexity(agents)
        # Cost: ~500 tokens

        # Allocate remaining budget proportionally
        remaining = total_budget - 500  # = 49,500
        allocations = self.proportional_allocation(
            task_analysis,
            remaining
        )

        return allocations

# Example result:
allocations = {
    "researcher": 24,750 tokens,  # 50% of 49,500
    "analyzer": 14,850 tokens,    # 30% of 49,500
    "writer": 9,900 tokens        # 20% of 49,500
}
coordination_overhead = 500 tokens (1%)
```

**Pros**: Smart allocation based on actual task analysis
**Cons**: 1% overhead

#### Strategy 4: Adaptive Coordination (Multi-Round Negotiation)

```python
class AdaptiveCoordinator:
    def allocate_with_negotiation(self, agents, total_budget):
        # Round 1: Each agent justifies request (200 tokens each)
        requests = {
            agent: agent.justify_budget_request(task)
            for agent in agents
        }
        # Cost: 600 tokens

        # Round 2: Coordinator analyzes and proposes (300 tokens)
        initial_allocation = self.propose_allocation(requests)
        # Cost: 300 tokens

        # Round 3: Agents counter-propose if dissatisfied (100 tokens each)
        counter_proposals = [
            agent.counter_propose(allocation)
            for agent in dissatisfied_agents
        ]
        # Cost: up to 300 tokens

        # Final allocation
        total_overhead = 600 + 300 + 300 = 1,200 tokens
        remaining = total_budget - 1,200  # = 48,800

        return allocations

coordination_overhead = 1,200 tokens (2.4%)
```

**Pros**: Most sophisticated, accounts for agent preferences
**Cons**: 2.4% overhead, may have diminishing returns

### Metrics

1. **Final Quality**: End-to-end pipeline quality score (0-100)
2. **Coordination Overhead**: Tokens spent on allocation vs execution
3. **Agent Satisfaction**: Did each agent have sufficient budget?
4. **Allocation Efficiency**: Quality gained per coordination token spent

### Expected Results

**Hypothesis**: Coordination has diminishing returns

| Strategy | Overhead | Quality | Efficiency |
|----------|----------|---------|------------|
| Equal Split | 0 (0%) | 75 | ∞ (baseline) |
| Heuristic | 0 (0%) | 82 | ∞ |
| Coordinator | 500 (1%) | 88 | +6 quality / 500 tokens = **1.2%** |
| Adaptive | 1,200 (2.4%) | 89 | +7 quality / 1,200 tokens = **0.58%** |

**Key Insight**: Simple heuristics often near-optimal! Expensive coordination shows diminishing returns.

### The Break-Even Analysis

**When is coordination worth it?**

- **Small budget** (5K tokens): 500 overhead = 10% waste → NOT worth it
- **Medium budget** (50K tokens): 500 overhead = 1% waste → Worth it if quality gain > 1%
- **Large budget** (500K tokens): 500 overhead = 0.1% waste → Almost always worth it

**Visualization**:
```
Coordination
Benefit (%)
    ^
    |   Worth it (benefit > overhead)
  8%| ●●●●●●●●●●●●●●●●●●●●●●●●●●●●
    |        ///////////////////
  6%|       ///////////////////
    |      ///  Coordinator  ///
  4%|     ///////////////////
    |    ●●●●●●●●●●●●●●●●●●●
  2%|   Not worth it (overhead > benefit)
    |  ●●●●  Adaptive   ●●●●
  0%|●●●●●●●●●●●●●●●●●●●●●●●●●●●●
    +----------------------------->
      10K   50K  100K  500K  Budget
```

### Implementation Notes

**Zero-Overhead Coordinator** (heuristic):
```python
def heuristic_allocation(agents, total_budget):
    """No LLM calls - pure heuristic."""
    complexity_scores = {
        "researcher": 5,  # High complexity
        "analyzer": 3,    # Medium complexity
        "writer": 2       # Low complexity
    }
    total_complexity = sum(complexity_scores.values())

    allocations = {
        agent: (score / total_complexity) * total_budget
        for agent, score in complexity_scores.items()
    }
    return allocations, 0  # Zero overhead
```

**Smart Coordinator** (LLM-based):
```python
def smart_allocation(agents, total_budget, overhead_budget=500):
    """Uses LLM to analyze task complexity."""

    # Build analysis prompt
    analysis_prompt = f"""
    Analyze these agent tasks and recommend budget allocation:

    Total budget: {total_budget} tokens

    Agents:
    {format_agent_tasks(agents)}

    Provide allocation as JSON with reasoning.
    """

    # Use LLM to analyze (costs ~500 tokens)
    response = llm.generate(
        analysis_prompt,
        max_tokens=overhead_budget
    )

    allocation = parse_allocation(response)
    actual_overhead = response.usage.total_tokens

    return allocation, actual_overhead
```

## Technical Implementation

### Core Framework (~400 lines)

**File**: `agent_budget/core.py`

```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

@dataclass
class TokenBudget:
    """Strategic token budget allocation."""
    reasoning_tokens: int  # For thinking/analysis
    output_tokens: int     # For response generation

    @property
    def total(self) -> int:
        return self.reasoning_tokens + self.output_tokens

    def validate(self) -> bool:
        """Ensure non-negative budgets."""
        return self.reasoning_tokens >= 0 and self.output_tokens >= 0

class AllocationStrategy(Enum):
    """Single-agent allocation strategies."""
    DEEP_THINKER = "deep"      # High reasoning, low output (80/20)
    BALANCED = "balanced"       # Equal split (50/50)
    VERBOSE = "verbose"         # Low reasoning, high output (20/80)

    def create_budget(self, total_tokens: int) -> TokenBudget:
        """Create budget for this strategy."""
        ratios = {
            "deep": (0.8, 0.2),
            "balanced": (0.5, 0.5),
            "verbose": (0.2, 0.8)
        }
        reasoning_ratio, output_ratio = ratios[self.value]
        return TokenBudget(
            reasoning_tokens=int(total_tokens * reasoning_ratio),
            output_tokens=int(total_tokens * output_ratio)
        )

class BudgetMonitor:
    """Track token usage for a single agent."""
    def __init__(self, budget: TokenBudget):
        self.budget = budget
        self.reasoning_used = 0
        self.output_used = 0

    def add_usage(self, reasoning: int, output: int):
        """Record token usage."""
        self.reasoning_used += reasoning
        self.output_used += output

    def is_exceeded(self) -> bool:
        """Check if budget exceeded."""
        return (self.reasoning_used > self.budget.reasoning_tokens or
                self.output_used > self.budget.output_tokens)

    def remaining(self) -> TokenBudget:
        """Calculate remaining budget."""
        return TokenBudget(
            reasoning_tokens=max(0, self.budget.reasoning_tokens - self.reasoning_used),
            output_tokens=max(0, self.budget.output_tokens - self.output_used)
        )

class BudgetCoordinator:
    """Allocate budget across multiple agents."""

    def allocate_equal(self, num_agents: int, total_budget: int) -> Dict[str, int]:
        """Equal split (zero overhead)."""
        per_agent = total_budget // num_agents
        return {f"agent_{i}": per_agent for i in range(num_agents)}

    def allocate_heuristic(
        self,
        agent_complexities: Dict[str, int],
        total_budget: int
    ) -> Dict[str, int]:
        """Heuristic-based allocation (zero overhead)."""
        total_complexity = sum(agent_complexities.values())
        return {
            agent: int((complexity / total_complexity) * total_budget)
            for agent, complexity in agent_complexities.items()
        }

    def allocate_smart(
        self,
        agents: List[str],
        agent_tasks: Dict[str, str],
        total_budget: int,
        llm_client,
        overhead_budget: int = 500
    ) -> tuple[Dict[str, int], int]:
        """LLM-based smart allocation (costs tokens)."""
        # Build analysis prompt
        prompt = self._build_analysis_prompt(agents, agent_tasks, total_budget)

        # Use LLM to analyze (costs tokens!)
        response = llm_client.generate(prompt, max_tokens=overhead_budget)

        # Parse allocation
        allocation = self._parse_allocation(response.text)
        overhead = response.usage.total_tokens

        # Adjust allocations to account for overhead
        remaining = total_budget - overhead
        adjusted = {
            agent: int((alloc / total_budget) * remaining)
            for agent, alloc in allocation.items()
        }

        return adjusted, overhead

    def _build_analysis_prompt(
        self,
        agents: List[str],
        agent_tasks: Dict[str, str],
        total_budget: int
    ) -> str:
        """Build prompt for LLM-based allocation."""
        return f"""
        Analyze these agent tasks and recommend optimal budget allocation.

        Total budget: {total_budget} tokens

        Agent tasks:
        {chr(10).join(f"- {agent}: {task}" for agent, task in agent_tasks.items())}

        Consider:
        1. Task complexity (more complex = more tokens)
        2. Information gathering needs (research heavy = more tokens)
        3. Output requirements (detailed output = more tokens)

        Provide allocation as percentages that sum to 100%.
        Format: {{"agent_name": percentage, ...}}
        """

    def _parse_allocation(self, response_text: str) -> Dict[str, int]:
        """Parse LLM response to extract allocation."""
        # Implementation: parse JSON from response
        import json
        # Extract JSON from markdown code blocks if present
        if "```" in response_text:
            json_str = response_text.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        else:
            json_str = response_text

        return json.loads(json_str.strip())

class ContractedAgent:
    """Agent wrapper that enforces token budget."""

    def __init__(self, agent, budget: TokenBudget, llm_client):
        self.agent = agent
        self.budget = budget
        self.monitor = BudgetMonitor(budget)
        self.llm_client = llm_client

    def execute(self, task: str) -> dict:
        """Execute task with budget enforcement."""
        # Check budget before execution
        if self.monitor.is_exceeded():
            raise BudgetExceededError(
                f"Budget exceeded: {self.monitor.reasoning_used}/{self.budget.reasoning_tokens} "
                f"reasoning, {self.monitor.output_used}/{self.budget.output_tokens} output"
            )

        # Execute with budget constraints
        remaining = self.monitor.remaining()
        response = self.llm_client.generate(
            task,
            thinking_budget=remaining.reasoning_tokens,
            max_output_tokens=remaining.output_tokens
        )

        # Track usage
        self.monitor.add_usage(
            reasoning=response.usage.thinking_tokens,
            output=response.usage.output_tokens
        )

        return {
            "output": response.text,
            "usage": {
                "reasoning": response.usage.thinking_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.total_tokens
            },
            "remaining": self.monitor.remaining()
        }

class BudgetExceededError(Exception):
    """Raised when agent exceeds budget."""
    pass
```

### Experiment Runner

**File**: `experiments/run_experiments.py`

```python
from agent_budget.core import (
    AllocationStrategy,
    BudgetCoordinator,
    ContractedAgent
)
import pandas as pd
from google import genai

def run_single_agent_experiments(task: str, total_budget: int = 10000):
    """Part 1: Single-agent strategic allocation."""

    strategies = [
        AllocationStrategy.DEEP_THINKER,
        AllocationStrategy.BALANCED,
        AllocationStrategy.VERBOSE
    ]

    results = []

    for strategy in strategies:
        budget = strategy.create_budget(total_budget)
        agent = ContractedAgent(agent=None, budget=budget, llm_client=genai_client)

        # Execute task
        import time
        start = time.time()
        result = agent.execute(task)
        duration = time.time() - start

        # Evaluate quality
        quality = evaluate_quality(result["output"], task)

        results.append({
            "strategy": strategy.value,
            "reasoning_budget": budget.reasoning_tokens,
            "output_budget": budget.output_tokens,
            "quality": quality,
            "duration": duration,
            "tokens_used": result["usage"]["total"],
            "efficiency": quality / (result["usage"]["total"] / 1000)
        })

    return pd.DataFrame(results)

def run_multi_agent_experiments(
    agent_tasks: Dict[str, str],
    total_budget: int = 50000
):
    """Part 2: Multi-agent budget coordination."""

    coordinator = BudgetCoordinator()

    strategies = [
        ("equal", lambda: coordinator.allocate_equal(len(agent_tasks), total_budget)),
        ("heuristic", lambda: coordinator.allocate_heuristic(
            {"researcher": 5, "analyzer": 3, "writer": 2},
            total_budget
        )),
        ("smart", lambda: coordinator.allocate_smart(
            list(agent_tasks.keys()),
            agent_tasks,
            total_budget,
            genai_client,
            overhead_budget=500
        ))
    ]

    results = []

    for strategy_name, allocate_fn in strategies:
        if strategy_name == "smart":
            allocation, overhead = allocate_fn()
        else:
            allocation = allocate_fn()
            overhead = 0

        # Execute pipeline with allocation
        pipeline_result = execute_pipeline(agent_tasks, allocation)

        results.append({
            "strategy": strategy_name,
            "overhead": overhead,
            "overhead_pct": (overhead / total_budget) * 100,
            "quality": pipeline_result["quality"],
            "allocation": allocation
        })

    return pd.DataFrame(results)

def evaluate_quality(output: str, task: str) -> float:
    """Evaluate output quality using LLM judge."""
    judge_prompt = f"""
    Evaluate this output on a scale of 0-100.

    Task: {task}

    Output: {output}

    Criteria:
    - Depth of analysis (40 points)
    - Accuracy and correctness (30 points)
    - Insight and originality (30 points)

    Provide a single number score (0-100).
    """

    response = genai_client.generate(judge_prompt, max_tokens=50)
    score = extract_score(response.text)
    return score
```

## Deliverables

### 1. Code Implementation (~400 lines)
- `agent_budget/core.py`: Core framework
- `agent_budget/gemini_client.py`: Gemini integration
- `experiments/run_experiments.py`: Experiment runner
- `experiments/visualize_results.py`: Result visualization

### 2. Jupyter Notebook Demo
- Part 1: Single-agent Pareto frontier (3 strategies × 3 tasks = 9 experiments)
- Part 2: Multi-agent coordination (4 strategies × 1 pipeline = 4 experiments)
- Visualizations: Quality-speed plots, overhead analysis, allocation comparisons

### 3. Documentation
- `README.md`: Project overview and setup
- `RESULTS.md`: Experiment findings and insights
- `DEPLOYMENT_GUIDE.md`: Production deployment guidelines

### 4. Presentation Materials
- Slides with key visualizations
- Live demo (if time permits)
- Code walkthrough

## Timeline Considerations

**If you have 1 week**:
- Focus on Part 1 only (simpler, cleaner)
- 3 strategies × 2 tasks = 6 experiments
- Solid Pareto frontier demonstration

**If you have 2 weeks**:
- Implement both Part 1 and Part 2
- Full experiment suite
- Comprehensive analysis

**If you have 3+ weeks**:
- Add adaptive coordination (Strategy 4)
- Multiple task types
- Statistical validation (multiple runs)
- Polished presentation

## Success Metrics

**Minimum viable capstone**:
- ✅ Clear problem statement
- ✅ Working code demonstration
- ✅ Empirical results showing tradeoffs
- ✅ Production deployment guide

**Impressive capstone**:
- ✅ Novel insights (coordination overhead analysis)
- ✅ Pareto frontier visualization
- ✅ Multiple task types validated
- ✅ Statistical rigor (multiple runs, confidence intervals)

**Outstanding capstone**:
- ✅ All of the above
- ✅ Break-even analysis for coordination
- ✅ Generalizable framework (works beyond demo tasks)
- ✅ Production-ready code quality

## Connection to Full Framework

**What you reveal**:
- Core concept: Strategic token allocation
- Budget enforcement mechanism
- Multi-agent coordination patterns

**What you keep proprietary**:
- Full contract theory (I, O, S, R, T, Φ, Ψ)
- Budget-aware prompting
- Quality frameworks and evaluation
- Multiple integrations (LangChain, LangGraph)
- All the validation research from agent-contracts repo

**Closing statement for presentation**:
> "This capstone demonstrates strategic token allocation for production AI agents. The key insights are:
> 1. Different allocation strategies create a Pareto frontier - no single strategy dominates
> 2. Coordination overhead has diminishing returns - simple heuristics often near-optimal
>
> This is part of a larger framework I'm developing that extends these concepts to temporal constraints, quality-cost-time tradeoffs, and integration with frameworks like LangChain and LangGraph for complex production workflows."

## Next Steps

1. **Choose scope**: Part 1 only vs Part 1 + Part 2
2. **Select demo tasks**: What tasks best demonstrate the tradeoffs?
3. **Set timeline**: When is the capstone due?
4. **Initialize repo**: Set up project structure
5. **Start implementation**: Begin with core framework

---

*Document created: November 16, 2025*
*Status: Planning phase*
*Repository: To be initialized*
