# Part 2: Multi-Agent Code Review Study Design

## Research Question

**Does budget awareness affect multi-agent team performance in iterative code generation tasks?**

## Executive Summary

**Key Finding**: Consequence-aware budget framing significantly improves first-iteration success rate in multi-agent code review tasks.

- First-iteration success: 52.9% (aware) vs 37.1% (unaware), **p=0.0034**
- Overall success: 75.7% vs 71.4% (+4.3pp, not significant)
- Token efficiency: Aware agents use 631 fewer tokens on average
- Zero truncation in aware condition vs 4.3% in unaware

The critical insight: Simply telling agents about budgets doesn't help. Telling them about **consequences** (truncation, task failure) does.

## Background

### Part 1 Findings (Single-Agent)

Part 1 tested budget awareness in single-agent question-answering:
- **Result**: No significant effect found (null result)
- Multiple prompt engineering approaches failed to show improvement
- Key insight: Agents understand constraints conceptually but cannot operationalize them strategically

### Why Multi-Agent Code Review?

We pivot from Q&A to code review because:

1. **Objective evaluation**: Code either passes tests or doesn't (no LLM-as-judge needed)
2. **Natural role separation**: Coder (generate) vs Reviewer (verify) have distinct functions
3. **Iterative refinement**: Multiple rounds allow budget to matter across iterations
4. **Tool-agent asymmetry**: Reviewer's tool execution is FREE (outside LLM) - creates meaningful budget allocation question

## Experimental Design

### Design: Within-Subjects with Difficulty Stratification

| Factor | Levels |
|--------|--------|
| **Awareness** | NO_AWARENESS vs OVERALL_AND_INDIVIDUAL |
| **Difficulty** | Easy vs Medium |

**Within-subjects design**: Each problem tested with BOTH conditions (140 total trials from 70 problems)

### Task: LiveCodeBench Code Generation

**Dataset**: LiveCodeBench (code_generation_lite, release_v6)
- Problems from February 2025+ (after model knowledge cutoff)
- 31 easy + 39 medium = 70 unique problems
- Objective test cases for evaluation

### Multi-Agent Architecture

**Pattern**: `LoopAgent` with Coder ⇄ Reviewer iteration

```
Problem → Coder → Reviewer → [APPROVE or iterate] → Final Code
              ↑__________________|
```

**Agent Roles**:

| Agent | Role | Tools | Budget |
|-------|------|-------|--------|
| **Coder** | Write/revise Python code | None (thinking-enabled) | 3000 tokens (medium) / 2000 (easy) |
| **Reviewer** | Test code, decide approve/revise | `test_code()` | 512 tokens |

**Maximum iterations**: 3

## Evolution of Awareness Manipulation

### Phase 1: Neutral Information (No Effect)

Initial approach - just provide budget information:
```
[RESOURCE CONTEXT]
You have 3000 tokens per iteration.
You have up to 3 iterations.
```

**Result**: No significant effect (-2.9pp, p=0.80)

### Phase 2: Consequence-Aware Framing (Significant Effect)

Refined approach - explain what happens when limits are hit:
```
[RESOURCE CONSTRAINTS]
- 3000 tokens per iteration (output is cut off if exceeded)
- 3 iterations maximum (task fails if all used without success)
```

**Result**: Significant improvement in first-iteration success (+15.7pp, p=0.0034)

### Key Insight

The difference is **consequence awareness**, not just resource awareness:
- "You have 3000 tokens" → Agent reads but doesn't operationalize
- "Output is cut off if exceeded" → Agent understands the stakes

## Study Results

### Primary Outcome: Overall Success Rate

| Condition | Success Rate | 95% CI |
|-----------|-------------|--------|
| NO_AWARENESS | 71.4% | [60.0%, 81.4%] |
| OVERALL_AND_INDIVIDUAL | 75.7% | [65.7%, 85.7%] |

**Difference**: +4.3pp [-11.4, +18.6] - Not statistically significant

### Key Finding: First-Iteration Success

| Condition | First-Iteration Success | 95% CI |
|-----------|------------------------|--------|
| NO_AWARENESS | 37.1% | [25.7%, 48.6%] |
| OVERALL_AND_INDIVIDUAL | 52.9% | [41.4%, 64.3%] |

**Difference**: +15.7pp [-1.4, +31.4]
**McNemar's test p-value**: 0.0034 (**Statistically Significant**)

Paired analysis:
- 25 problems: Both succeeded on first try
- 1 problem: Only unaware succeeded first try
- **12 problems: Only aware succeeded first try**

### Token Efficiency

| Condition | Avg Total Tokens | Avg Coder Tokens |
|-----------|-----------------|------------------|
| NO_AWARENESS | 4,440 | 4,365 |
| OVERALL_AND_INDIVIDUAL | 3,809 | 3,716 |

**Difference**: -631 tokens (aware uses 14% fewer tokens)

### Iteration Analysis

| Condition | Avg Iterations | Iteration Distribution |
|-----------|---------------|----------------------|
| NO_AWARENESS | 2.04 | 1=26, 2=15, 3=29 |
| OVERALL_AND_INDIVIDUAL | 1.79 | 1=37, 2=11, 3=22 |

### Truncation Analysis

| Condition | Truncation Rate |
|-----------|----------------|
| NO_AWARENESS | 4.3% (3/70) |
| OVERALL_AND_INDIVIDUAL | 0.0% (0/70) |

### By Difficulty

| Difficulty | NO_AWARENESS | AWARE | Effect |
|------------|-------------|-------|--------|
| Easy | 87% | 90% | +3.2pp |
| Medium | 59% | 64% | +5.1pp |

## Interpretation

### Why Consequence Awareness Works

1. **Truncation warning is actionable**: "Output is cut off" tells agents their code might be incomplete - they respond by being more concise

2. **Failure framing creates urgency**: "Task fails if all iterations used" encourages getting it right the first time

3. **No behavioral guidance needed**: Just stating consequences lets agents figure out the appropriate response

### What Doesn't Work

1. **Pure information**: "You have X tokens" - agents read but don't adapt
2. **Challenge framing**: "Sufficient for a well-crafted solution" - no better than neutral
3. **Reasoning/output split**: Explaining token allocation mechanism doesn't help

### The Mechanism

Consequence-aware agents show:
- **Higher first-attempt success** (53% vs 37%)
- **Fewer total tokens** (3,809 vs 4,440)
- **Zero truncation** (vs 4.3%)
- **Fewer iterations needed** (1.79 vs 2.04)

This suggests agents:
1. Think more carefully before generating code
2. Write more concise solutions
3. Avoid the "spiral" of repeated failed attempts

## Prompts Used

### Condition 1: NO_AWARENESS (Baseline)

**Coder prompt**:
```
You are the CODER in a 2-agent team. Your partner is a Reviewer who will test your code.

PROBLEM:
{problem_description}

REQUIREMENTS:
1. Write a complete Python program that reads from stdin and writes to stdout
2. If you see feedback from the Reviewer, fix the issues they identified
3. Output ONLY the Python code - no explanations or markdown

Think through the algorithm carefully, then write clean, correct code.
```

**Reviewer prompt**:
```
You are the REVIEWER in a 2-agent team. Your partner is a Coder who writes the code.

WORKFLOW:
1. Call test_code() to run the Coder's code against the test case
2. Based on the result, make your decision

OUTPUT FORMAT:
DECISION: APPROVE (if tests pass) or REQUEST_REVISION (if tests fail)
FEEDBACK: [Brief explanation of what happened]

Be concise - the test result tells you everything you need to know.
```

### Condition 2: OVERALL_AND_INDIVIDUAL (Consequence-Aware)

**Coder prompt** (with constraint prefix):
```
[RESOURCE CONSTRAINTS]
- 3000 tokens per iteration (output is cut off if exceeded)
- 3 iterations maximum (task fails if all used without success)

{standard coder instructions - same as NO_AWARENESS}
```

**Reviewer prompt** (with constraint prefix):
```
[RESOURCE CONSTRAINTS]
- 512 tokens per iteration (output is cut off if exceeded)
- 3 iterations maximum (task fails if all used without success)

{standard reviewer instructions - same as NO_AWARENESS}
```

## Technical Implementation

### Budget Configuration

```python
# Per-difficulty coder budgets
CODE_REVIEW_CODER_BUDGETS = {
    "easy": TokenBudget(reasoning_tokens=1000, output_tokens=1000),  # 2000 total
    "medium": TokenBudget(reasoning_tokens=1500, output_tokens=1500),  # 3000 total
}

# Reviewer budget (same for all difficulties)
CODE_REVIEW_REVIEWER_BUDGET = TokenBudget(
    reasoning_tokens=256,
    output_tokens=256,
)  # 512 total
```

### Token Tracking

```python
async for event in runner.run_async(...):
    if hasattr(event, "usage_metadata") and event.usage_metadata:
        thinking = getattr(event.usage_metadata, "thoughts_token_count", 0) or 0
        output = getattr(event.usage_metadata, "candidates_token_count", 0) or 0

        if event.author == "Coder":
            coder_tokens += thinking + output
        elif event.author == "Reviewer":
            reviewer_tokens += thinking + output
```

## Conclusions

### Main Findings

1. **Consequence awareness significantly improves first-iteration success** (p=0.0034)
2. **Pure budget information has no effect** - agents need to understand stakes
3. **Aware agents are more efficient** - fewer tokens, fewer iterations, zero truncation
4. **Effect is consistent across difficulty levels** but stronger on medium problems

### Implications for AI Agent Design

1. **Don't just inform - explain consequences**: Agents respond to stakes, not just numbers
2. **Truncation warnings are actionable**: Agents can adapt their output length
3. **Iteration limits create urgency**: First-attempt quality improves when failure has consequences

### Limitations

1. Single model (Gemini 2.5 Flash Lite) - may not generalize
2. Code generation task - may not apply to other domains
3. Artificial consequences - real systems may have different dynamics

## Files

- **Study runner**: `experiments/part2_multi_agent/run_code_review_study.py`
- **Analysis script**: `experiments/part2_multi_agent/analyze_code_review_study.py`
- **Results**: `experiments/results/part2_code_review/`
- **Prompts**: `agent_budget/code_review_prompts.py`
- **Comprehensive report**: `docs/PART2_COMPREHENSIVE_REPORT.md`

## References

- Part 1 findings: `docs/PART1_COMPREHENSIVE_FINDINGS.md`
- LiveCodeBench: https://livecodebench.github.io/
- Google ADK LoopAgent: https://google.github.io/adk-docs/agents/multi-agents
