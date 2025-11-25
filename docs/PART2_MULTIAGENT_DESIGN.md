# Part 2: Multi-Agent Code Review Study Design

## Research Question

**Does budget awareness affect multi-agent team performance in iterative code generation tasks?**

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

### Hypotheses

**H1 (Main effect)**: Budget awareness affects team performance (pass rate)

**H2 (Direction)**: Budget awareness may help in multi-agent settings where:
- Role specialization channels metacognition productively
- Explicit allocation reduces coordination uncertainty

**H3 (Difficulty moderation)**: Any awareness effect is stronger on harder problems where budget allocation matters more

## Experimental Design

### Design: 2 × 2 Between-Subjects with Difficulty Stratification

| Factor | Levels |
|--------|--------|
| **Awareness** | NO_AWARENESS vs OVERALL_AND_INDIVIDUAL |
| **Difficulty** | Medium vs Hard |

**Why simplified from original 4 conditions?**
- Part 1 showed that prompt variations don't matter much
- Cleaner comparison: baseline vs full awareness
- Reduces sample size requirements

### Task: LiveCodeBench Code Generation

**Dataset**: LiveCodeBench (code_generation_lite, release_v6)
- Problems from February 2025+ (after model knowledge cutoff)
- Difficulty labels: medium, hard
- Objective test cases for evaluation

**Why LiveCodeBench?**
- Recent problems avoid memorization
- Clear pass/fail criteria
- Difficulty labels enable moderation analysis

### Multi-Agent Architecture

**Pattern**: `LoopAgent` with Coder ⇄ Reviewer iteration

```
Problem → Coder → Reviewer → [APPROVE or iterate] → Final Code
              ↑__________________|
```

**Agent Roles**:

| Agent | Role | Tools | Budget Share |
|-------|------|-------|--------------|
| **Coder** | Write/revise Python code | None (thinking-enabled) | 80% |
| **Reviewer** | Test code, decide approve/revise | `test_code()` | 20% |

**Why this allocation?**
- Coder needs substantial thinking for algorithm design
- Reviewer's tool execution is FREE (outside LLM loop)
- Reviewer only needs tokens to call tool and interpret result

### Budget Configuration

```python
# From core.py
CODE_REVIEW_CODER_BUDGET = TokenBudget(
    reasoning_tokens=1536,  # Thinking for algorithm design
    output_tokens=512,      # Code output
)  # 2048 total (80%)

CODE_REVIEW_REVIEWER_BUDGET = TokenBudget(
    reasoning_tokens=256,   # Minimal thinking
    output_tokens=256,      # Decision output
)  # 512 total (20%)

CODE_REVIEW_TEAM_BUDGET = 2560  # Total
```

**Maximum iterations**: 3 (mimics realistic code review)

## Two Awareness Conditions

### Design Principles (from Literature Review)

Based on our literature review on resource awareness and team performance, we apply these key principles:

1. **Challenge vs. Threat Framing**: Frame budget as "sufficient for" not "limited to" - this promotes challenge appraisal over threat response
2. **Focusing Dividend**: Constraints help prioritize what matters - emphasize this positive aspect
3. **Iteration Context**: Include temporal awareness (iteration X of Y) to support planning
4. **Actionable Guidance**: Tell agents what they CAN do, not just what they can't
5. **Avoid Threatening Language**: No 🚨 WARNING/CRITICAL emojis - these trigger threat response

### Condition 1: NO_AWARENESS (Baseline)

- **No budget information** in initial prompts
- **No status updates** between iterations
- Pure focus on task instructions only
- **Same team framing** as AWARE condition (consistent role context)

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

### Condition 2: OVERALL_AND_INDIVIDUAL (Budget Aware - Challenge Framed)

**Design Evolution**: Based on pilot study findings, we refined the budget message to use **challenge framing** without behavioral guidance. The key insight from literature: framing resources as "sufficient for" (challenge) rather than "limited to" (threat) promotes better performance.

**Two components of awareness**:
1. **Initial prompt framing** (challenge-oriented, no behavioral advice)
2. **Ongoing status updates** (factual token usage only)

**Coder prompt** (with resource context prefix):
```
[RESOURCE CONTEXT]
You have 3000 tokens per iteration - sufficient for a well-crafted solution.
This includes 1500 for reasoning and 1500 for code output.

You have up to 3 iterations to get it right.

{standard coder instructions - same as NO_AWARENESS}
```

**Reviewer prompt** (with resource context prefix):
```
[RESOURCE CONTEXT]
You have 500 tokens per iteration - sufficient for testing and clear feedback.
This includes 300 for analysis and 200 for your response.

You have up to 3 iterations.

{standard reviewer instructions - same as NO_AWARENESS}
```

**Key design decisions**:
- **No behavioral guidance**: Removed "focus on X" advice to isolate awareness effect
- **No partner info**: Agents don't know partner's budget (realistic - we rarely know colleagues' allocations)
- **Per-iteration framing**: Clear that budget is per iteration, not cumulative
- **Consistent base instructions**: Both conditions use identical task instructions

**Status updates between iterations** (factual only):
```
[STATUS: Iteration 1 of 3 complete]
Coder tokens used: 2,000
Reviewer tokens used: 150
2 iteration(s) remaining.
```

## Sample Size and Power

### Pilot Study
- **N = 40 problems** (10 per cell in 2×2 design)
- Purpose: Verify framework, estimate effect sizes

### Full Study (if pilot shows signal)
- **N = 120 problems** (30 per cell)
- Power analysis: 80% power to detect medium effect (d=0.5)

### Problem Selection
- Filter: Contest date ≥ February 2025
- Stratify: Equal medium/hard split
- Random assignment: Each problem → one condition

## Metrics

### Primary Outcome
**Pass rate**: Binary (all public tests pass = 1, else = 0)

### Secondary Outcomes
- **Iterations to success**: 1, 2, 3, or failed
- **First-attempt pass rate**: Success on iteration 1
- **Token efficiency**: Total tokens used / success

### Process Metrics
- **Coder tokens**: Per-iteration and total
- **Reviewer tokens**: Per-iteration and total
- **Token ratio**: Coder / Reviewer (should be ~4:1 given allocation)

## Analysis Plan

### Primary Analysis
1. **Chi-square test**: Pass rate by awareness condition
2. **Logistic regression**: Pass ~ Awareness + Difficulty + Awareness×Difficulty

### Secondary Analyses
1. **Iteration analysis**: Do aware agents succeed faster?
2. **Token analysis**: Do aware agents use tokens differently?
3. **Difficulty moderation**: Is awareness effect stronger for hard problems?

### Expected Outcomes

| Scenario | Finding | Interpretation |
|----------|---------|----------------|
| **A: Awareness helps** | Aware > Unaware pass rate | Role specialization framing works |
| **B: Awareness hurts** | Aware < Unaware pass rate | Metacognitive overhead (like Part 1 initial result) |
| **C: Null effect** | No difference | Budget awareness fundamentally doesn't affect LLM behavior |
| **D: Difficulty interaction** | Awareness helps on hard only | Budget matters when resources are constrained |

## Implementation Status

### Completed ✅
- [x] Core budget configuration (`core.py`)
- [x] Prompt generation with meaningful framing (`code_review_prompts.py`)
- [x] Code review runner with token tracking (`code_review_runner.py`)
- [x] Agent factory for code review teams (`agent_factory.py`)
- [x] Test script verifying both conditions work (`test_code_review_loop.py`)

### Ready for Pilot
- [ ] Experiment runner script (`run_part2_code_review.py`)
- [ ] Results collection and storage
- [ ] Analysis script

## Technical Details

### Token Tracking

Tokens are tracked from LLM response events:
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

### Test Execution

The `test_code()` function:
1. Reads code from session state (`current_code`)
2. Writes to temp file
3. Executes with problem's test input via stdin
4. Compares stdout to expected output
5. Returns PASS/FAIL with details

**Critical**: Tool execution is outside the LLM token budget - only the function call and result interpretation count.

### Loop Termination

The loop exits when:
1. Reviewer outputs "APPROVE" → Success
2. Max iterations (3) reached → Failure
3. (Rare) Unrecoverable error → Logged and skipped

## Data Collection Format

```python
@dataclass
class CodeReviewTrial:
    problem_id: str
    problem_title: str
    difficulty: str  # "medium" or "hard"
    awareness_condition: MultiAgentAwarenessCondition

    # Outcomes
    success: bool
    num_iterations: int
    final_decision: str  # "APPROVE" or "MAX_ITERATIONS_REACHED"

    # Token usage
    team_total_tokens: int
    coder_tokens: int
    reviewer_tokens: int
```

## Future Enhancement: Planning Phase with Budget Estimation

### Motivation

In real human teams, project work typically begins with a **planning phase** where:
1. A project lead evaluates the task difficulty
2. Estimates required effort (time, resources, iterations)
3. Communicates these estimates to team members

This mirrors how contracted projects work - teams know both the resource limit AND have an estimate of what's needed before starting.

### Proposed Design

Add a **planning agent** (or extend CheckApprovalAgent) that:
1. Reads the problem description before main execution
2. Estimates difficulty and expected iterations
3. Passes estimates to Coder/Reviewer via their prompts

### Implementation Approach (Low Effort)

**Planning prompt** (single LLM call):
```
Analyze this programming problem and estimate:
1. Difficulty (easy/medium/hard)
2. Expected iterations needed (1-3)
3. Brief reasoning

PROBLEM:
{problem_description}

Output JSON: {"difficulty": "...", "estimated_iterations": N, "reasoning": "..."}
```

**Enhanced budget message for AWARE condition**:
```
[RESOURCE CONTEXT]
You have 3000 tokens per iteration - sufficient for a well-crafted solution.
This includes 1500 for reasoning and 1500 for code output.

Task assessment: This appears to be a medium-difficulty problem,
likely solvable in 2 iterations.

You have up to 3 iterations to get it right.
```

### Why This Is More Realistic

- Human teams don't just know their budget - they also estimate effort
- Planning creates **shared mental models** (from literature review)
- Estimates enable agents to calibrate their approach
- Still focuses on awareness (not adaptive behavior) - keeps experiment clean

### Research Questions This Enables

1. **Estimation accuracy**: How well do agents estimate difficulty?
2. **Calibration effects**: Does knowing the estimate change behavior?
3. **Interaction**: Does accurate estimation + budget awareness improve outcomes?

### Implementation Notes

- Keep actual budgets fixed (don't change based on estimates)
- Log estimates for analysis (compare to actual difficulty labels)
- Simple addition: one LLM call + modified prompt template
- Can be tested as a follow-up study after current pilot

## References

- Part 1 findings: `docs/PART1_COMPREHENSIVE_FINDINGS.md`
- LiveCodeBench: https://livecodebench.github.io/
- Google ADK LoopAgent: https://google.github.io/adk-docs/agents/multi-agents
