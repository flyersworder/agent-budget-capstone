# Part 2: Multi-Agent Budget Awareness Study - Comprehensive Report

## Executive Summary

This study investigated whether budget awareness affects multi-agent team performance in iterative code generation tasks. We tested a Coder-Reviewer team architecture on LiveCodeBench problems using within-subjects designs.

### Study 1: Consequence-Aware Framing (n=140)

**Key Finding**: Consequence-aware budget framing significantly improves first-iteration success.

| Metric | NO_AWARENESS | CONSEQUENCE-AWARE | Difference | p-value |
|--------|-------------|-------------------|------------|---------|
| First-iteration success | 37.1% | 52.9% | +15.7pp | **0.0034** |
| Overall success | 71.4% | 75.7% | +4.3pp | 0.55 |
| Avg tokens | 4,440 | 3,809 | -631 | - |
| Truncation rate | 4.3% | 0.0% | -4.3pp | - |

### Study 2: Dynamic Budget Planning (n=140)

**Key Finding**: Planner-estimated dynamic budgets **hurt** performance compared to fixed budgets.

| Metric | Fixed Budget | Planner Estimated | Difference | p-value |
|--------|-------------|-------------------|------------|---------|
| First-iteration success | 54.3% | 41.4% | -12.9pp | **0.022** |
| Overall success | 80.0% | 68.6% | -11.4pp | 0.12 |
| Avg tokens | 3,644 | 4,427 | +783 | - |

### Combined Insight

**Simple, consequence-aware fixed budgets are optimal.**
- Telling agents about consequences helps (+15.7pp)
- Adding dynamic planning hurts (-12.9pp)
- More information isn't always better

---

## Study Design

### Task
Coder-Reviewer teams solve LiveCodeBench programming problems:
- **Coder**: Writes Python code to solve the problem
- **Reviewer**: Tests code against public test cases, approves or requests revision
- **Maximum**: 3 iterations per problem

### Dataset
- 70 unique problems from LiveCodeBench (release_v6)
- Contest date ≥ February 2025 (after model knowledge cutoff)
- 31 easy + 39 medium difficulty
- Each problem tested with BOTH conditions (within-subjects)

### Conditions

**NO_AWARENESS (Baseline)**:
```
You are the CODER in a 2-agent team. Your partner is a Reviewer who will test your code.
[standard task instructions]
```

**CONSEQUENCE-AWARE**:
```
[RESOURCE CONSTRAINTS]
- 3000 tokens per iteration (output is cut off if exceeded)
- 3 iterations maximum (task fails if all used without success)

You are the CODER in a 2-agent team. Your partner is a Reviewer who will test your code.
[standard task instructions]
```

---

## Main Results

### 1. First-Iteration Success (Primary Finding)

The consequence-aware framing significantly improved first-attempt success:

| Condition | First-Iteration Success | 95% CI |
|-----------|------------------------|--------|
| NO_AWARENESS | 37.1% (26/70) | [25.7%, 48.6%] |
| CONSEQUENCE-AWARE | 52.9% (37/70) | [41.4%, 64.3%] |

**McNemar's paired test**: p = 0.0034

Paired analysis breakdown:
- 25 problems: Both succeeded on first try
- 12 problems: **Only AWARE succeeded on first try**
- 1 problem: Only UNAWARE succeeded on first try
- 32 problems: Neither succeeded on first try

### 2. Overall Success Rate

| Condition | Success Rate | 95% CI |
|-----------|-------------|--------|
| NO_AWARENESS | 71.4% (50/70) | [60.0%, 81.4%] |
| CONSEQUENCE-AWARE | 75.7% (53/70) | [65.7%, 85.7%] |

**Difference**: +4.3pp [-11.4, +18.6] - Not statistically significant

### 3. Token Efficiency

| Condition | Successful Trials | Failed Trials | Overall |
|-----------|------------------|---------------|---------|
| NO_AWARENESS | 2,962 tokens | 7,872 tokens | 4,440 tokens |
| CONSEQUENCE-AWARE | 2,389 tokens | 7,853 tokens | 3,809 tokens |

**Key insight**: Aware agents use 19% fewer tokens when successful (2,389 vs 2,962).

### 4. Iteration Distribution

| Condition | 1 iteration | 2 iterations | 3 iterations | Avg |
|-----------|-------------|--------------|--------------|-----|
| NO_AWARENESS | 26 | 15 | 29 | 2.04 |
| CONSEQUENCE-AWARE | 37 | 11 | 22 | 1.79 |

### 5. Truncation

| Condition | Truncated Trials |
|-----------|-----------------|
| NO_AWARENESS | 3/70 (4.3%) |
| CONSEQUENCE-AWARE | 0/70 (0.0%) |

---

## Qualitative Analysis

### When Awareness Helped (7 problems)

All cases where unaware failed but aware succeeded:

| Problem | Difficulty | Unaware | Aware | Token Savings |
|---------|-----------|---------|-------|---------------|
| ARC Arc | medium | 3 iter, 8995 tok, FAIL | 1 iter, 2998 tok, SUCCESS | -5,997 |
| 2^a b^2 | medium | 3 iter, 8993 tok, FAIL | 1 iter, 2536 tok, SUCCESS | -6,457 |
| Gravity | medium | 3 iter, 8994 tok, FAIL | 2 iter, 5716 tok, SUCCESS | -3,278 |
| find-minimum-cost-to-remove-array-elements | medium | 3 iter, 8994 tok, FAIL | 2 iter, 5904 tok, SUCCESS | -3,090 |
| transform-array-by-parity | easy | 3 iter, 4623 tok, FAIL | 2 iter, 2133 tok, SUCCESS | -2,490 |
| find-the-largest-almost-missing-integer | easy | 3 iter, 4815 tok, FAIL | 2 iter, 2587 tok, SUCCESS | -2,228 |
| longest-palindrome-after-substring-concatenation-i | medium | 3 iter, 6555 tok, FAIL | 3 iter, 6175 tok, SUCCESS | -380 |

**Pattern**: In 6/7 cases, the aware agent used dramatically fewer tokens AND succeeded faster. The unaware agent "spiraled" through all 3 iterations without finding the solution.

### When Awareness Hurt (4 problems)

| Problem | Difficulty | Unaware | Aware |
|---------|-----------|---------|-------|
| find-valid-pair-of-adjacent-digits-in-string | easy | 3 iter, SUCCESS | 3 iter, FAIL |
| maximum-sum-with-at-most-k-elements | medium | 2 iter, SUCCESS | 3 iter, FAIL |
| reschedule-meetings-for-maximum-free-time-i | medium | 3 iter, SUCCESS | 3 iter, FAIL |
| select-k-disjoint-special-substrings | medium | 2 iter, SUCCESS | 3 iter, FAIL |

**Pattern**: In 3/4 cases, the aware agent used MORE tokens and STILL failed. These may be cases where being more concise actually hurt (insufficient detail in the solution).

### First-Iteration Success Case Studies

**12 problems where ONLY the aware agent succeeded on first try**:

1. **assign-elements-to-groups-with-constraints** (medium)
   - Unaware: 2 iterations, eventually succeeded
   - Aware: 1 iteration, 2172 tokens - got it right immediately

2. **ARC Arc** (medium)
   - Unaware: 3 iterations, 8995 tokens, FAILED
   - Aware: 1 iteration, 2998 tokens - clean first-try solution

3. **2^a b^2** (medium)
   - Unaware: 3 iterations, 8993 tokens, FAILED
   - Aware: 1 iteration, 2536 tokens - efficient and correct

4. **Takahashi the Wall Breaker** (medium)
   - Unaware: 3 iterations, eventually succeeded
   - Aware: 1 iteration, 2531 tokens - focused approach

**Only 1 problem where UNAWARE succeeded first and AWARE didn't**:
- **Bib** (medium) - An edge case where conciseness may have hurt

---

## Interpretation

### Why Consequence Awareness Works

1. **Truncation Warning is Actionable**
   - "Output is cut off if exceeded" → Agent writes more concise code
   - Evidence: 19% fewer tokens in successful trials (2,389 vs 2,962)
   - Evidence: 0% truncation vs 4.3%

2. **Failure Framing Creates Urgency**
   - "Task fails if all iterations used" → Better first-attempt quality
   - Evidence: 53% vs 37% first-iteration success
   - Evidence: More problems solved in 1 iteration (37 vs 26)

3. **Prevents the "Spiral"**
   - Unaware agents often hit max iterations with increasing tokens
   - Aware agents either solve it quickly or recognize they can't
   - Evidence: 7 cases where aware succeeded with fewer tokens while unaware spiraled

### Why Pure Information Doesn't Work

Previous study (neutral framing) showed:
```
[RESOURCE CONTEXT]
You have 3000 tokens per iteration.
You have up to 3 iterations.
```

**Result**: No effect (-2.9pp, p=0.80)

The difference: **consequences vs information**
- "You have X tokens" → Read but not operationalized
- "Output is cut off" → Understood and acted upon

### The Mechanism

```
CONSEQUENCE AWARENESS
        ↓
    More careful first attempt
        ↓
    More concise code
        ↓
    Higher first-iteration success
        ↓
    Fewer total tokens used
        ↓
    Zero truncation
```

---

## Comparison: Two Framing Approaches

| Metric | Neutral Framing | Consequence Framing |
|--------|----------------|---------------------|
| Overall success diff | -2.9pp | +4.3pp |
| First-iter success diff | -2.9pp | **+15.7pp** |
| Token diff | +299 | **-631** |
| Truncation (aware) | 8.6% | **0.0%** |
| p-value (first-iter) | 0.73 | **0.0034** |

The consequence framing flipped the direction and achieved statistical significance.

---

## Statistical Details

### Primary Analysis: McNemar's Test for First-Iteration Success

```
Contingency Table:
                    Aware First-Iter Success    Aware First-Iter Fail
Unaware Success:              25                        1
Unaware Fail:                 12                       32

McNemar's χ² = (|12 - 1| - 1)² / (12 + 1) = 7.69
p-value = 0.0034
```

### Bootstrap Confidence Intervals

10,000 bootstrap resamples with replacement:
- First-iteration success difference: +15.7pp [-1.4, +31.4]
- Overall success difference: +4.3pp [-11.4, +18.6]
- Token difference: -631 [-1,620, +370]

---

## Limitations

1. **Single Model**: Tested only on Gemini 2.5 Flash Lite - may not generalize to other models

2. **Code Generation Task**: Results may not apply to other domains (Q&A, summarization, etc.)

3. **Artificial Consequences**: The "cut off" and "task fails" consequences are real but imposed - real-world systems may have different dynamics

4. **Within-Subjects Design**: Same problems tested twice - potential for order effects (mitigated by randomization)

5. **Sample Size**: 70 problems provides good power for large effects but may miss smaller effects

---

## Implications

### For AI Agent Design

1. **Explain Consequences, Not Just Limits**
   - Bad: "You have 3000 tokens"
   - Good: "Output is cut off if you exceed 3000 tokens"

2. **Make Truncation Warnings Explicit**
   - Agents can and will adapt output length when warned about truncation
   - This is actionable in a way that abstract limits are not

3. **Frame Iteration Limits as Stakes**
   - "Task fails if all iterations used" creates productive urgency
   - Agents prioritize first-attempt quality

### For Multi-Agent Systems

1. **Coordination Through Consequence Awareness**
   - Both agents knowing the stakes improves team performance
   - No need for explicit coordination protocols

2. **Efficiency Gains**
   - 14% fewer total tokens
   - 12% fewer iterations on average
   - Significant cost savings at scale

---

## Conclusions

This study demonstrates that **consequence-aware budget framing significantly improves first-iteration success** in multi-agent code generation tasks. The effect is:

- **Statistically significant** (p=0.0034)
- **Practically meaningful** (+15.7pp first-iteration success)
- **Efficient** (-631 tokens, -0.25 iterations on average)
- **Robust** (zero truncation in aware condition)

The key insight is that agents respond to **stakes** (what happens when limits are exceeded) rather than **information** (what the limits are). This has direct implications for prompt engineering in resource-constrained AI systems.

---

## Extension Study: Dynamic Budget Planning

### Research Question

Can a **planner agent** that dynamically estimates budget requirements per problem improve performance compared to fixed difficulty-based budgets?

### Motivation

The consequence-aware condition uses fixed budgets (2000 tokens for easy, 3000 for medium). We hypothesized that a planning stage could:
1. Better allocate resources to complex problems
2. Avoid over-allocating to simple problems
3. Provide problem-specific guidance to the coder

### Design

**OVERALL_AND_INDIVIDUAL (Fixed Budget)**:
- Easy problems: 2000 tokens/iteration
- Medium problems: 3000 tokens/iteration
- Uses consequence-aware framing

**PLANNER_ESTIMATED (Dynamic Budget)**:
- Planner agent analyzes problem first
- Estimates tokens needed per iteration (500-4000 range)
- Estimates iterations likely needed (1-3)
- Coder receives planner's estimates in prompt

**Planner Implementation**:
- Single LLM call with structured JSON output
- Same model (Gemini 2.5 Flash Lite)
- Low temperature (0.1) for consistent estimates
- XML-structured prompt with 3 few-shot examples

### Results (n=140, 70 problems × 2 conditions)

| Metric | Fixed Budget | Planner Estimated | Difference | p-value |
|--------|-------------|-------------------|------------|---------|
| Overall success | 80.0% | 68.6% | **-11.4pp** | 0.12 |
| First-iteration success | 54.3% | 41.4% | **-12.9pp** | **0.022** |
| Avg iterations | 1.73 | 2.03 | +0.30 | - |
| Avg tokens | 3,644 | 4,427 | +783 | - |

**Key Finding**: Dynamic budget planning **hurts** performance compared to fixed budgets.

### Paired Analysis

| Outcome Pattern | Count |
|-----------------|-------|
| Both succeeded | 42 |
| Both failed | 8 |
| Fixed Budget won | 14 |
| Planner won | 6 |

**McNemar's test**: p = 0.115 (overall), p = 0.022 (first-iteration)

The 14:6 ratio of discordant pairs shows Fixed Budget winning more than twice as often.

### Qualitative Insights

#### 1. Paradox of Complexity Signaling

When the planner assigns high budgets (3500 tokens), success rate drops dramatically:

| Planner Budget | Trials | Success Rate |
|----------------|--------|--------------|
| ≤1200 tokens | 16 | 86% |
| 1800-2800 | 32 | 70% |
| 3500 tokens | 21 | **48%** |

**Interpretation**: High budget signals "this is hard" to the coder, potentially inducing:
- Over-engineering
- Overthinking the solution
- Premature optimization

#### 2. Estimation Calibration is Inverted

| Trial Outcome | Mean Estimation Error |
|---------------|----------------------|
| Successful | -31.6% (under-estimated) |
| Failed | +29.7% (over-estimated) |

The planner's perception of complexity is **inversely correlated** with actual tractability. Problems it thinks are hard tend to be solved; problems it thinks are easy often fail.

#### 3. Token Limit Behavior

| Condition | Iterations Hitting Token Limit |
|-----------|-------------------------------|
| Fixed Budget | 26.4% |
| Planner Est | 35.9% |

Counter-intuitively, the planner condition hits token limits **more often**, not less.

#### 4. Case Studies: When Fixed Budget Won

| Problem | Fixed | Planner | Pattern |
|---------|-------|---------|---------|
| maximize-active-section-with-trade-i | 1 iter, SUCCESS | 3 iter, FAIL | Planner over-complicated |
| eat-pizzas | 1 iter, SUCCESS | 3 iter, FAIL | Planner over-complicated |
| Takahashi the Wall Breaker | 1 iter, SUCCESS | 3 iter, FAIL | Planner set 3500 budget |
| Gravity | 3 iter, SUCCESS | 3 iter, FAIL | Similar tokens, different outcome |

**Pattern**: When Fixed Budget won, it often solved in fewer iterations. The planner's high budget estimate seemed to give the coder "permission" to be verbose.

#### 5. Case Studies: When Planner Won

| Problem | Fixed | Planner | Pattern |
|---------|-------|---------|---------|
| assign-elements-to-groups-with-constraints | 3 iter, FAIL | 2 iter, SUCCESS | Planner: 2800 tokens, 2 iterations |
| zero-array-transformation-iv | 3 iter, FAIL | 2 iter, SUCCESS | Planner: 3500 tokens, 3 iterations |
| maximum-containers-on-a-ship | 3 iter, FAIL (truncation) | 3 iter, SUCCESS | Fixed hit truncation |

**Pattern**: Planner helped when Fixed Budget hit truncation (rare) or when its 2-iteration estimate created urgency.

### Why Dynamic Planning Hurts

#### Hypothesis A: Constraint Distraction

The planner-estimated condition adds complexity to the prompt:
```
The planner estimates you'll need approximately 3500 tokens per iteration
and 3 iterations to solve this problem.
```

This additional information may:
- Distract from the core coding task
- Create "token anxiety"
- Reduce focus on problem-solving

#### Hypothesis B: Self-Fulfilling Prophecy

When told a problem needs 3 iterations:
- Coder may not try as hard on iteration 1
- Expects to iterate, so produces "draft" code
- First-iteration success drops significantly

#### Hypothesis C: KISS Principle

Simple fixed budgets work better because:
- Less cognitive load in the prompt
- Model can focus purely on the problem
- No interpretation of planner estimates needed

### Implications

1. **Dynamic resource allocation may backfire**
   - More information isn't always better
   - Problem complexity signals can be counterproductive

2. **Fixed budgets are robust**
   - Simple difficulty-based heuristics outperform sophisticated estimation
   - 2000/3000 token split is "good enough"

3. **Planner-in-the-loop has hidden costs**
   - Extra LLM call overhead
   - Potential for miscalibration
   - Complexity signaling effects

4. **For multi-agent systems**: Sometimes the simplest coordination mechanism (fixed budgets) beats sophisticated planning.

### Files

- **Planner results**: `experiments/results/part2_code_review/study_20251125_200256.json`
- **Planner implementation**: `agent_budget/planner.py`

---

## Overall Conclusions

This study demonstrates two key findings about budget awareness in multi-agent systems:

1. **Consequence framing works**: Telling agents about consequences (+15.7pp first-iteration success, p=0.003)

2. **Dynamic planning backfires**: Planner-estimated budgets hurt performance (-12.9pp first-iteration success, p=0.022)

The combined insight: **Simple, consequence-aware fixed budgets** are the optimal approach. Adding complexity through dynamic planning provides no benefit and may actively harm performance through complexity signaling and constraint distraction.

---

## Files

- **Raw results**: `experiments/results/part2_code_review/study_20251125_161402.json`
- **Analysis script**: `experiments/part2_multi_agent/analyze_code_review_study.py`
- **Study runner**: `experiments/part2_multi_agent/run_code_review_study.py`
- **Prompts**: `agent_budget/code_review_prompts.py`
- **Design document**: `docs/PART2_MULTIAGENT_DESIGN.md`
