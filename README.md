# Budget-Aware Agents: When Knowing Your Limits Doesn't Help

**Agents Intensive Capstone Project | Freestyle Track**

## The Problem

As AI agents are deployed at scale, managing computational costs becomes critical. A natural assumption is that agents should *know* their resource constraints so they can plan accordingly—just like humans budget time and money.

But does telling an agent about its token budget actually help?

## What We Did

We ran controlled experiments to test whether **budget awareness**—explicitly telling agents their token limits—improves performance across two settings:

### Part 1: Single-Agent Budget Awareness
- **Task**: Factual question-answering (TruthfulQA, HotpotQA)
- **Model**: Gemini 2.5 Flash Lite with extended thinking
- **Tools**: Google Search (to test strategic tool usage under constraints)
- **Scale**: 200+ matched pairs across multiple prompt variants

### Part 2: Multi-Agent Code Review Teams
- **Task**: Competitive programming (LeetCode, AtCoder via LiveCodeBench)
- **Data**: Problems from contests after Feb 2025 (post model knowledge cutoff to avoid contamination)
- **Architecture**: Coder-Reviewer iterative loop (max 3 iterations)
- **Conditions**: Consequence-aware framing, fixed vs. dynamic budgets
- **Scale**: 280 trials (140 matched problem pairs across 2 studies)

---

## Experimental Design

Both studies use a **within-subjects design**: each question/problem is tested under both conditions, eliminating confounding from task difficulty. This is critical for LLM experiments where question difficulty varies widely.

**Statistical Approach:**
- **Power analysis**: Sample sizes chosen to detect medium effects (Cohen's d ≥ 0.3) with 80% power
- **Confidence intervals**: 95% CIs computed via bootstrap resampling (10,000 iterations)
- **Paired tests**: McNemar's test for binary outcomes (success/failure)

### Part 1: Single-Agent Setup

**Agent Configuration:**
- Model: `gemini-2.5-flash-lite` with extended thinking enabled
- Tools: `google_search` (to test strategic tool usage)
- Budget: Controlled via `max_output_tokens` (thinking tokens count against this limit)

**Conditions Compared:**

| Condition | Prompt Includes |
|-----------|-----------------|
| Unaware (Control) | Task instructions only |
| Aware (Treatment) | Task instructions + budget information |

**Example Prompts:**

*Unaware condition:*
```
You are a fact-checking research assistant specializing in accurate, truthful responses.
Answer the following question with a factual, truthful response.
Begin your response with "Answer: " followed by the factual answer in 1-3 sentences.
```

*Aware condition (one variant):*
```
[BUDGET AWARENESS]
You have 512 thinking tokens and 128 output tokens (640 total).

HOW TOKENS WORK:
- Every word you think counts against your thinking budget
- Each Google Search consumes ~20-50 thinking tokens
- Plan carefully: Is this search worth the cost?

[Task instructions same as above...]
```

We tested 5 prompt variants (simple awareness, mechanistic explanation, time-based framing, positive reframing, emphatic language) - all showed null effects.

---

### Part 2: Multi-Agent Setup

**Architecture:** Coder-Reviewer loop using Google ADK's `LoopAgent`

```
┌─────────────────────────────────────────────────┐
│                  LoopAgent                       │
│  ┌─────────┐    ┌──────────┐    ┌────────────┐ │
│  │  Coder  │───▶│ Reviewer │───▶│CheckApproval│ │
│  └─────────┘    └──────────┘    └────────────┘ │
│       ▲                               │         │
│       └───────── if REVISE ───────────┘         │
│                  (max 3 iterations)             │
└─────────────────────────────────────────────────┘
```

**Agent Configurations:**

| Agent | Model | Tools | Budget |
|-------|-------|-------|--------|
| Coder | gemini-2.5-flash-lite | None (code only) | 2000-3000 tokens |
| Reviewer | gemini-2.5-flash-lite | `test_code()` | 500 tokens |

**Conditions Compared:**

*Study 1: No Awareness vs Consequence-Aware*

| Condition | Coder Prompt Prefix |
|-----------|---------------------|
| NO_AWARENESS | (none) |
| CONSEQUENCE_AWARE | `[RESOURCE CONSTRAINTS]`<br>`- 3000 tokens per iteration (output is cut off if exceeded)`<br>`- 3 iterations maximum (task fails if all used without success)` |

*Study 2: Fixed Budget vs Planner-Estimated*

| Condition | Budget Source |
|-----------|---------------|
| Fixed Budget | Difficulty-based: easy=2000, medium=3000 tokens |
| Planner-Estimated | AI planner analyzes problem, estimates tokens needed |

**Planner prompt example:**
```
Analyze the problem and estimate:
1. Tokens needed per iteration (code length)
2. Number of iterations likely needed (1-3)

Token estimates by complexity:
- Simple (basic I/O, single loop): 500-1500 tokens
- Medium (multiple functions): 1500-2500 tokens
- Complex (DP, graphs): 2500-4000 tokens
```

---

## Key Findings

### Part 1: Budget Awareness Shows No Effect in Single Agents

We tested multiple prompt engineering approaches to make agents aware of their token budgets. **None worked.**

| Prompt Variant | n pairs | Accuracy Difference | 95% CI | Result |
|----------------|---------|---------------------|--------|--------|
| Simple budget awareness | 100 | -0.5pp | [-8.0, +7.0] | NULL |
| Mechanistic explanation | 75 | -2.7pp | [-10.7, +5.3] | NULL |
| Time awareness (30-90s) | 75 | +2.7pp | [-5.3, +10.7] | NULL |
| Positive reframing | 50 | -4.0pp | [-14.0, +6.0] | NULL |
| Strong/emphatic language | 30 | -1.7pp | [-15.0, +11.7] | NULL |

All confidence intervals include zero, indicating no reliable effect.

**Key observations:**
- Aware agents use **+42 to +137 more reasoning tokens** (overthinking)
- But **81% use the exact same number of Google searches** (no strategic adaptation)
- Agents understand constraints conceptually but **cannot operationalize them**

**Lesson learned:** We initially used a between-subjects design that showed unstable, contradictory results. Switching to within-subjects eliminated confounding and revealed the true null effect.

---

### Part 2: Consequence Framing Helps, But Dynamic Planning Hurts

#### Study 1: Consequence-Aware Framing (n=140)

Telling agents about *consequences* (not just limits) significantly improved first-attempt success:

| Metric | No Awareness | Consequence-Aware | Difference | 95% CI |
|--------|--------------|-------------------|------------|--------|
| First-iteration success | 37.1% | 52.9% | **+15.7pp** | [+4.3, +27.1] |
| Overall success | 71.4% | 75.7% | +4.3pp | [-7.1, +15.7] |
| Avg tokens used | 4,440 | 3,809 | -631 | [-1,620, +370] |
| Truncation rate | 4.3% | 0.0% | -4.3pp | - |

The first-iteration CI excludes zero → **statistically significant improvement**.

**What worked:** Framing like "output is cut off if exceeded" and "task fails if all iterations used" created productive urgency.

#### Study 2: Dynamic Budget Planning (n=140)

We tested whether an AI planner that estimates per-problem budgets could outperform fixed difficulty-based budgets. **It made things worse.**

| Metric | Fixed Budget | Planner Estimated | Difference | 95% CI |
|--------|--------------|-------------------|------------|--------|
| First-iteration success | 54.3% | 41.4% | **-12.9pp** | [-24.3, -1.4] |
| Overall success | 80.0% | 68.6% | -11.4pp | [-22.9, +1.4] |
| Avg tokens used | 3,644 | 4,427 | +783 | - |

The first-iteration CI excludes zero → **statistically significant degradation**.

**The Complexity Signaling Paradox:** High planner estimates correlated with *lower* success rates:
- Low estimates (≤1500 tokens): 83% success
- High estimates (≥2500 tokens): 50% success

The budget estimate acts as a self-fulfilling prophecy—signaling "this is hard" causes over-engineering.

---

## Summary of Findings

| Setting | Intervention | Effect | Verdict |
|---------|--------------|--------|---------|
| Single-agent | Budget awareness (any framing) | No effect | ❌ Doesn't help |
| Multi-agent | Consequence-aware framing | +15.7pp first-try | ✅ Works |
| Multi-agent | AI-planned dynamic budgets | -12.9pp first-try | ❌ Hurts |

---

## Why This Matters

1. **Budget awareness intuitions are wrong**: Telling single agents their limits doesn't help them use resources more efficiently—they just overthink

2. **Consequences > Information**: Agents respond to stakes ("output will be cut off") not facts ("you have 3000 tokens")

3. **Dynamic planning can backfire**: AI-estimated budgets introduce complexity signals that hurt performance

4. **Simpler is often better**: Fixed difficulty-based budgets outperform sophisticated per-problem estimation

## Practical Recommendations

| Don't Do This | Do This Instead |
|---------------|-----------------|
| Tell single agents their exact token budget | Use API-level constraints invisibly |
| State limits neutrally ("you have X tokens") | Frame consequences ("output cut off if exceeded") |
| Have AI estimate task complexity for budgeting | Use simple fixed budgets by difficulty tier |
| Assume more information helps agents | Test whether information improves outcomes |

---

## Project Structure

```
agent-budget-capstone/
├── agent_budget/                        # Core framework
│   ├── core.py                          # Budget configurations & types
│   ├── agent_factory.py                 # Agent/team creation factory
│   ├── awareness.py                     # Budget awareness conditions (Part 1)
│   ├── code_review_runner.py            # Multi-agent trial runner
│   ├── code_review_prompts.py           # Coder/Reviewer prompt templates
│   ├── planner.py                       # Dynamic budget estimation
│   ├── loop_agents.py                   # CheckApprovalAgent for loops
│   ├── tracking_loop_agent.py           # Token-tracking LoopAgent
│   ├── usage_tracker.py                 # Usage tracking utilities
│   └── monitor.py                       # UsageMonitor & metrics
├── experiments/
│   ├── part1_single_agent/              # Part 1: Single-agent experiments
│   │   ├── run_within_subjects.py       # Main within-subjects runner
│   │   ├── run_time_awareness.py        # Time awareness variant
│   │   ├── analyze_within_subjects.py   # Statistical analysis
│   │   └── analyze_qualitative_within.py
│   ├── part2_multi_agent/               # Part 2: Multi-agent experiments
│   │   ├── run_code_review_study.py     # Coder-Reviewer study runner
│   │   └── analyze_code_review_study.py # Analysis with McNemar's test
│   ├── shared/                          # Shared evaluation utilities
│   │   ├── evaluator.py                 # LLM-as-judge evaluator
│   │   ├── evaluator_hotpotqa.py        # HotpotQA-specific evaluation
│   │   ├── evaluator_truthfulqa.py      # TruthfulQA-specific evaluation
│   │   └── validate_thinking.py         # Thinking token validation
│   ├── tasks/                           # Dataset loaders
│   │   ├── truthful_qa_tasks.py         # TruthfulQA dataset
│   │   ├── hotpotqa_tasks.py            # HotpotQA dataset
│   │   └── research_tasks.py            # Research task definitions
│   └── exploratory/                     # Exploratory analyses
│       └── explore_livecodebench.py     # LiveCodeBench exploration
├── docs/
│   ├── PART1_COMPREHENSIVE_FINDINGS.md  # Full Part 1 report
│   ├── PART2_COMPREHENSIVE_REPORT.md    # Full Part 2 report
│   ├── PART1_BUDGET_AWARENESS_DESIGN.md # Part 1 study design
│   ├── PART2_MULTIAGENT_DESIGN.md       # Part 2 study design
│   ├── LLM_JUDGE_DESIGN.md              # LLM evaluator design
│   ├── EVALUATOR_MIGRATION_NOTES.md     # Evaluator upgrade notes
│   ├── GEMINI_PROMPTING_RESEARCH.md     # Gemini prompting best practices
│   └── literature_review_resource_awareness_team_performance.md
└── tests/                               # Validation tests
    ├── test_agent_factory.py            # Factory tests
    ├── test_code_review_loop.py         # Code review loop tests
    ├── test_planner.py                  # Planner unit tests
    ├── test_planner_integration.py      # Planner integration tests
    ├── test_llm_evaluator.py            # Evaluator tests
    └── ...                              # Additional test files
```

## Running Experiments

```bash
# Setup
uv sync

# Part 1: Single-agent budget awareness
PYTHONPATH=. uv run python experiments/part1_single_agent/run_within_subjects.py
PYTHONPATH=. uv run python experiments/part1_single_agent/analyze_within_subjects.py

# Part 2: Multi-agent code review
PYTHONPATH=. uv run python experiments/part2_multi_agent/run_code_review_study.py
PYTHONPATH=. uv run python experiments/part2_multi_agent/analyze_code_review_study.py <results_file.json>
```

## References

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [TruthfulQA Dataset](https://github.com/sylinrl/TruthfulQA)
- [LiveCodeBench](https://livecodebench.github.io/)
