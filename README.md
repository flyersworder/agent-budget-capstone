# Agent Token Budget Research

Research project investigating how AI agents respond to computational budget constraints, specifically focusing on budget awareness and multi-agent coordination.

## Research Questions

### Part 1: Budget Awareness (COMPLETED ✅)
**Does explicit budget awareness improve agent performance?**

**Answer**: **NO** - Budget awareness significantly **hurts** performance (-12.9pp accuracy, p=0.028)

Key findings:
- Aware agents use 47% MORE reasoning tokens but achieve 12.9pp LOWER accuracy
- Effect is statistically significant and replicates across independent trials
- Evidence of "budget awareness overhead" - meta-cognitive load reduces task performance

See `PART1_COMPREHENSIVE_REPORT.md` for full results.

### Part 2: Multi-Agent Coordination (IN PROGRESS 🚧)
**Can multi-agent systems with coordination overcome the budget awareness paradox?**

Planned conditions:
- (a) Multi-agent without budget awareness (baseline)
- (b) Multi-agent with overall budget awareness only
- (c) Multi-agent with overall + individual budget awareness
- (d) Multi-agent with overall budget awareness + negotiation

## Key Technical Discovery

**Agents are NOT aware of their configured budgets by default**:

#### Extended Thinking Budget (`thinking_budget`)
- **Configuration**: Set via `ThinkingConfig` in `BuiltInPlanner`
- **Agent Awareness**: ❌ **None** - The model cannot access this value
- **Function**: **Soft API constraint** that guides generation (can be exceeded)
- **Minimum Value**: **512 tokens** (API constraint for Gemini models)

#### Output Token Budget (`max_output_tokens`)
- **Configuration**: Set via `GenerateContentConfig`
- **Agent Awareness**: ❌ **None** - The model cannot access this value
- **Function**: **Hard API constraint** that truncates output generation
- **CRITICAL**: When `include_thoughts=True`, thinking tokens count against this limit!

**How Budgets Actually Work**:

| Aspect | API-Level Constraints | Agent-Level Knowledge |
|--------|----------------------|----------------------|
| **Visibility** | Hidden from model | Must be explicitly communicated |
| **Influence** | Affects generation process | Affects reasoning strategy |
| **Analogy** | Like temperature or top_p | Like system prompts |

## Project Structure

```
agent-budget-capstone/
├── agent_budget/              # Core implementation
│   ├── core.py                # Token budget definitions
│   ├── agent_factory.py       # Agent creation (old Part 1 - archived)
│   ├── awareness.py           # Budget awareness conditions (Part 1)
│   └── monitor.py             # Usage tracking
├── experiments/               # Experiment runners
│   ├── run_part1_full.py      # Part 1: Budget awareness study
│   ├── analyze_part1_full.py  # Per-run statistical analysis
│   ├── analyze_part1_combined.py  # Pooled analysis (n=200)
│   ├── evaluator_objective.py # LLM-as-judge for correctness
│   └── tasks/
│       └── truthful_qa_tasks.py  # TruthfulQA dataset integration
├── docs/                      # Study designs
│   └── PART1_BUDGET_AWARENESS_DESIGN.md
├── archive/                   # Archived studies
│   └── old_part1_allocation_strategies/  # Original allocation study
└── PART1_COMPREHENSIVE_REPORT.md  # Full research report
```

## Part 1 Study Design

**Type**: Between-subjects factorial design (n=200)

**Factors**:
- Budget Awareness: 2 levels (Unaware, Aware)
- Budget Size: 3 levels (Tight: 640, Moderate: 1280, Comfortable: 2560)

**Evaluation**:
- Dataset: TruthfulQA (factual question-answering)
- Metric: Objective correctness via LLM-as-judge
- Model: Gemini 2.5 Flash Lite with extended thinking

## Running Experiments

### Part 1: Budget Awareness Study
```bash
# Run full study (100 questions, between-subjects design)
uv run python -m experiments.run_part1_full

# Analyze results (per-run analysis with bootstrap CIs)
uv run python -m experiments.analyze_part1_full

# Pooled analysis across both runs (n=200)
uv run python -m experiments.analyze_part1_combined
```

### Part 2: Multi-Agent Study (Coming Soon)
Design in progress...

## Dependencies

```bash
# Core
uv add google-adk python-dotenv

# Analysis
uv add --group analysis scipy numpy pandas

# Development
uv sync --group dev  # pre-commit hooks
```

## Key Files

- `PART1_COMPREHENSIVE_REPORT.md` - Complete Part 1 research report
- `docs/PART1_BUDGET_AWARENESS_DESIGN.md` - Study design and hypotheses
- `agent_budget/awareness.py` - Budget awareness implementation
- `experiments/results/` - Raw experimental data

## Citation

If you use this research, please cite:

```
Budget Awareness Paradox in LLM Agents (2025)
Experimental evidence that explicit budget constraints reduce agent performance
Gemini 2.5 Flash Lite with Extended Thinking on TruthfulQA
```
