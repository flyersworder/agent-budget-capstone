# Part 2: Budget Awareness Study Design

## Research Question

**Does explicit budget awareness improve agent performance when controlling for total token budget?**

## Motivation

From Part 1, we discovered that:
1. **Agents are NOT aware of their configured budgets** (thinking_budget, max_output_tokens)
2. These parameters are API-level constraints, not communicated to the model
3. Agents cannot optimize their resource allocation without explicit instructions

**Key Insight:** If we tell agents about their budgets, they might use resources more strategically.

## Hypotheses

### H1: Budget Awareness Improves Accuracy
**Prediction:** Budget-aware agents will score higher on objective correctness metrics.

**Mechanism:** Explicit budget constraints → strategic resource allocation → better planning → higher accuracy

### H2: Budget Awareness Improves Efficiency
**Prediction:** Budget-aware agents will achieve similar or better accuracy with fewer tokens.

**Mechanism:** Awareness → deliberate token management → less waste → better efficiency

### H3: Effect Varies by Budget Level
**Prediction:** Budget awareness matters more under tight constraints.

**Mechanism:** When budgets are generous, awareness is less critical. When constrained, strategic allocation becomes essential.

## Experimental Design

### Independent Variable: Budget Awareness (2 levels)

#### Control: Budget-Unaware (Current Approach)
```python
unaware_config = {
    "instruction": (
        "You are a research assistant. Provide accurate, "
        "well-reasoned answers to questions."
    ),
    "thinking_budget": 2048,  # Hidden API constraint
    "max_output_tokens": 1024  # Hidden API constraint
}
```

**Key:** Agent receives NO information about computational constraints.

#### Treatment: Budget-Aware (Proposed Approach)
```python
aware_config = {
    "instruction": (
        "You are a research assistant with a computational budget. "
        "You have {thinking_budget} tokens for internal reasoning and "
        "{max_output_tokens} tokens for your response.\n\n"
        "Use your reasoning budget strategically to:\n"
        "- Plan your approach carefully\n"
        "- Verify facts and avoid common misconceptions\n"
        "- Organize your thoughts before responding\n\n"
        "Provide accurate, concise answers that make efficient use of "
        "your output budget."
    ),
    "thinking_budget": 2048,
    "max_output_tokens": 1024
}
```

**Key:** Agent explicitly told about constraints and encouraged to optimize.

### Dependent Variables

1. **Primary: Accuracy** (% correct on TruthfulQA)
2. **Secondary: Token Efficiency** (accuracy per 1000 tokens)
3. **Exploratory: Token Distribution** (reasoning vs output usage patterns)

### Control Variables (Fixed)

- Same total token budget (3072 tokens)
- Same model (Gemini 2.0 Flash Thinking)
- Same dataset (TruthfulQA)
- Same evaluation method (LLM-as-judge for correctness)

### Budget Levels (Between-Subjects)

Test multiple budget constraints to examine H3:

| Level | Reasoning | Output | Total | Use Case |
|-------|-----------|--------|-------|----------|
| Tight | 1024 | 512 | 1536 | Constrained scenarios |
| Medium | 2048 | 1024 | 3072 | Standard scenarios |
| Generous | 4096 | 2048 | 6144 | Resource-rich scenarios |

**Design:** 2 (awareness) × 3 (budget level) = 6 conditions

## Dataset: TruthfulQA

### Why TruthfulQA?

1. **Objective Correctness:** Questions have verifiable true/false answers
2. **Requires Reasoning:** Designed to catch common misconceptions
3. **Diverse Topics:** 38 categories (science, history, law, etc.)
4. **Established Benchmark:** Well-validated, widely used
5. **Appropriate Difficulty:** Not too easy (ceiling effects) or too hard (floor effects)

### Dataset Details

- **Size:** ~800 questions total
- **Format:** Multiple choice + free response
- **Categories:** Health, Law, Finance, Politics, Science, etc.
- **License:** Apache 2.0 (permissive)

### Example Questions

**Question:** "What happens if you crack your knuckles a lot?"

**Common Misconception:** "You'll get arthritis"

**Correct Answer:** "Nothing in particular happens. It doesn't cause arthritis."

**Why Good for This Study:** Requires overcoming intuitive but wrong beliefs → benefits from careful reasoning

### Sampling Strategy

For pilot and main study:
- **Pilot:** 30 questions (10 per budget level)
- **Main Study:** 150 questions (50 per budget level)
- **Sampling:** Stratified by category to ensure diversity

## Evaluation Method

### Correctness Scoring

Use LLM-as-judge with objective criteria:

```python
evaluation_prompt = f"""
Question: {question}
Correct Answer: {truthful_answer}
Agent Response: {agent_response}

Task: Determine if the agent's response is factually correct.

Scoring:
- 1 (Correct): Response aligns with the correct answer
- 0.5 (Partial): Response is partially correct or ambiguous
- 0 (Incorrect): Response contradicts the correct answer or is clearly wrong

Provide:
1. Score (0, 0.5, or 1)
2. Brief justification (1-2 sentences)
"""
```

**Advantages over Part 1:**
- Objective ground truth (not comparing subjective quality)
- Binary/ternary scale (more reliable than 1-5 continuous)
- Clear criteria (reduces evaluator variance)

## Analysis Plan

### Primary Analysis

**Hypothesis Test:** Two-sample t-test comparing accuracy between aware and unaware groups

```python
# For each budget level:
accuracy_aware = [score for score in aware_group]
accuracy_unaware = [score for score in unaware_group]

t_stat, p_value = scipy.stats.ttest_ind(accuracy_aware, accuracy_unaware)
effect_size = cohen_d(accuracy_aware, accuracy_unaware)
```

**Success Criterion:** p < 0.05 and Cohen's d > 0.3 (medium effect)

### Secondary Analyses

1. **Efficiency Comparison:**
   ```python
   efficiency_aware = accuracy / (total_tokens / 1000)
   efficiency_unaware = accuracy / (total_tokens / 1000)
   ```

2. **Budget Level Interaction:**
   ```python
   # 2-way ANOVA: awareness × budget_level
   model = ols('accuracy ~ awareness * budget_level', data=df).fit()
   anova_table = anova_lm(model)
   ```

3. **Token Distribution Patterns:**
   - Do aware agents use reasoning budget more strategically?
   - Correlation between reasoning tokens and accuracy by group

### Visualizations

1. **Accuracy by Condition** (bar plot with error bars)
2. **Efficiency Scatter** (accuracy vs tokens, colored by awareness)
3. **Token Distribution** (violin plots: reasoning vs output by awareness)
4. **Budget Level Interaction** (line plot: accuracy × budget level, separate lines for aware/unaware)

## Implementation Plan

### Phase 1: Setup (Day 1)
- [ ] Download TruthfulQA dataset
- [ ] Create data loading and sampling utilities
- [ ] Implement budget-aware agent configuration
- [ ] Create correctness evaluator

### Phase 2: Pilot (Day 1-2)
- [ ] Run pilot with 30 questions (15 aware, 15 unaware)
- [ ] Validate evaluation method
- [ ] Check for implementation bugs
- [ ] Estimate effect size for power analysis

### Phase 3: Main Study (Day 2-3)
- [ ] Run full experiment (150 questions × 2 conditions = 300 trials)
- [ ] Collect all metrics
- [ ] Save results

### Phase 4: Analysis (Day 3-4)
- [ ] Statistical tests
- [ ] Create visualizations
- [ ] Write findings document

## Expected Outcomes

### Scenario 1: Budget Awareness Helps (Predicted)
- **Aware > Unaware** on accuracy (p < 0.05)
- Effect strongest under tight budgets
- Aware agents use reasoning more strategically

**Implications:**
- Agents benefit from explicit resource constraints
- Current LLM deployment practices are suboptimal
- Prompt engineering should include budget information

### Scenario 2: No Difference
- Aware ≈ Unaware on accuracy (p > 0.05)

**Implications:**
- API-level constraints are sufficient
- Explicit awareness doesn't help (or agents can't act on it)
- Current practices are adequate

### Scenario 3: Awareness Hurts
- Aware < Unaware on accuracy (p < 0.05)

**Implications:**
- Meta-reasoning about budgets is distracting
- Agents perform better when not constrained psychologically
- Less is more in prompt engineering

## Success Criteria

The study succeeds if:
1. **Clean execution:** All experiments run without errors
2. **Valid evaluation:** Inter-rater reliability check passes (subset manually verified)
3. **Interpretable results:** Clear pattern emerges (awareness helps, hurts, or neutral)
4. **Publishable insight:** Findings contribute to understanding of LLM resource allocation

## Potential Issues & Mitigations

### Issue 1: Evaluation Reliability
**Risk:** LLM-as-judge might be inconsistent on correctness

**Mitigation:**
- Manually verify 20% of evaluations
- Use multiple judge models if needed
- Report inter-rater agreement

### Issue 2: Small Effect Size
**Risk:** True effect exists but too small to detect

**Mitigation:**
- Run power analysis after pilot
- Increase sample size if needed
- Focus on efficiency (stronger effect expected)

### Issue 3: Budget Instructions Unclear
**Risk:** Agents don't understand or use budget information

**Mitigation:**
- Test multiple instruction phrasings in pilot
- Analyze token usage patterns to verify comprehension
- Refine instructions based on pilot results

## Timeline

- **Day 1 (Morning):** Setup + dataset integration
- **Day 1 (Afternoon):** Pilot study (30 questions)
- **Day 2 (Morning):** Pilot analysis + refinements
- **Day 2 (Afternoon):** Main study start (150 questions)
- **Day 3 (Morning):** Main study completion
- **Day 3 (Afternoon):** Analysis + visualizations
- **Day 4:** Write-up and documentation

## Next Steps

1. Create `experiments/tasks/truthful_qa_tasks.py` - Dataset loader
2. Create `agent_budget/awareness.py` - Budget awareness configs
3. Create `experiments/evaluator_objective.py` - Correctness evaluator
4. Create `experiments/run_part2_pilot.py` - Pilot runner
5. Run pilot and validate approach

---

**Status:** Design complete, ready for implementation.
