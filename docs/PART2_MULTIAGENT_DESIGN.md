# Part 2: Multi-Agent Budget Coordination Study Design

## Research Question

**Can multi-agent systems with coordination overcome the budget awareness paradox observed in single agents?**

## Background

Part 1 found that **budget awareness hurts single-agent performance** (-12.9pp accuracy, p=0.028).

**Key question**: Does this paradox persist in multi-agent systems, or can coordination mechanisms help?

## Hypotheses

### H1: Multi-Agent Coordination Helps
**Prediction**: Multi-agent teams can distribute cognitive load, potentially mitigating the overhead of budget awareness.

### H2: Explicit Budget Allocation Improves Performance
**Prediction**: Knowing individual allocations allows agents to specialize and optimize better than only knowing overall budget.

### H3: Negotiation Enables Adaptive Allocation
**Prediction**: Allowing agents to negotiate budgets enables dynamic resource allocation based on task complexity.

## Experimental Design

### Study Type
**Between-subjects factorial design**

**Factors**:
- Team Size: 3 agents (fixed for simplicity)
- Budget Awareness: 4 conditions (described below)
- Overall Budget: 3 levels (tight, moderate, comfortable)

**Total conditions**: 4 awareness × 3 budgets = **12 conditions**

**Sample size**: 120 questions (10 per condition), between-subjects

### Multi-Agent Architecture

**ADK Pattern**: `SequentialAgent` with 3 sub-agents

```
Question → Researcher → Analyzer → Synthesizer → Final Answer
```

**Agent Roles**:
1. **Researcher**: Gathers information using Google Search
2. **Analyzer**: Evaluates information for correctness
3. **Synthesizer**: Produces final concise answer

**Why Sequential?**
- Natural task decomposition (research → analyze → synthesize)
- Budget accumulates across stages (easy to track)
- State passed between agents via ADK's built-in mechanism

### Budget Levels

Same total budgets as Part 1 for direct comparison:

| Level | Total | Per Agent (equal split) |
|-------|-------|-------------------------|
| Tight | 640 | ~213 each |
| Moderate | 1280 | ~426 each |
| Comfortable | 2560 | ~853 each |

**Note**: Actual allocation varies by condition

## Four Awareness Conditions

### Condition A: No Awareness (Baseline)
**Description**: Multi-agent with no budget information

**Instructions**: Standard task-focused instructions, no mention of budgets

**Budget allocation**: Equal split (total/3)

**Expected behavior**:
- Agents operate independently
- No resource optimization
- Baseline for multi-agent performance

**Comparison to Part 1**: Tests whether multi-agent alone helps (vs. Part 1 unaware)

---

### Condition B: Overall Budget Awareness
**Description**: All agents know the total team budget, but not individual allocations

**Instructions**:
```
"Your team has a total budget of {total} tokens. Use this information to
coordinate resource usage across all agents. You are the {role} agent."
```

**Budget allocation**: Equal split (total/3)

**Expected behavior**:
- Agents might self-limit to save budget for teammates
- Awareness of collective constraint
- No explicit individual targets

**Hypothesis**: Shared awareness creates collective overhead (like Part 1)

---

### Condition C: Overall + Individual Budget Awareness
**Description**: Each agent knows both the total budget AND their individual allocation

**Instructions**:
```
"Your team has a total budget of {total} tokens.
Your individual allocation as {role} is {individual} tokens.
Optimize your work within your allocation while considering the team's overall budget."
```

**Budget allocation**: Role-based allocation
- Researcher: 40% (needs tools, thinking)
- Analyzer: 35% (deep reasoning)
- Synthesizer: 25% (concise output)

**Expected behavior**:
- Agents optimize for their specific allocation
- Clear boundaries reduce anxiety
- Specialization based on role

**Hypothesis**: Granular info reduces overhead (clearer targets)

---

### Condition D: Overall Budget + Negotiation
**Description**: Agents can request additional budget from a shared pool

**Instructions**:
```
"Your team has a total budget of {total} tokens.
Your initial allocation as {role} is {initial} tokens.
If needed, you can request additional budget using the request_budget tool.
Provide justification for your request."
```

**Budget allocation**:
- Initial: Conservative (60% of total distributed)
- Reserve pool: 40% available for requests
- Negotiation: Via `request_budget(amount, justification)` tool

**Request evaluation**: Simple rule-based (approve if pool has funds + justification mentions "complexity" or "accuracy")

**Expected behavior**:
- Agents request more for hard questions
- Adaptive allocation based on need
- Negotiation overhead vs. flexibility

**Hypothesis**: Adaptive allocation improves performance on hard tasks

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `MultiAgentConfig` dataclass in `core.py`
2. Refactor `AgentFactory` to support multi-agent teams
3. Create negotiation tool (`request_budget`)

### Phase 2: Experiment Runner
1. Create `experiments/run_part2_multiagent.py`
2. Implement SequentialAgent with 3 roles
3. Add budget tracking across agent chain
4. Implement awareness condition logic

### Phase 3: Evaluation
1. Use same TruthfulQA dataset (different questions from Part 1)
2. Objective correctness evaluation (LLM-as-judge)
3. Track: total tokens, per-agent tokens, negotiation attempts
4. Compare to Part 1 single-agent results

## Success Metrics

**Primary**: Accuracy on TruthfulQA questions

**Secondary**:
- Token utilization efficiency (% of budget used)
- Per-agent contribution (which agent uses most tokens?)
- Negotiation patterns (in condition D: approval rate, request frequency)
- Comparison to Part 1 single-agent baseline

## Expected Outcomes

**Scenario 1: Multi-agent overcomes paradox**
- Condition B, C, or D > Condition A
- Coordination mechanisms mitigate awareness overhead
- Publishable: "Multi-agent coordination solves budget awareness problem"

**Scenario 2: Paradox persists**
- Condition A > B, C, D
- Awareness still hurts, even with coordination
- Publishable: "Budget awareness overhead is fundamental, not architectural"

**Scenario 3: Granularity matters**
- Condition C > B (individual info helps)
- Or Condition D > C (negotiation helps)
- Publishable: "Budget allocation strategy affects multi-agent performance"

## Technical Implementation Details

### SequentialAgent Configuration

```python
from google.adk.agents import SequentialAgent, LlmAgent

researcher = LlmAgent(
    name="researcher",
    instruction=get_instruction(role="researcher", condition=condition),
    tools=[google_search],
    output_key="research_findings"
)

analyzer = LlmAgent(
    name="analyzer",
    instruction=get_instruction(role="analyzer", condition=condition),
    # Can reference: {research_findings}
    output_key="analysis"
)

synthesizer = LlmAgent(
    name="synthesizer",
    instruction=get_instruction(role="synthesizer", condition=condition),
    # Can reference: {research_findings}, {analysis}
    output_key="final_answer"
)

team = SequentialAgent(
    name="research_team",
    sub_agents=[researcher, analyzer, synthesizer]
)
```

### Budget Tracking

Track tokens across the chain:
```python
total_tokens = 0
for event in runner.run_stream(team, ...):
    if hasattr(event, 'usage_metadata'):
        total_tokens += event.usage_metadata.total_token_count
    # Check against budget limit
    if total_tokens > budget_config.total:
        # Log budget exceeded
        break
```

### Negotiation Tool (Condition D)

```python
def request_budget(
    amount: int,
    justification: str,
    tool_context: ToolContext
) -> dict:
    """Request additional budget from shared pool.

    Args:
        amount: Additional tokens requested
        justification: Reason for request

    Returns:
        Approval status and allocated amount
    """
    pool = tool_context.state.get("budget_pool", 0)

    # Simple rule: approve if available and justified
    is_complex = any(word in justification.lower()
                     for word in ["complex", "difficult", "accuracy"])

    if pool >= amount and is_complex:
        tool_context.state["budget_pool"] -= amount
        return {"approved": True, "allocated": amount}
    else:
        return {"approved": False, "allocated": 0, "reason": "Insufficient pool or justification"}
```

## Data Collection

Same as Part 1:
- Question ID, condition, budget level
- Per-agent tokens (reasoning + output)
- Total team tokens
- Thinking text from each agent
- Final response
- Correctness score
- Negotiation log (condition D only)

## Analysis Plan

### Statistical Tests
1. **Main effect of multi-agent**: Condition A vs Part 1 unaware
2. **Main effect of awareness**: Condition A vs B/C/D
3. **Granularity effect**: Condition B vs C
4. **Negotiation effect**: Condition C vs D
5. **Budget level interactions**: 3-way ANOVA

### Qualitative Analysis
- How do agents specialize?
- What triggers budget requests?
- Do agents reference team budget in reasoning?
- Division of labor patterns

## Timeline

- **Week 1**: Implement infrastructure (factory, configs)
- **Week 2**: Build experiment runner, test with pilot (n=12)
- **Week 3**: Run full study (n=120)
- **Week 4**: Analysis and documentation

## Open Questions

1. Should we use equal split or role-based initial allocation for condition A/B?
   - **Decision**: Equal split for cleaner comparison

2. What if agents exceed budget mid-chain?
   - **Decision**: Allow completion but log as "budget_exceeded"

3. Should condition D agents see each other's requests?
   - **Decision**: No, keep requests independent to avoid strategic behavior

## References

- Part 1 findings: `PART1_COMPREHENSIVE_REPORT.md`
- ADK SequentialAgent docs: https://google.github.io/adk-docs/agents/multi-agents
- TruthfulQA dataset: https://github.com/sylinrl/TruthfulQA
