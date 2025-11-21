# Part 2: Multi-Agent Budget Coordination - Design Notes

## Research Question
How do different levels of budget awareness affect coordination and performance in multi-agent teams?

## Experimental Design

### 2-Agent Iterative Setup

**Researcher ⇄ Validator** (using ADK LoopAgent)

#### Agent 1: Researcher
- **Tools**: `google_search`
- **Role**: Search for information, propose/revise answer based on feedback
- **Output (Round 1)**: "Based on [evidence], I propose the answer is [X] because [reasoning]"
- **Output (Round N)**: "Based on feedback, I revise my answer to [Y] because [new reasoning]"

#### Agent 2: Validator
- **Tools**: `google_search` (can independently verify if needed)
- **Role**: Critically evaluate Researcher's proposal, provide feedback or approve
- **Output Options**:
  - **Approve**: "APPROVED. Final answer: [X] because [validation reasoning]"
  - **Request clarification**: "Please verify: Did you check [Y]?"
  - **Identify error**: "Your reasoning has a flaw: [Z]. Please revise."
  - **Request more info**: "You're missing information about [W]."

#### Agent 3: StopChecker (custom BaseAgent)
- **Role**: Check if Validator approved; escalate to exit loop if yes
- **Logic**: If "APPROVED" in validator_feedback → escalate (exit loop)

### Iteration Control
- **Max 3 rounds** to prevent infinite loops
- **Validator-controlled stop**: Validator can approve early if answer is good
- **Budget managed across rounds**: Both agents must allocate their budget efficiently across iterations
- **State tracking**: All proposals and feedback stored in session state

### Why Iterative?
- ✅ **More realistic**: Mirrors peer review, editorial collaboration
- ✅ **Tests coordination**: Aware teams should iterate more efficiently
- ✅ **Budget awareness impact**: Unaware teams may waste budget on unnecessary rounds
- ✅ **Natural for Condition D**: Researcher can request budget to address feedback
- ✅ **Richer metrics**: Track iteration count, feedback quality, improvement per round

### Implementation: ADK LoopAgent Pattern
Based on ADK's iterative refinement pattern:
```python
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
from google.adk.events import Event, EventActions

researcher = LlmAgent(
    name="Researcher",
    instruction="...",
    tools=[google_search],
    output_key="researcher_output"  # Saves to state['researcher_output']
)

validator = LlmAgent(
    name="Validator",
    instruction="...",
    tools=[google_search],
    output_key="validator_feedback"  # Saves to state['validator_feedback']
)

class CheckApproval(BaseAgent):
    async def _run_async_impl(self, ctx):
        feedback = ctx.session.state.get("validator_feedback", "")
        approved = "APPROVED" in feedback
        yield Event(author=self.name, actions=EventActions(escalate=approved))

iterative_team = LoopAgent(
    name="ResearchValidationLoop",
    max_iterations=3,
    sub_agents=[researcher, validator, CheckApproval(name="StopChecker")]
)
```

### Awareness Conditions (4)

**Condition A: No Awareness (Baseline)**
- Agents receive no budget information
- **Expected behavior**: May over-iterate, waste budget on unnecessary refinement rounds

**Condition B: Overall Team Budget Only**
- "Your team has 1600 tokens total across all iterations"
- **Expected behavior**: Teams should iterate more efficiently to conserve shared budget

**Condition C: Overall + Individual Allocations**
- "Team: 1600 tokens total"
- "Researcher: 960 tokens (60%)" / "Validator: 640 tokens (40%)"
- **Expected behavior**: Each agent manages their budget across iterations
- Researcher might limit searches per round, Validator might approve faster

**Condition D: With Negotiation**
- Agents can call `request_budget(amount, justification)` to access reserve pool
- **Expected behavior**:
  - Researcher: "I need 200 more tokens to address your feedback on X"
  - Validator: "I need 100 tokens to thoroughly verify the revised answer"
- Natural coordination through budget requests

### Task Selection

**Dataset: HotpotQA (via Hugging Face datasets library)**
- ✅ Already installed: `datasets>=4.4.1`
- 7,405 validation examples available
- Two question types:
  - **Bridge (5,918)**: Multi-hop questions requiring chaining facts
    - Example: "Who is the director of the film that features X?"
  - **Comparison (1,487)**: Questions requiring comparing two entities
    - Example: "Were Scott Derrickson and Ed Wood of the same nationality?"
- All marked as "hard" difficulty
- Ground truth answers provided
- Objective evaluation possible

**Pilot Study**: Use mix of both question types to evaluate:
- Which type better demonstrates coordination value
- Whether question type affects budget awareness impact

**Example Bridge Question**:
```
Q: "What was the population of the country that hosted the 2016 Summer Olympics?"

Expected workflow:
- Researcher: Searches "2016 Olympics host" → Brazil
              Searches "Brazil population 2016" → 207M
              Proposes: "207 million because Brazil hosted 2016 Olympics"

- Validator:  Independently verifies Brazil hosted 2016 Olympics ✓
              Cross-checks population figure ✓
              Final answer: "207 million (verified via [sources])"
```

**Example Comparison Question**:
```
Q: "Were Scott Derrickson and Ed Wood of the same nationality?"

Expected workflow:
- Researcher: Searches both directors
              Finds both are American
              Proposes: "Yes, both are American"

- Validator:  Verifies nationalities independently
              Confirms reasoning
              Final answer: "Yes, both are American (verified)"
```

**Required Output Format**:
- Both agents must provide **reasoning** with their answers
- Researcher: "I propose [answer] because [reasoning]"
- Validator: "I confirm/correct to [answer] because [validation reasoning]"
- This allows evaluation of both answer correctness AND reasoning quality

### Budget Allocation Strategy

**Total team budget**: 2000 tokens
- **NOTE**: Minimum 2000 required to meet Gemini's 512-token thinking budget minimum

**Researcher allocation**: 60% (1200 tokens)
- Primary information gathering role
- 720 reasoning + 480 output

**Validator allocation**: 40% (800 tokens)
- Verification and validation, with independent search capability
- 520 reasoning + 280 output (65% reasoning to ensure 512+ minimum)

**Condition D negotiation reserve**: 20% (400 tokens held back)
- Initial allocation: 1600 tokens (80%)
- Reserve pool: 400 tokens (20%)
- Validator gets 512 thinking tokens minimum (80% of 640)

## Implementation Checklist

### Phase 1: Foundation
- [ ] Extend UsageMonitor for multi-agent tracking
- [ ] Create MultiAgentMetrics dataclass
- [ ] Test 2-agent factory method
- [ ] Validate traceability (can capture both agent outputs)

### Phase 2: Task Integration
- [ ] Load HotpotQA dataset via `datasets` library
- [ ] Sample pilot questions: 5 questions (mix of bridge + comparison types)
- [ ] Create Part2Task dataclass (question, answer, type)
- [ ] Create HotpotQA evaluator (can reuse ObjectiveEvaluator pattern)

### Phase 3: Pilot Execution
- [ ] Implement run_part2_pilot.py
- [ ] Test: 5 questions × 4 awareness conditions = 20 trials
- [ ] Validate:
  - [ ] Both agents produce reasoning
  - [ ] Validator can search when needed
  - [ ] Budget tracking works for both agents
  - [ ] Can extract Researcher's proposal + Validator's verification
- [ ] Analyze pilot results to determine:
  - [ ] Which question type works better (bridge vs. comparison)
  - [ ] Whether 1600 token budget is appropriate
  - [ ] If awareness conditions show different patterns

### Phase 4: Full Study (if pilot is successful)
- [ ] Select final question set based on pilot learnings
- [ ] Run full study (target: 15-20 questions × 4 conditions = 60-80 trials)
- [ ] Comprehensive analysis and report

## Key Metrics

**Performance**:
- Final answer correctness (0.0-1.0 score, same as Part 1)
- Answer quality and reasoning quality

**Iteration Metrics** (NEW for Part 2):
- **Number of iterations**: How many rounds before approval?
- **Approval rate by round**: Did Validator approve in Round 1, 2, or 3?
- **Answer improvement**: Did reasoning improve from R1 → R2 → R3?
- **Iteration efficiency**: Fewer iterations = more efficient? Or better quality?

**Coordination**:
- Token usage per agent per round
- Total team token usage across all iterations
- Budget efficiency (% of budget used)
- Whether teams hit max_iterations (3) or approved early

**Feedback Quality**:
- Type of feedback: approval, clarification, error identification, info request
- Whether feedback led to improved answers
- Validator search usage (did they verify independently?)

**Negotiation (Condition D only)**:
- Number of budget requests per round
- Approval rate for requests
- Amount requested vs. allocated
- Which agent requests more (Researcher vs. Validator)

**Traceability**:
- All Researcher proposals (R1, R2, R3)
- All Validator feedback (R1, R2, R3)
- Final approved answer
- Complete iteration history

## Resolved Design Decisions

1. **Dataset**: ✅ **HotpotQA via datasets library**
   - Already installed, easy integration
   - 7,405 validation examples
   - Mix of bridge + comparison question types

2. **Study Scale**: ✅ **Pilot first, then full study**
   - Pilot: 5 questions × 4 conditions = 20 trials
   - Full: 15-20 questions × 4 conditions = 60-80 trials
   - Let pilot guide final scale

3. **Validator Tools**: ✅ **Yes, give Validator google_search**
   - More realistic (fact-checkers can verify independently)
   - Tests coordination differently (can double-check vs. must trust)
   - Still valuable: Validator reviews Researcher's reasoning first

4. **Answer Format**: ✅ **Require reasoning from both agents**
   - Enables evaluation of reasoning quality, not just final answer
   - Validator must explain validation/correction

5. **Question Types**: ✅ **Mix of bridge + comparison in pilot**
   - Compare performance across types
   - Determine which better demonstrates coordination value

## Next Steps

**Immediate priorities**:
1. **Implement LoopAgent-based iterative team** in agent_factory.py:
   - Create Researcher LlmAgent with google_search
   - Create Validator LlmAgent with google_search
   - Create CheckApproval BaseAgent (custom stop condition)
   - Assemble into LoopAgent with max_iterations=3
   - Support all 4 awareness conditions in instructions

2. **Extend monitor.py** for iteration tracking:
   - Create `IterationMetrics` dataclass (round number, proposal, feedback)
   - Create `MultiAgentMetrics` dataclass (list of iterations, total tokens per agent)
   - Extract iteration history from session state

3. **Create experiments/tasks/hotpotqa_tasks.py**:
   - Load HotpotQA dataset via `datasets` library
   - Sample pilot questions (mix of bridge + comparison)
   - Create HotpotQATask dataclass

4. **Implement run_part2_pilot.py**:
   - 5 questions × 4 conditions = 20 trials
   - Track iteration count, proposals, feedback per trial
   - Extract and evaluate final answers

**Success criteria for pilot**:
- All 20 trials complete successfully
- Iteration loop works (can stop early on approval or hit max 3 rounds)
- Both agents produce reasoning we can extract per round
- Can track: # iterations, token usage per agent per round, feedback type
- Clear behavioral differences across conditions A, B, C, D
- Budget awareness affects iteration efficiency (hypothesis: aware teams iterate less)
