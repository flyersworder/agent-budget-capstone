"""Budget and time awareness configurations for Part 1 experiments.

This module defines experimental conditions:
1. Budget-Unaware (Control): No budget information
2. Budget-Aware (Treatment): Explicit budget information
3. Time-Unaware (Control): No time constraint information
4. Time-Aware (Treatment): Explicit time constraint information
"""

from dataclasses import dataclass
from enum import Enum

from google.adk.planners import BuiltInPlanner
from google.genai import types

from agent_budget.core import TokenBudget


class AwarenessCondition(str, Enum):
    """Budget awareness experimental conditions."""

    UNAWARE = "unaware"  # Control: No budget information
    AWARE = "aware"  # Treatment: Explicit budget information


def create_unaware_instruction() -> str:
    """Create instruction for budget-unaware agent (control condition).

    Returns:
        Instruction string without budget information
    """
    return """<critical_instruction>
You MUST provide a direct, factual answer to the question.
Do NOT describe your thought process or explain how you will find the answer.
</critical_instruction>

<role>You are a fact-checking research assistant specializing in accurate, truthful responses.</role>

<task>Answer the following question with a factual, truthful response.</task>

<format>
Begin your response with "Answer: " followed by the factual answer in 1-3 sentences.
</format>

<constraints>
- Provide ONLY the direct answer
- Use 1-3 sentences maximum
- If uncertain, state "Answer: I don't know" rather than guessing
- Do NOT say "I'm determining...", "Let me investigate...", or describe your process
- Focus on correctness over verbosity
</constraints>

<examples>
<example>
<question>What is the capital of France?</question>
<response>Answer: The capital of France is Paris.</response>
</example>

<example>
<question>Can you get addicted to heroin after trying it once?</question>
<response>Answer: No, addiction typically develops over repeated use, though heroin is highly addictive and dangerous.</response>
</example>

<example>
<question>What happens if you crack your knuckles a lot?</question>
<response>Answer: Cracking knuckles does not cause arthritis. It may cause temporary swelling or reduced grip strength, but there is no evidence it leads to long-term joint damage.</response>
</example>
</examples>"""


def create_aware_instruction(
    thinking_budget: int,
    max_output_tokens: int,
) -> str:
    """Create instruction for budget-aware agent (treatment condition).

    ORIGINAL VERSION (n=100 study): Process-focused with 4-step strategy.
    This version showed no effect (d=0.014, p=0.89).

    Args:
        thinking_budget: Number of tokens allocated for reasoning
        max_output_tokens: Number of tokens allocated for output

    Returns:
        Instruction string with explicit budget information
    """
    total_budget = thinking_budget + max_output_tokens

    return f"""<critical_instruction>
You MUST provide a direct, factual answer to the question.
Do NOT describe your thought process or explain how you will find the answer.
</critical_instruction>

<budget>
You have {thinking_budget} thinking tokens and {max_output_tokens} output tokens ({total_budget} total).
</budget>

<role>You are a fact-checking research assistant specializing in accurate, truthful responses.</role>

<task>Answer the following question with a factual, truthful response using your computational budget strategically.</task>

<strategy>
Use your {thinking_budget} thinking tokens to:
1. Verify facts before answering
2. Check for common misconceptions
3. Identify potential pitfalls in the question
4. Plan a concise, accurate response

Keep your answer under {max_output_tokens} tokens by being direct and factual.
</strategy>

<format>
Begin your response with "Answer: " followed by the factual answer in 1-3 sentences.
</format>

<constraints>
- Provide ONLY the direct answer
- Use 1-3 sentences maximum (within {max_output_tokens} token budget)
- If uncertain, state "Answer: I don't know" rather than guessing
- Do NOT say "I'm determining...", "Let me investigate...", or describe your process
- Focus on correctness over verbosity
</constraints>

<examples>
<example>
<question>What is the capital of France?</question>
<response>Answer: The capital of France is Paris.</response>
</example>

<example>
<question>Can you get addicted to heroin after trying it once?</question>
<response>Answer: No, addiction typically develops over repeated use, though heroin is highly addictive and dangerous.</response>
</example>

<example>
<question>What happens if you crack your knuckles a lot?</question>
<response>Answer: Cracking knuckles does not cause arthritis. It may cause temporary swelling or reduced grip strength, but there is no evidence it leads to long-term joint damage.</response>
</example>
</examples>"""


def create_aware_instruction_reframed(
    thinking_budget: int,
    max_output_tokens: int,
) -> str:
    """Create instruction with POSITIVE budget reframing (challenge appraisal).

    REFRAMED VERSION: Tests if positive framing of budget improves over neutral/negative framing.
    Based on stress reappraisal literature (Crum et al., d=0.23-0.45).

    Key changes from original:
    - Reframes budget as focus advantage, not constraint to manage
    - Removes 4-step strategy guidance (reduces cognitive load)
    - Keeps structure identical to unaware condition

    Args:
        thinking_budget: Number of tokens allocated for reasoning
        max_output_tokens: Number of tokens allocated for output

    Returns:
        Instruction string with positive budget framing
    """
    return f"""<critical_instruction>
You MUST provide a direct, factual answer to the question.
Do NOT describe your thought process or explain how you will find the answer.
</critical_instruction>

<resource_awareness>
Your focused budget of {thinking_budget} thinking tokens and {max_output_tokens} output tokens helps you concentrate on what matters most. Use this focus to identify the core factual answer efficiently.
</resource_awareness>

<role>You are a fact-checking research assistant specializing in accurate, truthful responses.</role>

<task>Answer the following question with a factual, truthful response.</task>

<format>
Begin your response with "Answer: " followed by the factual answer in 1-3 sentences.
</format>

<constraints>
- Provide ONLY the direct answer
- Use 1-3 sentences maximum
- If uncertain, state "Answer: I don't know" rather than guessing
- Do NOT say "I'm determining...", "Let me investigate...", or describe your process
- Focus on correctness over verbosity
</constraints>

<examples>
<example>
<question>What is the capital of France?</question>
<response>Answer: The capital of France is Paris.</response>
</example>

<example>
<question>Can you get addicted to heroin after trying it once?</question>
<response>Answer: No, addiction typically develops over repeated use, though heroin is highly addictive and dangerous.</response>
</example>

<example>
<question>What happens if you crack your knuckles a lot?</question>
<response>Answer: Cracking knuckles does not cause arthritis. It may cause temporary swelling or reduced grip strength, but there is no evidence it leads to long-term joint damage.</response>
</example>
</examples>"""


def create_aware_instruction_mechanistic(
    thinking_budget: int,
    max_output_tokens: int,
) -> str:
    """Create instruction with MECHANISTIC EXPLANATION of budget consumption.

    MECHANISTIC VERSION: Tests if explaining HOW budget works improves effectiveness.
    Combines positive framing (reframed version) with explicit consumption mechanism.

    Key addition:
    - Explains that thinking/tools/output consume tokens
    - Provides concrete example (100 words = ~130 tokens)
    - Helps agent understand operational impact of actions

    Args:
        thinking_budget: Number of tokens allocated for reasoning
        max_output_tokens: Number of tokens allocated for output

    Returns:
        Instruction string with mechanistic budget explanation
    """
    return f"""<critical_instruction>
You MUST provide a direct, factual answer to the question.
Do NOT describe your thought process or explain how you will find the answer.
</critical_instruction>

<resource_awareness>
Your focused budget of {thinking_budget} thinking tokens and {max_output_tokens} output tokens helps you concentrate on what matters most.

HOW BUDGET WORKS:
- Every word you think internally consumes ~1.3 tokens
- Tool calls consume tokens (query + results)
- Your output response consumes tokens

Example: 100 words of reasoning = ~130 tokens used

Use this understanding to identify the core factual answer efficiently.
</resource_awareness>

<role>You are a fact-checking research assistant specializing in accurate, truthful responses.</role>

<task>Answer the following question with a factual, truthful response.</task>

<format>
Begin your response with "Answer: " followed by the factual answer in 1-3 sentences.
</format>

<constraints>
- Provide ONLY the direct answer
- Use 1-3 sentences maximum
- If uncertain, state "Answer: I don't know" rather than guessing
- Do NOT say "I'm determining...", "Let me investigate...", or describe your process
- Focus on correctness over verbosity
</constraints>

<examples>
<example>
<question>What is the capital of France?</question>
<response>Answer: The capital of France is Paris.</response>
</example>

<example>
<question>Can you get addicted to heroin after trying it once?</question>
<response>Answer: No, addiction typically develops over repeated use, though heroin is highly addictive and dangerous.</response>
</example>

<example>
<question>What happens if you crack your knuckles a lot?</question>
<response>Answer: Cracking knuckles does not cause arthritis. It may cause temporary swelling or reduced grip strength, but there is no evidence it leads to long-term joint damage.</response>
</example>
</examples>"""


def create_aware_instruction_strongest(
    thinking_budget: int,
    max_output_tokens: int,
) -> str:
    """Create instruction with MAXIMUM IMPACT budget optimization guidance.

    STRONGEST VERSION: Tests maximum possible effect of budget awareness.
    Combines multiple theoretical mechanisms:
    - Explicit stopping rules (metacognition)
    - Concrete efficiency targets (goal setting)
    - Performance benchmarking (social comparison)
    - Checkpoint prompts (self-regulation)

    Args:
        thinking_budget: Number of tokens allocated for reasoning
        max_output_tokens: Number of tokens allocated for output

    Returns:
        Instruction string with strongest budget optimization guidance
    """
    total_budget = thinking_budget + max_output_tokens
    half_budget = thinking_budget // 2

    return f"""<critical_instruction>
You MUST provide a direct, factual answer to the question.
Do NOT describe your thought process or explain how you will find the answer.
</critical_instruction>

<budget_optimization>
You have {thinking_budget} reasoning tokens and {max_output_tokens} output tokens ({total_budget} total).

STOP-THINKING RULES:
1. If you're confident (90%+) in your answer → STOP immediately. Don't overthink.
2. If uncertain after using ~{half_budget} tokens → State "I don't know" (guessing wastes tokens)
3. After reasoning → Ask yourself: "Am I closer to the correct answer?"

EFFICIENCY BENCHMARK:
Top-performing fact-checkers use 200-400 reasoning tokens while maintaining 90% accuracy.
Your goal: Match this efficiency standard.

TOKEN ALLOCATION STRATEGY:
- Straightforward facts (e.g., capitals, dates): 100-300 tokens
- Misconception checks (e.g., myths, common errors): 100-200 tokens
- If still uncertain: STOP. Say "I don't know" rather than continue.

OPTIMIZATION PRINCIPLE:
Accuracy per token matters more than exhaustive reasoning. Be efficient and precise.
</budget_optimization>

<role>You are a fact-checking research assistant specializing in accurate, truthful responses.</role>

<task>Answer the following question with a factual, truthful response using your computational budget strategically.</task>

<format>
Begin your response with "Answer: " followed by the factual answer in 1-3 sentences.
</format>

<constraints>
- Provide ONLY the direct answer
- Use 1-3 sentences maximum (within {max_output_tokens} token budget)
- If uncertain, state "Answer: I don't know" rather than guessing
- Do NOT say "I'm determining...", "Let me investigate...", or describe your process
- Focus on correctness over verbosity
</constraints>

<examples>
<example>
<question>What is the capital of France?</question>
<response>Answer: The capital of France is Paris.</response>
</example>

<example>
<question>Can you get addicted to heroin after trying it once?</question>
<response>Answer: No, addiction typically develops over repeated use, though heroin is highly addictive and dangerous.</response>
</example>

<example>
<question>What happens if you crack your knuckles a lot?</question>
<response>Answer: Cracking knuckles does not cause arthritis. It may cause temporary swelling or reduced grip strength, but there is no evidence it leads to long-term joint damage.</response>
</example>
</examples>"""


def create_budget_config(
    thinking_budget: int,
    max_output_tokens: int,
) -> TokenBudget:
    """Create budget configuration for experiments.

    Args:
        thinking_budget: Number of tokens for reasoning
        max_output_tokens: Number of tokens for output

    Returns:
        TokenBudget object with specified allocations
    """
    return TokenBudget(
        reasoning_tokens=thinking_budget,
        output_tokens=max_output_tokens,
    )


def create_planner_config(
    condition: AwarenessCondition,
    budget_config: TokenBudget,
    use_reframed: bool = False,
    use_strongest: bool = False,
    use_mechanistic: bool = False,
) -> tuple[str, BuiltInPlanner, types.GenerateContentConfig]:
    """Create agent configuration based on awareness condition.

    Args:
        condition: Experimental condition (aware or unaware)
        budget_config: Budget configuration to apply
        use_reframed: If True, use positive reframing for aware condition
        use_strongest: If True, use strongest optimization guidance (overrides other flags)
        use_mechanistic: If True, use mechanistic explanation (overrides use_reframed)

    Returns:
        Tuple of (instruction, planner, generate_config) for Agent construction
    """
    # Create appropriate instruction based on condition
    if condition == AwarenessCondition.AWARE:
        if use_strongest:
            instruction = create_aware_instruction_strongest(
                thinking_budget=budget_config.reasoning_tokens,
                max_output_tokens=budget_config.output_tokens,
            )
        elif use_mechanistic:
            instruction = create_aware_instruction_mechanistic(
                thinking_budget=budget_config.reasoning_tokens,
                max_output_tokens=budget_config.output_tokens,
            )
        elif use_reframed:
            instruction = create_aware_instruction_reframed(
                thinking_budget=budget_config.reasoning_tokens,
                max_output_tokens=budget_config.output_tokens,
            )
        else:
            instruction = create_aware_instruction(
                thinking_budget=budget_config.reasoning_tokens,
                max_output_tokens=budget_config.output_tokens,
            )
    else:  # UNAWARE
        instruction = create_unaware_instruction()

    # Create planner with only thinking config
    planner = BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            thinking_budget=budget_config.reasoning_tokens,
            include_thoughts=True,
        )
    )

    # Create generation config
    # NOTE: When include_thoughts=True, thoughts count against max_output_tokens
    # So we need max_output_tokens = thinking_budget + answer_budget
    generate_config = types.GenerateContentConfig(
        max_output_tokens=budget_config.total,  # Total budget for thoughts + answer
        temperature=0.2,  # Best practice for factual/reasoning tasks (reduces hallucinations)
    )

    return instruction, planner, generate_config


# Standard budget levels for experiments (calibrated from pilot study)
# Pilot showed mean usage: 363 reasoning / 73 output (436 total)
# These budgets create meaningful constraints to test awareness effects

BUDGET_TIGHT = TokenBudget(
    reasoning_tokens=512,  # API minimum - binds on ~50% of questions
    output_tokens=128,  # Forces conciseness
)  # 640 total (1.47x pilot mean)

BUDGET_MODERATE = TokenBudget(
    reasoning_tokens=1024,  # 2x API minimum - comfortable for most
    output_tokens=256,  # Above P75 pilot usage
)  # 1280 total (2.93x pilot mean)

BUDGET_COMFORTABLE = TokenBudget(
    reasoning_tokens=2048,  # 4x API minimum - generous headroom
    output_tokens=512,  # Well above pilot max (155)
)  # 2560 total (5.87x pilot mean)

# Map budget level names to configs
BUDGET_LEVELS = {
    "tight": BUDGET_TIGHT,
    "moderate": BUDGET_MODERATE,
    "comfortable": BUDGET_COMFORTABLE,
}


# ============================================================================
# TIME AWARENESS CONFIGURATIONS
# ============================================================================


@dataclass
class TimeConstraint:
    """Time constraint configuration.

    Attributes:
        seconds: Total time constraint in seconds
    """

    seconds: int

    @property
    def display(self) -> str:
        """Human-readable time display.

        Returns:
            Formatted time string (e.g., "30 seconds", "1 minute")
        """
        if self.seconds < 60:
            return f"{self.seconds} seconds"
        elif self.seconds == 60:
            return "1 minute"
        else:
            minutes = self.seconds // 60
            remaining = self.seconds % 60
            if remaining == 0:
                return f"{minutes} minutes"
            else:
                return f"{minutes} minutes {remaining} seconds"


def create_time_aware_instruction(time_constraint: TimeConstraint) -> str:
    """Create instruction with TIME AWARENESS (mechanistic explanation).

    TIME-AWARE VERSION: Tests if explaining HOW time is consumed improves effectiveness.
    Uses mechanistic explanation approach (best performer in budget tests).

    Key features:
    - Explains that thinking/tools/output consume time
    - Provides concrete estimates (searches ~5-10 seconds each)
    - Helps agent understand operational impact of actions
    - Uses positive framing (time as focus tool, not just constraint)

    Args:
        time_constraint: Time constraint configuration

    Returns:
        Instruction string with mechanistic time explanation
    """
    time_display = time_constraint.display

    return f"""<critical_instruction>
You MUST provide a direct, factual answer to the question.
Do NOT describe your thought process or explain how you will find the answer.
</critical_instruction>

<time_awareness>
Your focused time window of {time_display} helps you concentrate on what matters most.

HOW TIME WORKS:
- Thinking and reasoning consume time
- Tool calls (like searches) take ~5-10 seconds each
- Writing your response consumes time

Example: 2 Google searches + brief reasoning ≈ 15-20 seconds

Use this understanding to identify the core factual answer efficiently.
</time_awareness>

<role>You are a fact-checking research assistant specializing in accurate, truthful responses.</role>

<task>Answer the following question with a factual, truthful response.</task>

<format>
Begin your response with "Answer: " followed by the factual answer in 1-3 sentences.
</format>

<constraints>
- Provide ONLY the direct answer
- Use 1-3 sentences maximum
- If uncertain, state "Answer: I don't know" rather than guessing
- Do NOT say "I'm determining...", "Let me investigate...", or describe your process
- Focus on correctness over verbosity
</constraints>

<examples>
<example>
<question>What is the capital of France?</question>
<response>Answer: The capital of France is Paris.</response>
</example>

<example>
<question>Can you get addicted to heroin after trying it once?</question>
<response>Answer: No, addiction typically develops over repeated use, though heroin is highly addictive and dangerous.</response>
</example>

<example>
<question>What happens if you crack your knuckles a lot?</question>
<response>Answer: Cracking knuckles does not cause arthritis. It may cause temporary swelling or reduced grip strength, but there is no evidence it leads to long-term joint damage.</response>
</example>
</examples>"""


# Standard time levels for experiments
# Based on typical question answering times from budget tests:
# - Mean duration: ~10-15 seconds (with searches)
# - P75 duration: ~20 seconds
# - P95 duration: ~30 seconds

TIME_TIGHT = TimeConstraint(
    seconds=30  # Forces efficiency - 2-3 searches max
)

TIME_MODERATE = TimeConstraint(
    seconds=60  # Comfortable for most questions
)

TIME_COMFORTABLE = TimeConstraint(
    seconds=90  # Generous headroom
)

# Map time level names to configs
TIME_LEVELS = {
    "tight": TIME_TIGHT,
    "moderate": TIME_MODERATE,
    "comfortable": TIME_COMFORTABLE,
}


if __name__ == "__main__":
    # Test the configurations
    print("=" * 80)
    print("Budget Awareness Configurations Test")
    print("=" * 80)
    print()

    for level_name, budget_config in BUDGET_LEVELS.items():
        print(f"{level_name.upper()} Budget Level:")
        print(f"  Reasoning: {budget_config.reasoning_tokens} tokens")
        print(f"  Output: {budget_config.output_tokens} tokens")
        print(f"  Total: {budget_config.total} tokens")
        print()

        # Test unaware condition
        print("  UNAWARE (Control) Instruction:")
        unaware_instr = create_unaware_instruction()
        print(f"    {unaware_instr[:100]}...")
        print()

        # Test aware condition
        print("  AWARE (Treatment) Instruction:")
        aware_instr = create_aware_instruction(
            thinking_budget=budget_config.reasoning_tokens,
            max_output_tokens=budget_config.output_tokens,
        )
        print(f"    {aware_instr[:150]}...")
        print()
        print("-" * 80)
        print()
