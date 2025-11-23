"""Budget awareness configurations for Part 1 experiments.

This module defines the two experimental conditions:
1. Budget-Unaware (Control): Current approach with no budget information
2. Budget-Aware (Treatment): Explicit budget information in instructions
"""

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
) -> tuple[str, BuiltInPlanner, types.GenerateContentConfig]:
    """Create agent configuration based on awareness condition.

    Args:
        condition: Experimental condition (aware or unaware)
        budget_config: Budget configuration to apply

    Returns:
        Tuple of (instruction, planner, generate_config) for Agent construction
    """
    # Create appropriate instruction based on condition
    if condition == AwarenessCondition.AWARE:
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
