"""Prompts and instructions for code review agents.

This module provides prompt generation functions for the Coder-Reviewer
iterative code review system.

Part 2 Experiment Design (Simplified):
- 2 conditions: NO_AWARENESS vs OVERALL_AND_INDIVIDUAL
- Meaningful framing: Budget reflects role specialization, not just numbers
"""

from agent_budget.core import (
    CODE_REVIEW_REVIEWER_BUDGET,
    MultiAgentAwarenessCondition,
    get_coder_budget,
)


def escape_curly_braces(text: str) -> str:
    """Escape curly braces to prevent ADK template variable substitution.

    ADK treats {var_name} as template variables. Problem descriptions often
    contain math notation like {K_i} which causes KeyError.

    We replace curly braces with Unicode lookalikes that render identically
    but won't be parsed as template variables.

    Args:
        text: Text that may contain curly braces

    Returns:
        Text with curly braces replaced with safe alternatives
    """
    # Use fullwidth curly brackets which look similar but aren't parsed
    # ｛ (U+FF5B) and ｝ (U+FF5D)
    # Or we can use angle brackets for math notation
    return text.replace("{", "❴").replace("}", "❵")


def generate_coder_instruction(
    problem_description: str,
    budget_message: str = "",
) -> str:
    """Generate instruction for Coder agent.

    Args:
        problem_description: The programming problem to solve
        budget_message: Optional budget awareness message

    Returns:
        Complete instruction string for Coder
    """
    # Budget message goes first if present (sets context)
    prefix = f"{budget_message}\n\n" if budget_message else ""

    # Escape curly braces in problem description to prevent ADK template errors
    safe_problem = escape_curly_braces(problem_description)

    # Team framing is consistent for BOTH conditions
    return f"""{prefix}You are the CODER in a 2-agent team. Your partner is a Reviewer who will test your code.

PROBLEM:
{safe_problem}

REQUIREMENTS:
1. Write a complete Python program that reads from stdin and writes to stdout
2. If you see feedback from the Reviewer, fix the issues they identified
3. Output ONLY the Python code - no explanations or markdown

Think through the algorithm carefully, then write clean, correct code."""


def generate_reviewer_instruction(
    budget_message: str = "",
) -> str:
    """Generate instruction for Reviewer agent.

    Args:
        budget_message: Optional budget awareness message

    Returns:
        Complete instruction string for Reviewer
    """
    prefix = f"{budget_message}\n\n" if budget_message else ""

    # Team framing is consistent for BOTH conditions
    return f"""{prefix}You are the REVIEWER in a 2-agent team. Your partner is a Coder who writes the code.

WORKFLOW:
1. Call test_code() to run the Coder's code against the test case
2. Based on the result, make your decision

OUTPUT FORMAT:
DECISION: APPROVE (if tests pass) or REQUEST_REVISION (if tests fail)
FEEDBACK: [Brief explanation of what happened]

Be concise - the test result tells you everything you need to know."""


def generate_budget_message(
    awareness_condition: MultiAgentAwarenessCondition,
    max_iterations: int,
    agent_role: str = "",
    difficulty: str = "medium",
) -> str:
    """Generate budget awareness message for agents.

    Design principles (from literature review):
    1. Challenge framing: Frame budget as "sufficient for" not "limited to"
    2. Focusing dividend: Constraints help prioritize what matters
    3. Role specialization: Give meaningful reason for allocation
    4. Actionable: Tell agents what they CAN do, not just what they can't

    Args:
        awareness_condition: Budget awareness level
        max_iterations: Maximum iterations
        agent_role: Agent role ("Coder" or "Reviewer")
        difficulty: Problem difficulty ("easy" or "medium") for budget lookup

    Returns:
        Budget message (empty for NO_AWARENESS)
    """
    if awareness_condition == MultiAgentAwarenessCondition.NO_AWARENESS:
        return ""

    # Get difficulty-based budget values
    coder_budget_obj = get_coder_budget(difficulty)
    coder_budget = coder_budget_obj.total
    reviewer_budget = CODE_REVIEW_REVIEWER_BUDGET.total

    # Get detailed budget breakdown for mechanism explanation
    coder_thinking = coder_budget_obj.reasoning_tokens
    coder_output = coder_budget_obj.output_tokens
    reviewer_thinking = CODE_REVIEW_REVIEWER_BUDGET.reasoning_tokens
    reviewer_output = CODE_REVIEW_REVIEWER_BUDGET.output_tokens

    if awareness_condition == MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL:
        if agent_role == "Coder":
            # Challenge-framed resource context - no behavioral guidance
            return f"""[RESOURCE CONTEXT]
You have {coder_budget} tokens per iteration - sufficient for a well-crafted solution.
This includes {coder_thinking} for reasoning and {coder_output} for code output.

You have up to {max_iterations} iterations to get it right."""

        else:  # Reviewer
            # Challenge-framed resource context - no behavioral guidance
            return f"""[RESOURCE CONTEXT]
You have {reviewer_budget} tokens per iteration - sufficient for testing and clear feedback.
This includes {reviewer_thinking} for analysis and {reviewer_output} for your response.

You have up to {max_iterations} iterations."""

    # Fallback for other conditions - return empty (not used in current design)
    return ""
