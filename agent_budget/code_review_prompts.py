"""Prompts and instructions for code review agents.

This module provides prompt generation functions for the Coder-Reviewer
iterative code review system.

Part 2 Experiment Design (Simplified):
- 2 conditions: NO_AWARENESS vs OVERALL_AND_INDIVIDUAL
- Meaningful framing: Budget reflects role specialization, not just numbers
"""

from agent_budget.core import (
    CODE_REVIEW_CODER_BUDGET,
    CODE_REVIEW_REVIEWER_BUDGET,
    CODE_REVIEW_TEAM_BUDGET,
    MultiAgentAwarenessCondition,
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

    return f"""{prefix}You are a Python programmer. Write code to solve this problem.

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

    return f"""{prefix}You are a code reviewer. Test the code and decide whether to approve it.

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

    Returns:
        Budget message (empty for NO_AWARENESS)
    """
    if awareness_condition == MultiAgentAwarenessCondition.NO_AWARENESS:
        return ""

    # Get budget values
    team_total = CODE_REVIEW_TEAM_BUDGET
    coder_budget = CODE_REVIEW_CODER_BUDGET.total
    coder_pct = round(100 * coder_budget / team_total)
    reviewer_budget = CODE_REVIEW_REVIEWER_BUDGET.total
    reviewer_pct = round(100 * reviewer_budget / team_total)

    # Get detailed budget breakdown for mechanism explanation
    coder_thinking = CODE_REVIEW_CODER_BUDGET.reasoning_tokens
    coder_output = CODE_REVIEW_CODER_BUDGET.output_tokens
    reviewer_thinking = CODE_REVIEW_REVIEWER_BUDGET.reasoning_tokens
    reviewer_output = CODE_REVIEW_REVIEWER_BUDGET.output_tokens

    if awareness_condition == MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL:
        if agent_role == "Coder":
            # Challenge framing with mechanism explanation
            return f"""[TEAM RESOURCES]
You are the CODER in a 2-agent team.

Your allocation: {coder_budget} tokens ({coder_pct}% of team)
- {coder_thinking} tokens for thinking/reasoning (internal deliberation)
- {coder_output} tokens for output (your code)

HOW TOKENS WORK: Both thinking and output consume your budget. Use them wisely:
- Efficient thinking: Focus on the core algorithm, avoid over-analyzing edge cases
- Efficient output: Write clean, complete code without verbose comments

Your partner (Reviewer): {reviewer_budget} tokens ({reviewer_pct}%)
- Tests your code and provides targeted feedback if needed

You have {max_iterations} iterations to succeed. A focused first attempt is most efficient."""

        else:  # Reviewer
            # Challenge framing with mechanism explanation
            return f"""[TEAM RESOURCES]
You are the REVIEWER in a 2-agent team.

Your allocation: {reviewer_budget} tokens ({reviewer_pct}% of team)
- {reviewer_thinking} tokens for thinking/reasoning
- {reviewer_output} tokens for output (your decision)

HOW TOKENS WORK: Both thinking and output consume your budget. Use them wisely:
- The test_code tool execution is FREE (doesn't use tokens)
- Your tokens are for interpreting results and writing feedback

Your partner (Coder): {coder_budget} tokens ({coder_pct}%)

You have {max_iterations} iterations. Be decisive: approve if tests pass, request specific fixes if not."""

    # Fallback for other conditions (OVERALL_ONLY, RESERVE_AWARENESS)
    # Not used in simplified 2x2 design but kept for compatibility
    return f"""[TEAM RESOURCES]
Team budget: {team_total} tokens across {max_iterations} iterations.
This is sufficient for the task - focus on quality."""
