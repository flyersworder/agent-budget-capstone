"""Prompts and instructions for code review agents.

This module provides prompt generation functions for the Coder-Reviewer
iterative code review system.
"""

from agent_budget.core import (
    CODE_REVIEW_CODER_BUDGET,
    CODE_REVIEW_REVIEWER_BUDGET,
    CODE_REVIEW_TEAM_BUDGET,
    MultiAgentAwarenessCondition,
)


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
    return f"""{budget_message}

YOU ARE A PYTHON CODE GENERATOR. Your ONLY job is to write Python code.

Problem to solve:
{problem_description}

INSTRUCTIONS:
1. Write a complete, working Python program
2. The program must read input from stdin
3. The program must write output to stdout
4. If you see review feedback in the conversation history, FIX the code based on that feedback
5. Return ONLY executable Python code - NO explanations, NO markdown, NO comments outside the code

Your response should be pure Python code that can be executed immediately."""


def generate_reviewer_instruction(
    budget_message: str = "",
) -> str:
    """Generate instruction for Reviewer agent.

    Args:
        budget_message: Optional budget awareness message

    Returns:
        Complete instruction string for Reviewer
    """
    return f"""{budget_message}

Your task: Test the code and make a decision.

CRITICAL: You MUST use the test_code function. Do NOT write code yourself.

Step 1: Call test_code() - it will automatically test the Coder's code
Step 2: Based on the test result, output your decision

After calling test_code, output:
DECISION: APPROVE or REQUEST_REVISION
FEEDBACK: [what the test showed]"""


def generate_budget_message(
    awareness_condition: MultiAgentAwarenessCondition,
    max_iterations: int,
    agent_role: str = "",
) -> str:
    """Generate budget awareness message for agents.

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
    reviewer_budget = CODE_REVIEW_REVIEWER_BUDGET.total

    if awareness_condition == MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL:
        # Role-specific message with individual and team awareness
        agent_budget = coder_budget if agent_role == "Coder" else reviewer_budget
        return f"""[BUDGET AWARENESS]
Your team has a total budget of {team_total} tokens for this task.
Your individual allocation: {agent_budget} tokens
Maximum {max_iterations} iterations available.
Use tokens wisely - be concise and focused.
"""

    # Fallback for other conditions (not used in clean 2x2 design)
    return f"""[BUDGET AWARENESS]
You are working in a team with a limited token budget.
Maximum {max_iterations} iterations available.
Use tokens wisely - be concise and focused.
"""
