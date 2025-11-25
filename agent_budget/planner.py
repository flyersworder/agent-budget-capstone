"""Budget Planner Agent for Part 2 Extension.

This module implements a planning stage that estimates budget requirements
for code generation tasks BEFORE the Coder-Reviewer team begins work.

Research Question:
- Does dynamic, problem-specific budget estimation improve performance
  compared to fixed, difficulty-based budgets?

Design:
- Single LLM call with structured output
- Estimates: token budget per iteration, expected iterations
- Same model (Gemini 2.5 Flash Lite) for consistency
- Consequence-aware framing uses planner estimates instead of fixed values
"""

import json
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types


@dataclass
class PlannerEstimate:
    """Structured output from the budget planner.

    Attributes:
        estimated_tokens_per_iteration: Estimated tokens needed per iteration
        estimated_iterations: Expected number of iterations to solve
        reasoning: Brief explanation of the estimates
        raw_response: Full LLM response for debugging
    """

    estimated_tokens_per_iteration: int
    estimated_iterations: int
    reasoning: str
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "estimated_tokens_per_iteration": self.estimated_tokens_per_iteration,
            "estimated_iterations": self.estimated_iterations,
            "reasoning": self.reasoning,
        }


# Planner prompt - asks for structured JSON output
PLANNER_PROMPT = """You are a planning agent. Your task is to estimate the resources needed to solve a programming problem.

PROBLEM:
{problem_description}

Based on this problem, estimate:
1. How many tokens (code output) will be needed per iteration attempt?
   - Simple problems: 500-1500 tokens
   - Medium problems: 1500-2500 tokens
   - Complex problems: 2500-4000 tokens

2. How many iterations will likely be needed?
   - Easy problems: typically 1-2 iterations
   - Medium problems: typically 2-3 iterations
   - Complex problems: may need all 3 iterations

Respond with ONLY a JSON object in this exact format:
{{
    "estimated_tokens_per_iteration": <integer>,
    "estimated_iterations": <integer 1-3>,
    "reasoning": "<brief 1-2 sentence explanation>"
}}"""


async def estimate_budget(
    problem_description: str,
    model: str = "gemini-2.5-flash-lite",
    max_tokens: int = 256,
) -> PlannerEstimate:
    """Estimate budget requirements for a coding problem.

    Makes a single LLM call to estimate token needs and iteration count.
    Uses the same model as the main agents for consistency.

    Args:
        problem_description: The full problem description
        model: Model to use for planning (default: gemini-2.5-flash-lite)
        max_tokens: Max output tokens for planner (default: 256)

    Returns:
        PlannerEstimate with token and iteration estimates
    """
    client = genai.Client()

    prompt = PLANNER_PROMPT.format(problem_description=problem_description[:4000])

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.1,  # Low temperature for consistent estimates
        ),
    )

    raw_text = response.text.strip()

    # Parse JSON response
    try:
        # Handle potential markdown code blocks
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        data = json.loads(raw_text)

        # Validate and clamp values
        tokens = int(data.get("estimated_tokens_per_iteration", 2000))
        tokens = max(500, min(5000, tokens))  # Clamp to reasonable range

        iterations = int(data.get("estimated_iterations", 2))
        iterations = max(1, min(3, iterations))  # Clamp to 1-3

        reasoning = str(data.get("reasoning", "No reasoning provided"))

        return PlannerEstimate(
            estimated_tokens_per_iteration=tokens,
            estimated_iterations=iterations,
            reasoning=reasoning,
            raw_response=response.text,
        )

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Fallback to defaults if parsing fails
        return PlannerEstimate(
            estimated_tokens_per_iteration=2000,
            estimated_iterations=2,
            reasoning=f"Parse error, using defaults: {str(e)}",
            raw_response=response.text,
        )


def estimate_budget_sync(
    problem_description: str,
    model: str = "gemini-2.5-flash-lite",
    max_tokens: int = 256,
) -> PlannerEstimate:
    """Synchronous wrapper for estimate_budget.

    Useful for testing and non-async contexts.
    """
    import asyncio

    return asyncio.run(estimate_budget(problem_description, model, max_tokens))
