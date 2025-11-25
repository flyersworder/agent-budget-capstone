"""Test the budget planner agent.

This script tests the planner's ability to estimate budget requirements
for different types of coding problems.
"""

import asyncio
from datasets import load_dataset
from dotenv import load_dotenv

from agent_budget.planner import estimate_budget


# Sample problems of varying complexity
SAMPLE_PROBLEMS = {
    "easy": """
Given an array of integers, return the sum of all elements.

Example:
Input: [1, 2, 3, 4, 5]
Output: 15

Constraints:
- 1 <= len(arr) <= 100
- -1000 <= arr[i] <= 1000
""",
    "medium": """
Given a string containing digits from 2-9 inclusive, return all possible
letter combinations that the number could represent on a phone keypad.

Example:
Input: "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

Constraints:
- 0 <= digits.length <= 4
- digits[i] is a digit in the range ['2', '9']
""",
    "hard": """
Given an m x n grid filled with non-negative numbers, find a path from top left
to bottom right which minimizes the sum of all numbers along its path.

You can only move either down or right at any point in time.

Additionally, you must implement dynamic programming with space optimization
(O(n) space instead of O(m*n)).

Example:
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7 (path: 1→3→1→1→1)

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 200
- 0 <= grid[i][j] <= 100
""",
}


async def test_planner_on_samples():
    """Test planner on sample problems."""
    print("=" * 70)
    print("PLANNER ACCURACY TEST - SAMPLE PROBLEMS")
    print("=" * 70)

    for difficulty, problem in SAMPLE_PROBLEMS.items():
        print(f"\n--- {difficulty.upper()} ---")
        print(f"Problem preview: {problem[:100].strip()}...")

        estimate = await estimate_budget(problem)

        print("\nPlanner Estimate:")
        print(f"  Tokens per iteration: {estimate.estimated_tokens_per_iteration}")
        print(f"  Expected iterations: {estimate.estimated_iterations}")
        print(f"  Reasoning: {estimate.reasoning}")

        # Expected ranges (rough heuristics)
        expected_tokens = {
            "easy": (500, 1500),
            "medium": (1500, 2500),
            "hard": (2500, 4000),
        }

        min_tok, max_tok = expected_tokens[difficulty]
        in_range = min_tok <= estimate.estimated_tokens_per_iteration <= max_tok

        print(f"\n  Expected range: {min_tok}-{max_tok} tokens")
        print(f"  In range: {'✓' if in_range else '✗'}")


async def test_planner_on_livecodebench():
    """Test planner on real LiveCodeBench problems."""
    print("\n" + "=" * 70)
    print("PLANNER ACCURACY TEST - LIVECODEBENCH")
    print("=" * 70)

    # Load a few problems
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="release_v6",
        trust_remote_code=True,
    )

    # Get 2 easy and 2 medium problems
    easy_problems = [p for p in dataset if p.get("difficulty") == "easy"][:2]
    medium_problems = [p for p in dataset if p.get("difficulty") == "medium"][:2]

    test_problems = easy_problems + medium_problems

    results = []
    for problem in test_problems:
        title = problem.get("question_title", "Unknown")
        difficulty = problem.get("difficulty", "unknown")
        description = problem.get("question_content", "")

        print(f"\n--- {title} ({difficulty}) ---")

        estimate = await estimate_budget(description)

        print(f"  Tokens: {estimate.estimated_tokens_per_iteration}")
        print(f"  Iterations: {estimate.estimated_iterations}")
        print(f"  Reasoning: {estimate.reasoning[:100]}...")

        results.append(
            {
                "title": title,
                "difficulty": difficulty,
                "estimated_tokens": estimate.estimated_tokens_per_iteration,
                "estimated_iterations": estimate.estimated_iterations,
            }
        )

    # Summary
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)

    for r in results:
        print(
            f"  {r['difficulty']:8} | {r['estimated_tokens']:5} tok | {r['estimated_iterations']} iter | {r['title'][:30]}"
        )

    # Check variance
    token_values = [r["estimated_tokens"] for r in results]
    print(f"\n  Token range: {min(token_values)} - {max(token_values)}")
    print(
        f"  Token variance exists: {'Yes' if max(token_values) != min(token_values) else 'No'}"
    )


async def main():
    """Run all planner tests."""
    load_dotenv()

    await test_planner_on_samples()
    await test_planner_on_livecodebench()

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
