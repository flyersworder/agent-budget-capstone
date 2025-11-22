"""Test the Part 2 module with a single LiveCodeBench problem."""

import asyncio
from datetime import datetime

from datasets import load_dataset
from dotenv import load_dotenv

from agent_budget.core import MultiAgentAwarenessCondition
from agent_budget.code_review_runner import run_code_review_trial


async def main():
    load_dotenv()

    # Load LiveCodeBench
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="release_v6",
        trust_remote_code=True,
    )

    # Filter to problems from Feb 2025 onwards (after model cutoff)
    all_problems = []
    for p in dataset:
        contest_date = datetime.fromisoformat(p["contest_date"])
        if contest_date >= datetime(2025, 2, 1):
            all_problems.append(p)

    # Get HARD problems
    hard_problems = [p for p in all_problems if p.get("difficulty") == "hard"]

    print(f"Found {len(hard_problems)} HARD problems after cutoff\n")

    # Test with first HARD problem
    problem = hard_problems[0]

    print("=" * 80)
    print("TESTING PART 2 MODULE")
    print("=" * 80)
    print(f"Problem: {problem['question_title']}")
    print(f"Platform: {problem['platform']}")
    print(f"Difficulty: {problem['difficulty']}")
    print()

    # Test NO_AWARENESS condition
    print("Testing with NO_AWARENESS condition...")
    print("-" * 80)

    result = await run_code_review_trial(
        problem=problem,
        awareness_condition=MultiAgentAwarenessCondition.NO_AWARENESS,
        max_iterations=3,
    )

    print("\nRESULTS:")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.num_iterations}")
    print(f"  Final decision: {result.final_decision}")
    print(f"  Team total tokens: {result.team_total_tokens:,}")
    print(f"  Test passed: {result.test_passed}")
    print()
    print("  Iteration history:")
    for iter_data in result.iterations:
        print(f"    Iteration {iter_data['iteration']}: {iter_data['status']}")

    print("\n" + "=" * 80)
    print("MODULE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
