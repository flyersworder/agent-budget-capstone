"""Test the Part 2 code review module with LiveCodeBench problems.

This test verifies:
1. The Coder-Reviewer loop works correctly
2. Both awareness conditions (NO_AWARENESS, OVERALL_AND_INDIVIDUAL) function
3. The iterative refinement loop handles test failures appropriately
"""

import asyncio
import logging
from datetime import datetime

from datasets import load_dataset
from dotenv import load_dotenv

from agent_budget.core import MultiAgentAwarenessCondition
from agent_budget.code_review_runner import run_code_review_trial

# Quieter logging - only show warnings and errors
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


async def main():
    load_dotenv()

    print("Loading LiveCodeBench dataset...")
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

    # Get MEDIUM problems (more tractable than HARD for testing)
    medium_problems = [p for p in all_problems if p.get("difficulty") == "medium"]
    hard_problems = [p for p in all_problems if p.get("difficulty") == "hard"]

    print(
        f"Found {len(medium_problems)} MEDIUM and {len(hard_problems)} HARD problems after cutoff\n"
    )

    # Test on multiple problems (first 5 medium problems)
    num_problems = min(5, len(medium_problems))
    test_problems = medium_problems[:num_problems]

    print("=" * 80)
    print(f"PART 2 CODE REVIEW TEST - Testing {num_problems} problems")
    print("=" * 80)

    results_unaware = []
    results_aware = []

    for i, problem in enumerate(test_problems):
        print(f"\n{'=' * 80}")
        print(f"PROBLEM {i + 1}/{num_problems}: {problem['question_title']}")
        print(f"Platform: {problem['platform']} | Difficulty: {problem['difficulty']}")
        print("=" * 80)

        # Test NO_AWARENESS condition
        print("\n[NO_AWARENESS]")
        result_unaware = await run_code_review_trial(
            problem=problem,
            awareness_condition=MultiAgentAwarenessCondition.NO_AWARENESS,
            max_iterations=3,
        )
        results_unaware.append(result_unaware)
        print(
            f"  Success: {result_unaware.success} | Iterations: {result_unaware.num_iterations} | Tokens: {result_unaware.team_total_tokens}"
        )

        # Test OVERALL_AND_INDIVIDUAL condition
        print("\n[OVERALL_AND_INDIVIDUAL]")
        result_aware = await run_code_review_trial(
            problem=problem,
            awareness_condition=MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL,
            max_iterations=3,
        )
        results_aware.append(result_aware)
        print(
            f"  Success: {result_aware.success} | Iterations: {result_aware.num_iterations} | Tokens: {result_aware.team_total_tokens}"
        )

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL PROBLEMS")
    print("=" * 80)

    # Per-problem summary
    print(f"\n{'Problem':<40} {'Unaware':<15} {'Aware':<15}")
    print("-" * 70)
    for i, (r_u, r_a) in enumerate(zip(results_unaware, results_aware)):
        title = (
            test_problems[i]["question_title"][:37] + "..."
            if len(test_problems[i]["question_title"]) > 40
            else test_problems[i]["question_title"]
        )
        u_result = f"{'✓' if r_u.success else '✗'} ({r_u.num_iterations} iter)"
        a_result = f"{'✓' if r_a.success else '✗'} ({r_a.num_iterations} iter)"
        print(f"{title:<40} {u_result:<15} {a_result:<15}")

    # Aggregate stats
    unaware_successes = sum(1 for r in results_unaware if r.success)
    aware_successes = sum(1 for r in results_aware if r.success)
    unaware_avg_tokens = sum(r.team_total_tokens for r in results_unaware) / len(
        results_unaware
    )
    aware_avg_tokens = sum(r.team_total_tokens for r in results_aware) / len(
        results_aware
    )

    print("\n" + "-" * 70)
    print(
        f"{'TOTALS':<40} {unaware_successes}/{num_problems:<14} {aware_successes}/{num_problems:<14}"
    )
    print(f"{'Avg Tokens':<40} {unaware_avg_tokens:<15.0f} {aware_avg_tokens:<15.0f}")


if __name__ == "__main__":
    asyncio.run(main())
