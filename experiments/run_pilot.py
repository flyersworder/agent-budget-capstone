"""Pilot study for Part 1: Test budget-controlled allocation strategies.

This script runs a pilot experiment with one task per complexity level to validate
the budget control system before running the full experiment suite.
"""

import asyncio

from agent_budget import AgentFactory, AllocationStrategy, UsageMonitor
from experiments import ExperimentRunner
from experiments.tasks import get_task_by_id


async def run_pilot() -> None:
    """Run pilot study with one task per complexity level."""
    print("=" * 80)
    print("PILOT STUDY: Budget-Controlled Token Allocation")
    print("=" * 80)
    print()
    print("Testing one task per complexity level:")
    print("  - simple_01: 1500 token budget")
    print("  - moderate_01: 3000 token budget")
    print("  - complex_01: 5000 token budget")
    print()
    print("Each task will be run with all three strategies:")
    print("  - DEEP (80% reasoning / 20% output)")
    print("  - BALANCED (50% reasoning / 50% output)")
    print("  - VERBOSE (20% reasoning / 80% output)")
    print()
    print("=" * 80)
    print()

    # Get pilot tasks (one per complexity)
    task_ids = ["simple_01", "moderate_01", "complex_01"]
    tasks = [get_task_by_id(task_id) for task_id in task_ids]

    # Ensure all tasks exist
    if any(task is None for task in tasks):
        print("❌ Error: One or more pilot tasks not found")
        return

    # Type narrowing: filter out None values (we already checked above)
    valid_tasks = [task for task in tasks if task is not None]

    # Initialize components
    factory = AgentFactory()
    monitor = UsageMonitor()
    runner = ExperimentRunner(factory=factory, monitor=monitor)

    # Run experiments
    strategies = list(AllocationStrategy)
    suite = await runner.run_experiment_suite(tasks=valid_tasks, strategies=strategies)

    # Print summary
    print()
    print("=" * 80)
    print("PILOT RESULTS SUMMARY")
    print("=" * 80)
    print()

    successful = suite.get_successful_results()
    within_budget = suite.get_within_budget_results()

    print(f"Total experiments: {len(suite.results)}")
    print(f"Successful: {len(successful)}")
    print(f"Within budget (±10%): {len(within_budget)}")
    print()

    # Budget compliance analysis
    print("Budget Compliance by Strategy:")
    print()

    for strategy in AllocationStrategy:
        strategy_results = [r for r in successful if r.strategy == strategy.value]

        if not strategy_results:
            continue

        within = sum(1 for r in strategy_results if r.within_budget)
        avg_util = sum(r.budget_utilization for r in strategy_results) / len(
            strategy_results
        )

        print(f"{strategy.value.upper()}:")
        print(f"  Within budget: {within}/{len(strategy_results)}")
        print(f"  Avg utilization: {avg_util:.1f}%")

        for result in strategy_results:
            status = "✓" if result.within_budget else "✗"
            print(
                f"    {status} {result.task_id}: "
                f"{result.metrics.total_tokens_used}/{result.budget_limit} tokens "
                f"({result.budget_utilization:.1f}%)"
            )
        print()

    # Check if pilot succeeds
    success_rate = len(successful) / len(suite.results) if suite.results else 0
    budget_rate = len(within_budget) / len(successful) if successful else 0

    print("=" * 80)
    print("PILOT EVALUATION")
    print("=" * 80)
    print()
    print(f"Success rate: {success_rate * 100:.0f}%")
    print(f"Budget compliance rate: {budget_rate * 100:.0f}%")
    print()

    if success_rate >= 0.9 and budget_rate >= 0.7:
        print("✅ PILOT PASSED")
        print()
        print("Budget control is working reasonably well.")
        print("Ready to proceed with full experiment suite.")
        print()
        print("Next step:")
        print("  python -m experiments.run_part1")
    else:
        print("⚠️ PILOT NEEDS REVIEW")
        print()
        if success_rate < 0.9:
            print(f"  - Low success rate ({success_rate * 100:.0f}%)")
        if budget_rate < 0.7:
            print(
                f"  - Many experiments exceeded budget ({budget_rate * 100:.0f}% compliant)"
            )
            print("  - Consider adjusting budget limits or implementation")
        print()
        print("Review results before proceeding to full suite.")

    print()
    print("=" * 80)


def main() -> None:
    """CLI entry point."""
    asyncio.run(run_pilot())


if __name__ == "__main__":
    main()
