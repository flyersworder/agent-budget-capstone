"""Main experiment runner for Part 1: Single-Agent Strategic Allocation.

This script runs experiments comparing three token allocation strategies:
- Deep Thinker (80/20): More reasoning, less output
- Balanced (50/50): Equal allocation
- Verbose (20/80): Less reasoning, more output

Usage:
    python -m experiments.run_part1 [--budget BUDGET] [--test] [--task-id TASK_ID]

Examples:
    # Run all experiments
    python -m experiments.run_part1

    # Run with smaller budget for testing
    python -m experiments.run_part1 --budget 5000

    # Test mode: run only one task per strategy
    python -m experiments.run_part1 --test

    # Run specific task
    python -m experiments.run_part1 --task-id simple_01
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from agent_budget import AgentFactory, AllocationStrategy, UsageMonitor
from experiments import ExperimentRunner, LLMResponseEvaluator
from experiments.runner import ExperimentResult
from experiments.tasks import get_all_tasks, get_task_by_id


async def run_experiments(
    total_budget: int = 10000,
    test_mode: bool = False,
    task_id: str | None = None,
    output_dir: str = "experiments/results",
) -> None:
    """Run Part 1 experiments.

    Args:
        total_budget: Total token budget per experiment
        test_mode: If True, run only one task per strategy for testing
        task_id: If specified, run only this specific task
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    factory = AgentFactory()
    monitor = UsageMonitor()
    runner = ExperimentRunner(factory=factory, monitor=monitor)
    evaluator = LLMResponseEvaluator()

    # Get tasks to run
    if task_id:
        task = get_task_by_id(task_id)
        if not task:
            print(f"❌ Task '{task_id}' not found")
            return
        tasks = [task]
        print(f"Running single task: {task.id}")
    elif test_mode:
        # In test mode, run only the first simple task
        task = get_task_by_id("simple_01")
        if not task:
            print("❌ Test task 'simple_01' not found")
            return
        tasks = [task]
        print("🧪 Test mode: Running one task per strategy")
    else:
        tasks = get_all_tasks()
        print(f"Running all {len(tasks)} tasks")

    # Strategies to test
    strategies = list(AllocationStrategy)

    print(f"\nBudget: {total_budget} tokens per experiment")
    print(f"Strategies: {[s.value for s in strategies]}")
    print(f"Tasks: {len(tasks)}")
    print(f"Total experiments: {len(strategies) * len(tasks)}")
    print(f"\n{'=' * 60}\n")

    # Run experiment suite
    suite = await runner.run_experiment_suite(
        tasks=tasks, strategies=strategies, total_budget=total_budget
    )

    # Evaluate responses using LLM-based pairwise comparison
    print(f"\n{'=' * 60}")
    print("Evaluating response quality with LLM-as-a-Judge...")
    print(f"{'=' * 60}\n")

    evaluated_results = []

    # Group results by task for pairwise comparison
    task_groups: dict[str, dict[str, ExperimentResult]] = {}
    for result in suite.get_successful_results():
        if result.task_id not in task_groups:
            task_groups[result.task_id] = {}
        task_groups[result.task_id][result.strategy] = result

    # Evaluate each task's responses via pairwise comparison
    for task_id, task_results in task_groups.items():
        # Get the task
        task = get_task_by_id(task_id)
        if not task:
            continue

        # Prepare responses for ranking
        responses = {
            strategy_name: result.response
            for strategy_name, result in task_results.items()
        }

        # Only evaluate if we have all three strategies
        if len(responses) != 3:
            print(
                f"⚠️  Skipping {task_id}: missing strategies (have {list(responses.keys())})"
            )
            continue

        # Rank strategies using pairwise comparison with position bias mitigation
        print(f"Evaluating {task_id}...")
        strategy_scores = evaluator.rank_strategies(responses, task)

        # Combine metrics with quality scores
        for strategy_name, result in task_results.items():
            quality_score = strategy_scores[strategy_name]

            evaluated_result = {
                "task_id": result.task_id,
                "strategy": result.strategy,
                "question": result.question,
                "response_preview": result.response[:200] + "..."
                if len(result.response) > 200
                else result.response,
                "response_length": len(result.response),
                **result.metrics.to_dict(),
                "quality_accuracy": quality_score.accuracy,
                "quality_completeness": quality_score.completeness,
                "quality_clarity": quality_score.clarity,
                "quality_depth": quality_score.depth,
                "quality_conciseness": quality_score.conciseness,
                "quality_overall": quality_score.overall,
            }

            evaluated_results.append(evaluated_result)

            print(
                f"  {strategy_name.upper()}: Overall {quality_score.overall:.2f}/5 "
                f"(A:{quality_score.accuracy:.2f} "
                f"Co:{quality_score.completeness:.2f} "
                f"Cl:{quality_score.clarity:.2f} "
                f"D:{quality_score.depth:.2f} "
                f"Cn:{quality_score.conciseness:.2f})"
            )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON (detailed)
    json_path = Path(output_dir) / f"part1_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "total_budget": total_budget,
                    "test_mode": test_mode,
                    "num_tasks": len(tasks),
                    "num_strategies": len(strategies),
                    "started_at": suite.started_at,
                    "completed_at": suite.completed_at,
                },
                "results": evaluated_results,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print("📊 Results Summary")
    print(f"{'=' * 60}\n")

    # Print summary statistics
    for strategy in strategies:
        strategy_results = [
            r for r in evaluated_results if r["strategy"] == strategy.value
        ]

        if not strategy_results:
            continue

        avg_quality = sum(r["quality_overall"] for r in strategy_results) / len(
            strategy_results
        )
        avg_tokens = sum(r["total_tokens_used"] for r in strategy_results) / len(
            strategy_results
        )
        avg_tool_calls = sum(r["total_tool_calls"] for r in strategy_results) / len(
            strategy_results
        )
        avg_duration = sum(r["duration_seconds"] for r in strategy_results) / len(
            strategy_results
        )

        print(f"{strategy.value.upper()} Strategy:")
        print(f"  Quality Score:     {avg_quality:.2f}/5.00")
        print(f"  Avg Tokens Used:   {avg_tokens:.0f}/{total_budget}")
        print(f"  Avg Tool Calls:    {avg_tool_calls:.1f}")
        print(f"  Avg Duration:      {avg_duration:.1f}s")
        print()

    print(f"✅ Results saved to: {json_path}")
    print(f"\n{'=' * 60}\n")

    # Print next steps
    print("Next Steps:")
    print("  1. Review results in the JSON file")
    print("  2. Run full experiment suite: python -m experiments.run_part1")
    print("  3. Create visualizations with notebooks/part1_analysis.ipynb")
    print()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Part 1 token allocation experiments"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10000,
        help="Total token budget per experiment (default: 10000)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: run only one task per strategy",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        help="Run specific task by ID (e.g., 'simple_01')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Output directory for results (default: experiments/results)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_experiments(
            total_budget=args.budget,
            test_mode=args.test,
            task_id=args.task_id,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
