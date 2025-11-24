"""Part 2: Code Review Pilot Study - Budget Awareness in Multi-Agent Teams.

Tests how budget awareness affects Coder-Reviewer team performance on
LiveCodeBench problems.

Design:
- 2 awareness conditions: NO_AWARENESS vs OVERALL_AND_INDIVIDUAL
- 2 difficulty levels: MEDIUM vs HARD
- 20 problems per cell = 80 total trials
- Sufficient for bootstrap confidence intervals

Sample size rationale:
- 20 per cell gives ~80% power to detect medium effect (d=0.5)
- Bootstrap with 10,000 resamples provides stable CIs
- Can detect ~15pp difference with reasonable confidence

Usage:
    python -m experiments.part2_multi_agent.run_code_review_pilot
"""

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv

from agent_budget.code_review_runner import run_code_review_trial
from agent_budget.core import MultiAgentAwarenessCondition


@dataclass
class PilotConfig:
    """Configuration for pilot study."""

    # Sample sizes
    problems_per_cell: int = 20  # 20 per cell in 2x2 design = 80 total

    # Conditions
    awareness_conditions: list[str] = field(
        default_factory=lambda: ["NO_AWARENESS", "OVERALL_AND_INDIVIDUAL"]
    )
    difficulties: list[str] = field(default_factory=lambda: ["medium", "hard"])

    # Execution
    max_iterations: int = 3
    random_seed: int = 42

    # Data filtering
    min_contest_date: str = "2025-02-01"  # After model cutoff


@dataclass
class PilotResults:
    """Results from pilot study."""

    config: dict[str, Any]
    trials: list[dict[str, Any]]
    started_at: str
    completed_at: str | None = None

    # Summary stats (computed after completion)
    total_trials: int = 0
    successful_trials: int = 0
    failed_trials: int = 0


def load_problems(config: PilotConfig) -> dict[str, list[dict[str, Any]]]:
    """Load and stratify LiveCodeBench problems.

    Returns:
        Dict mapping difficulty -> list of problems
    """
    print("Loading LiveCodeBench dataset...")
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="release_v6",
        trust_remote_code=True,
    )

    # Filter by date
    cutoff = datetime.fromisoformat(config.min_contest_date)
    filtered = []
    for p in dataset:
        contest_date = datetime.fromisoformat(p["contest_date"])
        if contest_date >= cutoff:
            filtered.append(p)

    print(f"Found {len(filtered)} problems after {config.min_contest_date}")

    # Stratify by difficulty
    problems_by_difficulty: dict[str, list[dict[str, Any]]] = {
        d: [] for d in config.difficulties
    }
    for p in filtered:
        diff = p.get("difficulty", "unknown")
        if diff in problems_by_difficulty:
            problems_by_difficulty[diff].append(p)

    for diff, probs in problems_by_difficulty.items():
        print(f"  {diff.upper()}: {len(probs)} problems")

    return problems_by_difficulty


def select_problems(
    problems_by_difficulty: dict[str, list[dict[str, Any]]], config: PilotConfig
) -> list[tuple[dict[str, Any], str]]:
    """Select problems for the pilot with stratified sampling.

    Returns:
        List of (problem, difficulty) tuples
    """
    random.seed(config.random_seed)
    selected = []

    for difficulty in config.difficulties:
        available = problems_by_difficulty[difficulty]
        n_needed = config.problems_per_cell * len(config.awareness_conditions)

        if len(available) < n_needed:
            print(
                f"WARNING: Only {len(available)} {difficulty} problems available, "
                f"need {n_needed}. Using all available."
            )
            sample = available
        else:
            sample = random.sample(available, n_needed)

        for problem in sample:
            selected.append((problem, difficulty))

    random.shuffle(selected)  # Randomize order
    return selected


async def run_pilot(config: PilotConfig | None = None) -> PilotResults:
    """Run the pilot study.

    Args:
        config: Pilot configuration (uses defaults if None)

    Returns:
        PilotResults with all trial data
    """
    if config is None:
        config = PilotConfig()

    print("=" * 80)
    print("PART 2 CODE REVIEW PILOT STUDY")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Problems per cell: {config.problems_per_cell}")
    print(f"  Awareness conditions: {config.awareness_conditions}")
    print(f"  Difficulties: {config.difficulties}")
    print(f"  Max iterations: {config.max_iterations}")
    print(
        f"  Total trials: {config.problems_per_cell * len(config.difficulties) * len(config.awareness_conditions)}"
    )
    print()

    # Load and select problems
    problems_by_difficulty = load_problems(config)
    selected_problems = select_problems(problems_by_difficulty, config)

    print(f"\nSelected {len(selected_problems)} problem instances")

    # Initialize results
    results = PilotResults(
        config=asdict(config),
        trials=[],
        started_at=datetime.now().isoformat(),
    )

    # Map condition strings to enums
    condition_map = {
        "NO_AWARENESS": MultiAgentAwarenessCondition.NO_AWARENESS,
        "OVERALL_AND_INDIVIDUAL": MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL,
    }

    # Run trials
    # Each problem gets tested with each awareness condition
    trial_num = 0
    total_trials = len(selected_problems)

    # Group problems by their identity to test same problem with both conditions
    problem_groups: dict[str, tuple[dict[str, Any], str]] = {}
    for problem, difficulty in selected_problems:
        pid = problem.get("question_id", str(hash(problem["question_title"])))
        problem_groups[pid] = (problem, difficulty)

    # Now assign half to each condition
    problem_ids = list(problem_groups.keys())
    random.seed(config.random_seed)
    random.shuffle(problem_ids)

    # Split problems: each problem tested with ONE condition (between-subjects)
    problems_per_condition = len(problem_ids) // len(config.awareness_conditions)

    # (problem, difficulty, condition_str)
    assignments: list[tuple[dict[str, Any], str, str]] = []
    for i, cond_str in enumerate(config.awareness_conditions):
        start_idx = i * problems_per_condition
        end_idx = start_idx + problems_per_condition
        for pid in problem_ids[start_idx:end_idx]:
            problem, difficulty = problem_groups[pid]
            assignments.append((problem, difficulty, cond_str))

    random.shuffle(assignments)  # Randomize execution order
    total_trials = len(assignments)

    print(f"\nStarting {total_trials} trials...")
    print("=" * 80)

    for problem, difficulty, condition_str in assignments:
        trial_num += 1
        condition = condition_map[condition_str]

        print(f"\n[{trial_num}/{total_trials}] {problem['question_title'][:50]}...")
        print(f"  Difficulty: {difficulty} | Condition: {condition_str}")

        start_time = time.time()
        try:
            trial_result = await run_code_review_trial(
                problem=problem,
                awareness_condition=condition,
                max_iterations=config.max_iterations,
            )

            # Convert to dict for storage
            trial_dict = {
                "trial_num": trial_num,
                "problem_id": trial_result.problem_id,
                "problem_title": trial_result.problem_title,
                "difficulty": difficulty,
                "awareness_condition": condition_str,
                "success": trial_result.success,
                "num_iterations": trial_result.num_iterations,
                "final_decision": trial_result.final_decision,
                "coder_tokens": trial_result.coder_tokens,
                "reviewer_tokens": trial_result.reviewer_tokens,
                "team_total_tokens": trial_result.team_total_tokens,
                "duration_seconds": time.time() - start_time,
                "error": None,
            }

            status = "✓" if trial_result.success else "✗"
            print(
                f"  Result: {status} | Iterations: {trial_result.num_iterations} | "
                f"Tokens: {trial_result.team_total_tokens}"
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            trial_dict = {
                "trial_num": trial_num,
                "problem_id": problem.get("question_id", "unknown"),
                "problem_title": problem.get("question_title", "unknown"),
                "difficulty": difficulty,
                "awareness_condition": condition_str,
                "success": False,
                "num_iterations": 0,
                "final_decision": "ERROR",
                "coder_tokens": 0,
                "reviewer_tokens": 0,
                "team_total_tokens": 0,
                "duration_seconds": time.time() - start_time,
                "error": str(e),
            }

        results.trials.append(trial_dict)

        # Save intermediate results every 10 trials
        if trial_num % 10 == 0:
            save_results(results, intermediate=True)

    results.completed_at = datetime.now().isoformat()
    results.total_trials = len(results.trials)
    results.successful_trials = sum(1 for t in results.trials if t["success"])
    results.failed_trials = results.total_trials - results.successful_trials

    return results


def save_results(results: PilotResults, intermediate: bool = False) -> Path:
    """Save results to JSON file.

    Args:
        results: Pilot results to save
        intermediate: If True, saves as intermediate checkpoint

    Returns:
        Path to saved file
    """
    output_dir = Path("experiments/results/part2_code_review_pilot")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_intermediate" if intermediate else ""
    output_file = output_dir / f"pilot_{timestamp}{suffix}.json"

    # Convert to serializable dict
    data = {
        "config": results.config,
        "started_at": results.started_at,
        "completed_at": results.completed_at,
        "total_trials": len(results.trials),
        "successful_trials": sum(1 for t in results.trials if t["success"]),
        "trials": results.trials,
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return output_file


def print_summary(results: PilotResults) -> None:
    """Print summary of pilot results."""
    print("\n" + "=" * 80)
    print("PILOT STUDY SUMMARY")
    print("=" * 80)

    print(f"\nTotal trials: {results.total_trials}")
    print(
        f"Successful: {results.successful_trials} ({results.successful_trials / results.total_trials * 100:.1f}%)"
    )
    print(f"Failed: {results.failed_trials}")

    # By condition
    print("\n" + "-" * 40)
    print("BY AWARENESS CONDITION:")
    print("-" * 40)

    for condition in ["NO_AWARENESS", "OVERALL_AND_INDIVIDUAL"]:
        trials = [t for t in results.trials if t["awareness_condition"] == condition]
        if not trials:
            continue

        successes = sum(1 for t in trials if t["success"])
        avg_iterations = sum(t["num_iterations"] for t in trials) / len(trials)
        avg_tokens = sum(t["team_total_tokens"] for t in trials) / len(trials)

        print(f"\n{condition}:")
        print(
            f"  Success rate: {successes}/{len(trials)} ({successes / len(trials) * 100:.1f}%)"
        )
        print(f"  Avg iterations: {avg_iterations:.2f}")
        print(f"  Avg tokens: {avg_tokens:.0f}")

    # By difficulty
    print("\n" + "-" * 40)
    print("BY DIFFICULTY:")
    print("-" * 40)

    for difficulty in ["medium", "hard"]:
        trials = [t for t in results.trials if t["difficulty"] == difficulty]
        if not trials:
            continue

        successes = sum(1 for t in trials if t["success"])
        avg_iterations = sum(t["num_iterations"] for t in trials) / len(trials)

        print(f"\n{difficulty.upper()}:")
        print(
            f"  Success rate: {successes}/{len(trials)} ({successes / len(trials) * 100:.1f}%)"
        )
        print(f"  Avg iterations: {avg_iterations:.2f}")

    # By condition × difficulty (2×2 design)
    print("\n" + "-" * 40)
    print("BY CONDITION × DIFFICULTY:")
    print("-" * 40)

    print(f"\n{'Condition':<30} {'Medium':<15} {'Hard':<15}")
    print("-" * 60)

    for condition in ["NO_AWARENESS", "OVERALL_AND_INDIVIDUAL"]:
        row = f"{condition:<30}"
        for difficulty in ["medium", "hard"]:
            trials = [
                t
                for t in results.trials
                if t["awareness_condition"] == condition
                and t["difficulty"] == difficulty
            ]
            if trials:
                successes = sum(1 for t in trials if t["success"])
                row += f" {successes}/{len(trials)} ({successes / len(trials) * 100:.0f}%)    "
            else:
                row += " N/A            "
        print(row)


async def main() -> None:
    """Main entry point."""
    load_dotenv()

    config = PilotConfig(
        problems_per_cell=20,  # 20 per cell × 2 conditions × 2 difficulties = 80 total
        random_seed=42,
    )

    results = await run_pilot(config)
    print_summary(results)
    save_results(results)

    print("\n" + "=" * 80)
    print("PILOT STUDY COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run: python -m experiments.part2_multi_agent.analyze_code_review_pilot")
    print("  2. Review bootstrap confidence intervals")
    print("  3. Decide on full study parameters")


if __name__ == "__main__":
    asyncio.run(main())
