"""Part 2: Code Review Study - Budget Awareness in Multi-Agent Teams.

Tests how budget awareness affects Coder-Reviewer team performance on
LiveCodeBench problems.

Design (WITHIN-SUBJECTS):
- 2 awareness conditions: NO_AWARENESS vs OVERALL_AND_INDIVIDUAL
- 2 difficulty levels: EASY vs MEDIUM
- All available problems from LiveCodeBench (31 easy + 39 medium = 70)
- Each problem tested with BOTH conditions = 140 total trials
- Within-subjects allows paired comparisons for more statistical power

Sample size rationale:
- 70 problems × 2 conditions = 140 trials total
- Paired design increases power vs between-subjects
- Bootstrap with 10,000 resamples provides stable CIs
- McNemar's test for paired binary outcomes

Usage:
    python -m experiments.part2_multi_agent.run_code_review_study
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
class StudyConfig:
    """Configuration for code review study."""

    # Sample sizes (within-subjects: each problem tested with both conditions)
    # Set high to use all available problems (31 easy + 39 medium = 70 problems)
    problems_per_difficulty: int = 100  # Will cap at available

    # Conditions
    awareness_conditions: list[str] = field(
        default_factory=lambda: ["NO_AWARENESS", "OVERALL_AND_INDIVIDUAL"]
    )
    difficulties: list[str] = field(default_factory=lambda: ["easy", "medium"])

    # Execution
    max_iterations: int = 3
    random_seed: int = 42

    # Data filtering
    min_contest_date: str = "2025-02-01"  # After model cutoff


@dataclass
class StudyResults:
    """Results from code review study."""

    config: dict[str, Any]
    trials: list[dict[str, Any]]
    started_at: str
    completed_at: str | None = None

    # Summary stats (computed after completion)
    total_trials: int = 0
    successful_trials: int = 0
    failed_trials: int = 0


def load_problems(config: StudyConfig) -> dict[str, list[dict[str, Any]]]:
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
    problems_by_difficulty: dict[str, list[dict[str, Any]]], config: StudyConfig
) -> list[tuple[dict[str, Any], str]]:
    """Select problems for the study with stratified sampling.

    Within-subjects design: selects N unique problems per difficulty.
    Each problem will be tested with both conditions.

    Returns:
        List of (problem, difficulty) tuples
    """
    random.seed(config.random_seed)
    selected = []

    for difficulty in config.difficulties:
        available = problems_by_difficulty[difficulty]
        n_needed = config.problems_per_difficulty

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


async def run_study(config: StudyConfig | None = None) -> StudyResults:
    """Run the code review study.

    Args:
        config: Study configuration (uses defaults if None)

    Returns:
        StudyResults with all trial data
    """
    if config is None:
        config = StudyConfig()

    print("=" * 80)
    print("PART 2 CODE REVIEW STUDY (WITHIN-SUBJECTS)")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Problems per difficulty: {config.problems_per_difficulty}")
    print(f"  Difficulties: {config.difficulties}")
    print(f"  Awareness conditions: {config.awareness_conditions}")
    print(f"  Max iterations: {config.max_iterations}")
    n_unique = config.problems_per_difficulty * len(config.difficulties)
    n_trials = n_unique * len(config.awareness_conditions)
    print(f"  Unique problems: {n_unique}")
    print(
        f"  Total trials: {n_trials} (each problem × {len(config.awareness_conditions)} conditions)"
    )
    print()

    # Load and select problems
    problems_by_difficulty = load_problems(config)
    selected_problems = select_problems(problems_by_difficulty, config)

    print(f"\nSelected {len(selected_problems)} unique problems")

    # Initialize results
    results = StudyResults(
        config=asdict(config),
        trials=[],
        started_at=datetime.now().isoformat(),
    )

    # Map condition strings to enums
    condition_map = {
        "NO_AWARENESS": MultiAgentAwarenessCondition.NO_AWARENESS,
        "OVERALL_AND_INDIVIDUAL": MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL,
    }

    # Run trials - WITHIN-SUBJECTS DESIGN
    # Each problem is tested with BOTH awareness conditions
    # This gives us paired comparisons for more statistical power
    trial_num = 0

    # Deduplicate problems (in case same problem appears multiple times)
    problem_groups: dict[str, tuple[dict[str, Any], str]] = {}
    for problem, difficulty in selected_problems:
        pid = problem.get("question_id", str(hash(problem["question_title"])))
        problem_groups[pid] = (problem, difficulty)

    # Create assignments: each problem × each condition
    # (problem, difficulty, condition_str)
    assignments: list[tuple[dict[str, Any], str, str]] = []
    for pid, (problem, difficulty) in problem_groups.items():
        for cond_str in config.awareness_conditions:
            assignments.append((problem, difficulty, cond_str))

    # Randomize execution order to avoid systematic effects
    random.seed(config.random_seed)
    random.shuffle(assignments)
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
                difficulty=difficulty,
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
                # New failure analysis fields
                "failure_reason": trial_result.failure_reason.value,
                "any_truncation": trial_result.any_truncation,
                # Per-iteration truncation details
                "iteration_truncations": [
                    {
                        "iteration": d.iteration,
                        "is_truncated": d.truncation_info.is_truncated
                        if d.truncation_info
                        else False,
                        "tokens_at_limit": d.truncation_info.tokens_at_limit
                        if d.truncation_info
                        else False,
                        "syntax_valid": d.truncation_info.syntax_valid
                        if d.truncation_info
                        else True,
                        "coder_tokens": d.coder_total_tokens,
                        "failure_reason": d.failure_reason.value,
                    }
                    for d in trial_result.iteration_details
                ],
            }

            # Show status with failure reason
            status = "✓" if trial_result.success else "✗"
            extra_info = ""
            if not trial_result.success:
                extra_info = f" [{trial_result.failure_reason.value}]"
                if trial_result.any_truncation:
                    extra_info += " ⚠️TRUNC"
            print(
                f"  Result: {status}{extra_info} | Iterations: {trial_result.num_iterations} | "
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
                # New failure analysis fields
                "failure_reason": "error",
                "any_truncation": False,
                "iteration_truncations": [],
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


def save_results(results: StudyResults, intermediate: bool = False) -> Path:
    """Save results to JSON file.

    Args:
        results: Study results to save
        intermediate: If True, saves as intermediate checkpoint

    Returns:
        Path to saved file
    """
    output_dir = Path("experiments/results/part2_code_review")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_intermediate" if intermediate else ""
    output_file = output_dir / f"study_{timestamp}{suffix}.json"

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


def print_summary(results: StudyResults) -> None:
    """Print summary of study results."""
    print("\n" + "=" * 80)
    print("STUDY SUMMARY")
    print("=" * 80)

    print(f"\nTotal trials: {results.total_trials}")
    print(
        f"Successful: {results.successful_trials} ({results.successful_trials / results.total_trials * 100:.1f}%)"
    )
    print(f"Failed: {results.failed_trials}")

    # Failure reason breakdown
    print("\n" + "-" * 40)
    print("FAILURE REASON BREAKDOWN:")
    print("-" * 40)

    failure_counts: dict[str, int] = {}
    truncation_count = 0
    for t in results.trials:
        if not t["success"]:
            reason = t.get("failure_reason", "unknown")
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
        if t.get("any_truncation", False):
            truncation_count += 1

    for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    print(f"\nTrials with ANY truncation: {truncation_count}/{results.total_trials}")

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

    for difficulty in ["easy", "medium"]:
        trials = [t for t in results.trials if t["difficulty"] == difficulty]
        if not trials:
            continue

        successes = sum(1 for t in trials if t["success"])
        avg_iterations = sum(t["num_iterations"] for t in trials) / len(trials)
        truncations = sum(1 for t in trials if t.get("any_truncation", False))

        print(f"\n{difficulty.upper()}:")
        print(
            f"  Success rate: {successes}/{len(trials)} ({successes / len(trials) * 100:.1f}%)"
        )
        print(f"  Avg iterations: {avg_iterations:.2f}")
        print(f"  Truncations: {truncations}/{len(trials)}")

    # By condition × difficulty (2×2 design)
    print("\n" + "-" * 40)
    print("BY CONDITION × DIFFICULTY:")
    print("-" * 40)

    print(f"\n{'Condition':<30} {'Easy':<15} {'Medium':<15}")
    print("-" * 60)

    for condition in ["NO_AWARENESS", "OVERALL_AND_INDIVIDUAL"]:
        row = f"{condition:<30}"
        for difficulty in ["easy", "medium"]:
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
    import sys

    load_dotenv()

    # Allow seed override via command line: python -m ... --seed 123
    seed = 42
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        if idx + 1 < len(sys.argv):
            seed = int(sys.argv[idx + 1])
            print(f"Using custom seed: {seed}")

    config = StudyConfig(
        # Uses all available problems (31 easy + 39 medium = 70 × 2 conditions = 140 trials)
        random_seed=seed,
    )

    results = await run_study(config)
    print_summary(results)
    save_results(results)

    print("\n" + "=" * 80)
    print("STUDY COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run: python -m experiments.part2_multi_agent.analyze_code_review_study")
    print("  2. Review paired comparisons and bootstrap CIs")


if __name__ == "__main__":
    asyncio.run(main())
