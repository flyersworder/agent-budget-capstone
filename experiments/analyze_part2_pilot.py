"""Detailed analysis of Part 2 pilot results with improved instructions.

Analyzes:
1. Performance patterns across conditions
2. Budget note protocol usage in NEGOTIATION_AWARENESS
3. Agent communication patterns
4. Token efficiency and iteration behavior
5. Readiness assessment for full study
"""

import json
from pathlib import Path
from typing import Any, cast


def load_latest_pilot() -> dict[str, Any]:
    """Load the most recent pilot results."""
    results_dir = Path("experiments/results/part2_pilot")
    pilot_files = sorted(results_dir.glob("pilot_*.json"))

    if not pilot_files:
        raise FileNotFoundError("No pilot results found")

    latest_file = pilot_files[-1]
    print(f"Analyzing: {latest_file.name}\n")

    with open(latest_file) as f:
        return cast(dict[str, Any], json.load(f))


def analyze_by_condition(results: dict[str, Any]) -> None:
    """Analyze performance grouped by awareness condition."""
    print("=" * 80)
    print("PERFORMANCE BY CONDITION")
    print("=" * 80)

    conditions = [
        "no_awareness",
        "overall_only",
        "overall_and_individual",
        "negotiation_awareness",
    ]

    for condition in conditions:
        trials = [t for t in results["trials"] if t["awareness_condition"] == condition]

        if not trials:
            continue

        # Calculate metrics
        correct = sum(1 for t in trials if t["correctness_score"]["score"] == 1.0)
        partial = sum(1 for t in trials if t["correctness_score"]["score"] == 0.5)
        incorrect = sum(1 for t in trials if t["correctness_score"]["score"] == 0.0)

        avg_iter = sum(t["num_iterations"] for t in trials) / len(trials)
        avg_tokens = sum(t["metrics"]["total_tokens"] for t in trials) / len(trials)

        researcher_tokens = sum(
            t["metrics"]["researcher_tokens"] for t in trials
        ) / len(trials)
        validator_tokens = sum(t["metrics"]["validator_tokens"] for t in trials) / len(
            trials
        )

        approved = sum(1 for t in trials if t["approved"])
        max_reached = sum(1 for t in trials if t["max_iterations_reached"])

        print(f"\n{condition.upper()}:")
        print("  Correctness:")
        print(
            f"    Correct:   {correct}/{len(trials)} ({correct / len(trials) * 100:.1f}%)"
        )
        print(f"    Partial:   {partial}/{len(trials)}")
        print(f"    Incorrect: {incorrect}/{len(trials)}")
        print("  Coordination:")
        print(
            f"    Approved:       {approved}/{len(trials)} ({approved / len(trials) * 100:.1f}%)"
        )
        print(f"    Max iterations: {max_reached}/{len(trials)}")
        print(f"    Avg iterations: {avg_iter:.2f}")
        print("  Token Usage:")
        print(f"    Total avg:      {avg_tokens:.0f}")
        print(f"    Researcher avg: {researcher_tokens:.0f}")
        print(f"    Validator avg:  {validator_tokens:.0f}")


def analyze_negotiation_protocol(results: dict[str, Any]) -> None:
    """Check if NEGOTIATION_AWARENESS trials used the [BUDGET NOTE:...] protocol."""
    print("\n" + "=" * 80)
    print("NEGOTIATION PROTOCOL USAGE")
    print("=" * 80)

    neg_trials = [
        t
        for t in results["trials"]
        if t["awareness_condition"] == "negotiation_awareness"
    ]

    print(f"\nTotal NEGOTIATION_AWARENESS trials: {len(neg_trials)}")

    # Check for budget note markers in outputs
    researcher_notes = 0
    validator_notes = 0

    for trial in neg_trials:
        r_output = trial["researcher_output"]
        v_feedback = trial["validator_feedback"]

        if "[BUDGET NOTE:" in r_output or "[BUDGET ACKNOWLEDGED:" in r_output:
            researcher_notes += 1

        if "[BUDGET NOTE:" in v_feedback or "[BUDGET ACKNOWLEDGED:" in v_feedback:
            validator_notes += 1

    print("\nProtocol usage:")
    print(
        f"  Researcher used budget notes: {researcher_notes}/{len(neg_trials)} ({researcher_notes / len(neg_trials) * 100:.1f}%)"
    )
    print(
        f"  Validator used budget notes:  {validator_notes}/{len(neg_trials)} ({validator_notes / len(neg_trials) * 100:.1f}%)"
    )

    # Show examples
    print("\nExample trials with protocol usage:")
    for i, trial in enumerate(neg_trials[:3], 1):
        r_output = trial["researcher_output"]
        v_feedback = trial["validator_feedback"]

        if "[BUDGET" in r_output or "[BUDGET" in v_feedback:
            print(f"\n  Trial {i} ({trial['trial_id']}):")
            print(f"    Question: {trial['question'][:80]}...")
            print(f"    Approved: {trial['approved']}")
            print(f"    Correct: {trial['correctness_score']['score']}")

            # Show budget note excerpts
            if "[BUDGET NOTE:" in r_output:
                start = r_output.index("[BUDGET NOTE:")
                excerpt = r_output[start : start + 150].replace("\n", " ")
                print(f"    Researcher: {excerpt}...")

            if "[BUDGET" in v_feedback:
                start = v_feedback.index("[BUDGET")
                excerpt = v_feedback[start : start + 150].replace("\n", " ")
                print(f"    Validator: {excerpt}...")


def analyze_communication_patterns(results: dict[str, Any]) -> None:
    """Analyze how agents communicate across conditions."""
    print("\n" + "=" * 80)
    print("COMMUNICATION PATTERNS")
    print("=" * 80)

    conditions = [
        "no_awareness",
        "overall_only",
        "overall_and_individual",
        "negotiation_awareness",
    ]

    for condition in conditions:
        trials = [t for t in results["trials"] if t["awareness_condition"] == condition]

        if not trials:
            continue

        # Measure output lengths as proxy for communication detail
        avg_researcher_len = sum(len(t["researcher_output"]) for t in trials) / len(
            trials
        )
        avg_validator_len = sum(len(t["validator_feedback"]) for t in trials) / len(
            trials
        )

        # Count multi-iteration trials (more back-and-forth)
        multi_iter = sum(1 for t in trials if t["num_iterations"] > 1)

        print(f"\n{condition.upper()}:")
        print(f"  Avg researcher output length: {avg_researcher_len:.0f} chars")
        print(f"  Avg validator feedback length: {avg_validator_len:.0f} chars")
        print(
            f"  Multi-iteration trials: {multi_iter}/{len(trials)} ({multi_iter / len(trials) * 100:.1f}%)"
        )


def show_example_trials(results: dict[str, Any]) -> None:
    """Show detailed examples of interesting trials."""
    print("\n" + "=" * 80)
    print("EXAMPLE TRIALS")
    print("=" * 80)

    # Find one correct and one incorrect from each condition
    conditions = ["no_awareness", "negotiation_awareness"]

    for condition in conditions:
        trials = [t for t in results["trials"] if t["awareness_condition"] == condition]

        correct_trial = next(
            (t for t in trials if t["correctness_score"]["score"] == 1.0), None
        )
        incorrect_trial = next(
            (t for t in trials if t["correctness_score"]["score"] == 0.0), None
        )

        print(f"\n{condition.upper()} - Correct Example:")
        if correct_trial:
            print(f"  Question: {correct_trial['question']}")
            print(f"  Ground truth: {correct_trial['ground_truth']}")
            print(f"  Researcher: {correct_trial['researcher_output'][:200]}...")
            print(f"  Iterations: {correct_trial['num_iterations']}")
            print(f"  Approved: {correct_trial['approved']}")
        else:
            print("  (No correct trials)")

        print(f"\n{condition.upper()} - Incorrect Example:")
        if incorrect_trial:
            print(f"  Question: {incorrect_trial['question']}")
            print(f"  Ground truth: {incorrect_trial['ground_truth']}")
            print(f"  Researcher: {incorrect_trial['researcher_output'][:200]}...")
            print(f"  Validator: {incorrect_trial['validator_feedback'][:200]}...")
            print(f"  Iterations: {incorrect_trial['num_iterations']}")
            print(f"  Approved: {incorrect_trial['approved']}")
        else:
            print("  (No incorrect trials)")


def readiness_assessment(results: dict[str, Any]) -> None:
    """Assess whether the experiment is ready for full-scale study."""
    print("\n" + "=" * 80)
    print("READINESS ASSESSMENT")
    print("=" * 80)

    # Check overall correctness range
    scores = [t["correctness_score"]["score"] for t in results["trials"]]
    avg_score = sum(scores) / len(scores)

    print("\nMeasurement Quality:")
    print(f"  Overall accuracy: {avg_score * 100:.1f}%")
    print(f"  Range: {min(scores) * 100:.0f}% - {max(scores) * 100:.0f}%")

    if 0.3 <= avg_score <= 0.7:
        print("  ✅ Good measurement range (30-70%)")
    else:
        print("  ⚠️  Outside ideal range (prefer 30-70%)")

    # Check variance across conditions
    conditions = [
        "no_awareness",
        "overall_only",
        "overall_and_individual",
        "negotiation_awareness",
    ]
    condition_scores = []

    for condition in conditions:
        trials = [t for t in results["trials"] if t["awareness_condition"] == condition]
        if trials:
            avg = sum(t["correctness_score"]["score"] for t in trials) / len(trials)
            condition_scores.append((condition, avg))

    max_diff = max(s[1] for s in condition_scores) - min(s[1] for s in condition_scores)

    print("\nCondition Differences:")
    for condition, score in condition_scores:
        print(f"  {condition}: {score * 100:.1f}%")
    print(f"  Max difference: {max_diff * 100:.1f} percentage points")

    if max_diff >= 0.15:
        print("  ✅ Observable differences (≥15pp)")
    else:
        print("  ⚠️  Small differences (<15pp) - may need larger sample")

    # Check coordination success
    approved_rate = sum(1 for t in results["trials"] if t["approved"]) / len(
        results["trials"]
    )

    print("\nCoordination Quality:")
    print(f"  Overall approval rate: {approved_rate * 100:.1f}%")

    if approved_rate >= 0.8:
        print("  ✅ High convergence rate")
    else:
        print("  ⚠️  Lower convergence - agents may need more iterations")

    # Overall recommendation
    print(f"\n{'=' * 80}")
    print("RECOMMENDATION:")
    print("=" * 80)

    if 0.3 <= avg_score <= 0.7 and max_diff >= 0.15 and approved_rate >= 0.8:
        print("✅ Ready for full study")
        print("   - Good measurement quality")
        print("   - Observable condition differences")
        print("   - High coordination success")
        print(
            "\nSuggested sample size: 20-30 questions per condition (80-120 total trials)"
        )
    else:
        issues = []
        if not (0.3 <= avg_score <= 0.7):
            issues.append("Adjust difficulty further")
        if max_diff < 0.15:
            issues.append("Consider stronger manipulations or larger sample")
        if approved_rate < 0.8:
            issues.append("Consider increasing max iterations")

        print("⚠️  Consider adjustments:")
        for issue in issues:
            print(f"   - {issue}")


def main() -> None:
    """Run full analysis."""
    results = load_latest_pilot()

    analyze_by_condition(results)
    analyze_negotiation_protocol(results)
    analyze_communication_patterns(results)
    show_example_trials(results)
    readiness_assessment(results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
