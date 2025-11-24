"""Analysis of Part 2 Code Review Pilot with Bootstrap Confidence Intervals.

Provides statistical analysis including:
1. Success rates by condition and difficulty
2. Bootstrap confidence intervals for effect sizes
3. Paired analysis with McNemar's test (within-subjects design)
4. Interaction effects (condition × difficulty)
5. Token usage patterns
6. Direction of effect assessment

Usage:
    python -m experiments.part2_multi_agent.analyze_code_review_pilot
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np


def load_latest_results() -> dict[str, Any]:
    """Load the most recent pilot results."""
    results_dir = Path("experiments/results/part2_code_review_pilot")

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find non-intermediate files
    pilot_files = sorted(
        [f for f in results_dir.glob("pilot_*.json") if "intermediate" not in f.name]
    )

    if not pilot_files:
        # Fall back to intermediate if no final
        pilot_files = sorted(results_dir.glob("pilot_*.json"))

    if not pilot_files:
        raise FileNotFoundError("No pilot results found")

    latest_file = pilot_files[-1]
    print(f"Loading: {latest_file.name}\n")

    with open(latest_file) as f:
        return cast(dict[str, Any], json.load(f))


def bootstrap_ci(
    data: Sequence[float | int],
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: Sample data
        statistic: "mean" or "proportion"
        n_bootstrap: Number of bootstrap resamples
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    if not data:
        return (0.0, 0.0, 0.0)

    rng = np.random.default_rng(seed)
    data_array = np.array(data)
    n = len(data_array)

    # Point estimate
    if statistic == "mean":
        point_est = float(np.mean(data_array))
    else:  # proportion
        point_est = float(np.mean(data_array))

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = rng.choice(data_array, size=n, replace=True)
        bootstrap_stats.append(float(np.mean(resample)))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method for CI
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_stats, alpha / 2 * 100))
    ci_upper = float(np.percentile(bootstrap_stats, (1 - alpha / 2) * 100))

    return (point_est, ci_lower, ci_upper)


def bootstrap_difference_ci(
    data1: Sequence[float | int],
    data2: Sequence[float | int],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap CI for difference between two groups.

    Args:
        data1: First group data
        data2: Second group data
        n_bootstrap: Number of bootstrap resamples
        ci_level: Confidence level
        seed: Random seed

    Returns:
        Tuple of (point_estimate_diff, ci_lower, ci_upper)
    """
    if not data1 or not data2:
        return (0.0, 0.0, 0.0)

    rng = np.random.default_rng(seed)
    arr1 = np.array(data1)
    arr2 = np.array(data2)

    # Point estimate
    point_diff = float(np.mean(arr1) - np.mean(arr2))

    # Bootstrap
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        resample1 = rng.choice(arr1, size=len(arr1), replace=True)
        resample2 = rng.choice(arr2, size=len(arr2), replace=True)
        bootstrap_diffs.append(float(np.mean(resample1) - np.mean(resample2)))

    bootstrap_diffs = np.array(bootstrap_diffs)

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_diffs, alpha / 2 * 100))
    ci_upper = float(np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100))

    return (point_diff, ci_lower, ci_upper)


def mcnemars_test(
    paired_outcomes: list[tuple[int, int]],
) -> tuple[float, float, str]:
    """McNemar's test for paired binary outcomes.

    For within-subjects design where same problem tested under both conditions.

    Args:
        paired_outcomes: List of (unaware_success, aware_success) tuples
                        where 1=success, 0=failure

    Returns:
        Tuple of (test_statistic, p_value, interpretation)
    """
    # Build contingency table
    # b = unaware fails, aware succeeds (awareness helps)
    # c = unaware succeeds, aware fails (awareness hurts)
    b = 0  # Discordant: unaware=0, aware=1
    c = 0  # Discordant: unaware=1, aware=0

    for unaware, aware in paired_outcomes:
        if unaware == 0 and aware == 1:
            b += 1
        elif unaware == 1 and aware == 0:
            c += 1

    # McNemar's test statistic (with continuity correction)
    if b + c == 0:
        return (0.0, 1.0, "No discordant pairs - cannot compute")

    # Exact binomial test for small samples
    n_discordant = b + c

    if n_discordant < 25:
        # Use exact binomial test
        from math import comb

        # Two-tailed p-value: P(X <= min(b,c)) + P(X >= max(b,c))
        k = min(b, c)
        p_value = 0.0
        for i in range(k + 1):
            p_value += comb(n_discordant, i) * (0.5**n_discordant)
        p_value *= 2  # Two-tailed
        p_value = min(p_value, 1.0)  # Cap at 1

        test_stat = float(b - c)
    else:
        # Chi-squared approximation with continuity correction
        test_stat = ((abs(b - c) - 1) ** 2) / (b + c)
        # Approximate p-value from chi-squared(1)
        # Using Wilson-Hilferty approximation
        z = np.sqrt(test_stat)
        p_value = 2 * (1 - 0.5 * (1 + np.math.erf(z / np.sqrt(2))))

    # Interpretation
    if b > c:
        direction = "Awareness HELPS"
    elif c > b:
        direction = "Awareness HURTS"
    else:
        direction = "No difference"

    interpretation = f"{direction} (b={b}, c={c})"

    return (test_stat, p_value, interpretation)


def analyze_paired_effects(results: dict[str, Any]) -> None:
    """Analyze paired effects using McNemar's test (within-subjects design).

    Groups trials by problem and compares outcomes across conditions.
    """
    print("\n" + "=" * 80)
    print("PAIRED ANALYSIS (WITHIN-SUBJECTS DESIGN)")
    print("=" * 80)

    trials = results["trials"]

    # Group by problem_id
    problems: dict[str, dict[str, dict[str, Any]]] = {}
    for t in trials:
        pid = t["problem_id"]
        cond = t["awareness_condition"]
        if pid not in problems:
            problems[pid] = {}
        problems[pid][cond] = t

    # Build paired outcomes
    paired_outcomes: list[tuple[int, int]] = []
    paired_details: list[dict[str, Any]] = []

    for pid, conditions in problems.items():
        if "NO_AWARENESS" in conditions and "OVERALL_AND_INDIVIDUAL" in conditions:
            unaware_success = 1 if conditions["NO_AWARENESS"]["success"] else 0
            aware_success = 1 if conditions["OVERALL_AND_INDIVIDUAL"]["success"] else 0
            paired_outcomes.append((unaware_success, aware_success))
            paired_details.append(
                {
                    "problem_id": pid,
                    "difficulty": conditions["NO_AWARENESS"]["difficulty"],
                    "unaware": unaware_success,
                    "aware": aware_success,
                }
            )

    n_pairs = len(paired_outcomes)
    print(f"\nPaired problems: {n_pairs}")

    if n_pairs == 0:
        print("No paired data available for analysis")
        return

    # Concordance table
    both_success = sum(1 for u, a in paired_outcomes if u == 1 and a == 1)
    both_fail = sum(1 for u, a in paired_outcomes if u == 0 and a == 0)
    unaware_only = sum(1 for u, a in paired_outcomes if u == 1 and a == 0)
    aware_only = sum(1 for u, a in paired_outcomes if u == 0 and a == 1)

    print("\nConcordance Table:")
    print("                    Aware Success    Aware Fail")
    print(f"  Unaware Success:      {both_success:3d}             {unaware_only:3d}")
    print(f"  Unaware Fail:         {aware_only:3d}             {both_fail:3d}")

    print(f"\n  Concordant pairs: {both_success + both_fail} (both same outcome)")
    print(f"  Discordant pairs: {unaware_only + aware_only}")
    print(f"    - Awareness helped: {aware_only} (unaware failed, aware succeeded)")
    print(f"    - Awareness hurt: {unaware_only} (unaware succeeded, aware failed)")

    # McNemar's test
    stat, p_value, interpretation = mcnemars_test(paired_outcomes)

    print("\nMcNemar's Test:")
    print(f"  Test statistic: {stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Interpretation: {interpretation}")

    if p_value < 0.05:
        print("  → Statistically significant (p < 0.05)")
    else:
        print("  → Not statistically significant (p >= 0.05)")

    # By difficulty
    print("\nPaired Analysis by Difficulty:")
    for difficulty in ["medium", "hard"]:
        diff_pairs = [
            (d["unaware"], d["aware"])
            for d in paired_details
            if d["difficulty"] == difficulty
        ]
        if diff_pairs:
            stat, p_value, interp = mcnemars_test(diff_pairs)
            aware_helps = sum(1 for u, a in diff_pairs if u == 0 and a == 1)
            aware_hurts = sum(1 for u, a in diff_pairs if u == 1 and a == 0)
            print(
                f"  {difficulty.upper()}: n={len(diff_pairs)}, "
                f"helps={aware_helps}, hurts={aware_hurts}, p={p_value:.3f}"
            )


def analyze_main_effects(results: dict[str, Any]) -> None:
    """Analyze main effects of awareness condition."""
    print("=" * 80)
    print("MAIN EFFECT: AWARENESS CONDITION")
    print("=" * 80)

    trials = results["trials"]

    # Get success rates by condition
    unaware_successes = [
        1 if t["success"] else 0
        for t in trials
        if t["awareness_condition"] == "NO_AWARENESS"
    ]
    aware_successes = [
        1 if t["success"] else 0
        for t in trials
        if t["awareness_condition"] == "OVERALL_AND_INDIVIDUAL"
    ]

    # Bootstrap CIs
    unaware_ci = bootstrap_ci(unaware_successes, statistic="proportion")
    aware_ci = bootstrap_ci(aware_successes, statistic="proportion")
    diff_ci = bootstrap_difference_ci(aware_successes, unaware_successes)

    print("\nSuccess Rates:")
    print(
        f"  NO_AWARENESS:           {unaware_ci[0] * 100:5.1f}% "
        f"[{unaware_ci[1] * 100:.1f}%, {unaware_ci[2] * 100:.1f}%] (n={len(unaware_successes)})"
    )
    print(
        f"  OVERALL_AND_INDIVIDUAL: {aware_ci[0] * 100:5.1f}% "
        f"[{aware_ci[1] * 100:.1f}%, {aware_ci[2] * 100:.1f}%] (n={len(aware_successes)})"
    )

    print("\nDifference (Aware - Unaware):")
    print(f"  Point estimate: {diff_ci[0] * 100:+.1f} percentage points")
    print(f"  95% CI: [{diff_ci[1] * 100:+.1f}pp, {diff_ci[2] * 100:+.1f}pp]")

    # Interpret
    if diff_ci[1] > 0:
        print("\n  → Awareness HELPS (CI entirely above 0)")
    elif diff_ci[2] < 0:
        print("\n  → Awareness HURTS (CI entirely below 0)")
    else:
        print("\n  → Effect UNCERTAIN (CI crosses 0)")

    # Direction assessment
    print("\nDirection Assessment:")
    if diff_ci[0] > 0:
        print(f"  Direction: Awareness appears to HELP (+{diff_ci[0] * 100:.1f}pp)")
    else:
        print(f"  Direction: Awareness appears to HURT ({diff_ci[0] * 100:.1f}pp)")


def analyze_difficulty_moderation(results: dict[str, Any]) -> None:
    """Analyze whether difficulty moderates the awareness effect."""
    print("\n" + "=" * 80)
    print("MODERATION: DIFFICULTY × AWARENESS")
    print("=" * 80)

    trials = results["trials"]

    # 2×2 breakdown
    cells = {}
    for condition in ["NO_AWARENESS", "OVERALL_AND_INDIVIDUAL"]:
        for difficulty in ["medium", "hard"]:
            cell_trials = [
                t
                for t in trials
                if t["awareness_condition"] == condition
                and t["difficulty"] == difficulty
            ]
            successes = [1 if t["success"] else 0 for t in cell_trials]
            cells[(condition, difficulty)] = successes

    # Print 2×2 table with CIs
    print("\nSuccess Rates by Cell (with 95% CIs):")
    print("-" * 70)
    print(f"{'Condition':<30} {'Medium':<20} {'Hard':<20}")
    print("-" * 70)

    for condition in ["NO_AWARENESS", "OVERALL_AND_INDIVIDUAL"]:
        row = f"{condition:<30}"
        for difficulty in ["medium", "hard"]:
            data = cells.get((condition, difficulty), [])
            if data:
                ci = bootstrap_ci(data)
                row += f" {ci[0] * 100:4.0f}% [{ci[1] * 100:.0f}-{ci[2] * 100:.0f}%]   "
            else:
                row += " N/A                "
        print(row)

    # Compute awareness effect by difficulty
    print("\nAwareness Effect by Difficulty:")

    for difficulty in ["medium", "hard"]:
        unaware = cells.get(("NO_AWARENESS", difficulty), [])
        aware = cells.get(("OVERALL_AND_INDIVIDUAL", difficulty), [])

        if unaware and aware:
            diff_ci = bootstrap_difference_ci(aware, unaware)
            sig = ""
            if diff_ci[1] > 0:
                sig = " *"
            elif diff_ci[2] < 0:
                sig = " *"

            print(
                f"  {difficulty.upper()}: {diff_ci[0] * 100:+5.1f}pp "
                f"[{diff_ci[1] * 100:+.1f}, {diff_ci[2] * 100:+.1f}]{sig}"
            )

    # Interaction: Is the awareness effect different for hard vs medium?
    print("\nInteraction Test:")
    medium_effect = bootstrap_difference_ci(
        cells.get(("OVERALL_AND_INDIVIDUAL", "medium"), []),
        cells.get(("NO_AWARENESS", "medium"), []),
    )
    hard_effect = bootstrap_difference_ci(
        cells.get(("OVERALL_AND_INDIVIDUAL", "hard"), []),
        cells.get(("NO_AWARENESS", "hard"), []),
    )

    interaction = hard_effect[0] - medium_effect[0]
    print(f"  Effect on HARD - Effect on MEDIUM = {interaction * 100:+.1f}pp")

    if abs(hard_effect[0]) > abs(medium_effect[0]):
        print("  → Awareness effect is STRONGER on hard problems")
    else:
        print("  → Awareness effect is STRONGER on medium problems")


def analyze_token_usage(results: dict[str, Any]) -> None:
    """Analyze token usage patterns."""
    print("\n" + "=" * 80)
    print("TOKEN USAGE ANALYSIS")
    print("=" * 80)

    trials = results["trials"]

    for condition in ["NO_AWARENESS", "OVERALL_AND_INDIVIDUAL"]:
        cond_trials = [t for t in trials if t["awareness_condition"] == condition]
        if not cond_trials:
            continue

        tokens = [t["team_total_tokens"] for t in cond_trials]
        coder_tokens = [t["coder_tokens"] for t in cond_trials]
        reviewer_tokens = [t["reviewer_tokens"] for t in cond_trials]

        tokens_ci = bootstrap_ci(tokens, statistic="mean")
        coder_ci = bootstrap_ci(coder_tokens, statistic="mean")
        reviewer_ci = bootstrap_ci(reviewer_tokens, statistic="mean")

        print(f"\n{condition}:")
        print(
            f"  Total tokens:    {tokens_ci[0]:,.0f} [{tokens_ci[1]:,.0f}, {tokens_ci[2]:,.0f}]"
        )
        print(
            f"  Coder tokens:    {coder_ci[0]:,.0f} [{coder_ci[1]:,.0f}, {coder_ci[2]:,.0f}]"
        )
        print(
            f"  Reviewer tokens: {reviewer_ci[0]:,.0f} [{reviewer_ci[1]:,.0f}, {reviewer_ci[2]:,.0f}]"
        )

    # Token difference
    unaware_tokens = [
        t["team_total_tokens"]
        for t in trials
        if t["awareness_condition"] == "NO_AWARENESS"
    ]
    aware_tokens = [
        t["team_total_tokens"]
        for t in trials
        if t["awareness_condition"] == "OVERALL_AND_INDIVIDUAL"
    ]

    if unaware_tokens and aware_tokens:
        diff_ci = bootstrap_difference_ci(aware_tokens, unaware_tokens)
        print("\nToken Difference (Aware - Unaware):")
        print(f"  {diff_ci[0]:+,.0f} tokens [{diff_ci[1]:+,.0f}, {diff_ci[2]:+,.0f}]")


def analyze_iterations(results: dict[str, Any]) -> None:
    """Analyze iteration patterns."""
    print("\n" + "=" * 80)
    print("ITERATION ANALYSIS")
    print("=" * 80)

    trials = results["trials"]

    for condition in ["NO_AWARENESS", "OVERALL_AND_INDIVIDUAL"]:
        cond_trials = [t for t in trials if t["awareness_condition"] == condition]
        if not cond_trials:
            continue

        iterations = [t["num_iterations"] for t in cond_trials]
        iter_ci = bootstrap_ci(iterations, statistic="mean")

        # First-attempt success rate
        first_attempt = [
            1 if t["num_iterations"] == 1 and t["success"] else 0 for t in cond_trials
        ]
        first_ci = bootstrap_ci(first_attempt, statistic="proportion")

        print(f"\n{condition}:")
        print(
            f"  Avg iterations: {iter_ci[0]:.2f} [{iter_ci[1]:.2f}, {iter_ci[2]:.2f}]"
        )
        print(
            f"  First-attempt success: {first_ci[0] * 100:.0f}% [{first_ci[1] * 100:.0f}%, {first_ci[2] * 100:.0f}%]"
        )

        # Distribution
        iter_dist = {1: 0, 2: 0, 3: 0}
        for t in cond_trials:
            iters = min(t["num_iterations"], 3)
            iter_dist[iters] = iter_dist.get(iters, 0) + 1

        print(
            f"  Iteration distribution: 1={iter_dist[1]}, 2={iter_dist[2]}, 3={iter_dist[3]}"
        )


def power_analysis(results: dict[str, Any]) -> None:
    """Estimate power and sample size recommendations."""
    print("\n" + "=" * 80)
    print("POWER ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)

    trials = results["trials"]

    # Get observed effect size
    unaware = [
        1 if t["success"] else 0
        for t in trials
        if t["awareness_condition"] == "NO_AWARENESS"
    ]
    aware = [
        1 if t["success"] else 0
        for t in trials
        if t["awareness_condition"] == "OVERALL_AND_INDIVIDUAL"
    ]

    if not unaware or not aware:
        print("Insufficient data for power analysis")
        return

    p1 = np.mean(unaware)
    p2 = np.mean(aware)
    diff = abs(p2 - p1)

    # Pooled standard deviation for effect size
    pooled_p = (sum(unaware) + sum(aware)) / (len(unaware) + len(aware))
    pooled_sd = np.sqrt(pooled_p * (1 - pooled_p))

    if pooled_sd > 0:
        cohens_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    else:
        cohens_h = 0

    print("\nObserved Effect:")
    print(f"  Unaware success rate: {p1 * 100:.1f}%")
    print(f"  Aware success rate: {p2 * 100:.1f}%")
    print(f"  Difference: {diff * 100:.1f} percentage points")
    print(f"  Cohen's h: {abs(cohens_h):.3f}")

    # Effect size interpretation
    if abs(cohens_h) < 0.2:
        effect_label = "small"
    elif abs(cohens_h) < 0.5:
        effect_label = "small-medium"
    elif abs(cohens_h) < 0.8:
        effect_label = "medium"
    else:
        effect_label = "large"
    print(f"  Effect size: {effect_label}")

    # Sample size recommendations
    print("\nSample Size Recommendations:")
    print("  For 80% power to detect observed effect:")

    if diff > 0.05:  # At least 5pp difference
        # Simplified calculation
        n_per_group = int(16 / (diff**2))  # Rough approximation
        n_per_group = max(n_per_group, 30)  # Minimum
        print(f"    ~{n_per_group} per condition ({n_per_group * 2} total)")
    else:
        print(
            "    Effect too small - would need very large sample (>200 per condition)"
        )

    print("\n  Current sample: {len(unaware)} unaware, {len(aware)} aware")


def summary_recommendation(results: dict[str, Any]) -> None:
    """Provide summary and recommendation."""
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 80)

    trials = results["trials"]

    unaware = [
        1 if t["success"] else 0
        for t in trials
        if t["awareness_condition"] == "NO_AWARENESS"
    ]
    aware = [
        1 if t["success"] else 0
        for t in trials
        if t["awareness_condition"] == "OVERALL_AND_INDIVIDUAL"
    ]

    diff_ci = bootstrap_difference_ci(aware, unaware)

    print("\nKey Findings:")
    print(
        f"  1. Awareness effect: {diff_ci[0] * 100:+.1f}pp [{diff_ci[1] * 100:+.1f}, {diff_ci[2] * 100:+.1f}]"
    )

    if diff_ci[1] > 0:
        print("  2. Direction: Awareness HELPS (statistically significant)")
        conclusion = "POSITIVE_EFFECT"
    elif diff_ci[2] < 0:
        print("  2. Direction: Awareness HURTS (statistically significant)")
        conclusion = "NEGATIVE_EFFECT"
    else:
        if diff_ci[0] > 0:
            print("  2. Direction: Trending positive but NOT significant")
        else:
            print("  2. Direction: Trending negative but NOT significant")
        conclusion = "NULL_EFFECT"

    # Recommendation
    print("\nRecommendation:")
    if conclusion == "NULL_EFFECT" and abs(diff_ci[0]) < 0.10:
        print("  → Effect likely too small to be meaningful")
        print("  → Consider: Different manipulation or accept null result")
    elif conclusion == "NULL_EFFECT":
        print("  → Effect direction visible but CI crosses zero")
        print("  → Consider: Larger sample for full study (~50 per cell)")
    else:
        print(f"  → Effect is significant and {conclusion.lower().replace('_', ' ')}")
        print("  → Current sample size may be sufficient")
        print("  → Document finding and consider replication")


def main() -> None:
    """Run full analysis."""
    results = load_latest_results()

    print(f"Analyzing {results['total_trials']} trials\n")

    analyze_main_effects(results)
    analyze_paired_effects(results)  # Within-subjects paired analysis
    analyze_difficulty_moderation(results)
    analyze_token_usage(results)
    analyze_iterations(results)
    power_analysis(results)
    summary_recommendation(results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
