"""Analysis script for Part 1 within-subjects design results.

Performs paired statistical tests with bootstrap confidence intervals
for accuracy and token usage differences.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def load_results(json_path: str) -> dict[str, Any]:
    """Load within-subjects results from JSON."""
    with open(json_path) as f:
        data: dict[str, Any] = json.load(f)
        return data


def bootstrap_paired_ci(
    pairs: list[tuple[float, float]],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calculate bootstrap CI for paired differences.

    Args:
        pairs: List of (value1, value2) tuples
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (mean_difference, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)

    # Compute observed difference
    differences = [v1 - v2 for v1, v2 in pairs]
    observed_diff = np.mean(differences)

    # Bootstrap: resample PAIRS with replacement
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.choice(len(pairs), size=len(pairs), replace=True)
        sample_pairs = [pairs[i] for i in indices]
        sample_diffs = [v1 - v2 for v1, v2 in sample_pairs]
        bootstrap_diffs.append(np.mean(sample_diffs))

    # Percentile method
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return observed_diff, lower, upper


def bootstrap_cohens_d_paired_ci(
    pairs: list[tuple[float, float]],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calculate bootstrap CI for Cohen's d (paired).

    For paired data, Cohen's d = mean(differences) / std(differences)

    Args:
        pairs: List of (value1, value2) tuples
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        seed: Random seed

    Returns:
        Tuple of (cohens_d, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)

    def compute_cohens_d_paired(pair_list: list[tuple[float, float]]) -> float:
        """Compute Cohen's d for paired data."""
        diffs = [v1 - v2 for v1, v2 in pair_list]
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        if std_diff == 0:
            return 0.0
        return float(mean_diff / std_diff)

    # Observed Cohen's d
    observed_d = compute_cohens_d_paired(pairs)

    # Bootstrap
    bootstrap_ds = []
    for _ in range(n_bootstrap):
        indices = rng.choice(len(pairs), size=len(pairs), replace=True)
        sample_pairs = [pairs[i] for i in indices]
        bootstrap_ds.append(compute_cohens_d_paired(sample_pairs))

    # Percentile method
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_ds, 100 * alpha / 2)
    upper = np.percentile(bootstrap_ds, 100 * (1 - alpha / 2))

    return observed_d, lower, upper


def analyze_main_effect(results: list[dict[str, Any]]) -> None:
    """Analyze main effect of awareness (collapsed across budget levels)."""
    print("=" * 80)
    print("MAIN EFFECT: Budget Awareness")
    print("=" * 80)
    print()

    # Group by question_id
    by_question: dict[str, dict[str, dict[str, Any]]] = {}
    for r in results:
        if not r["success"]:
            continue

        qid = r["question_id"]
        condition = r["condition"]

        if qid not in by_question:
            by_question[qid] = {}
        by_question[qid][condition] = r

    # Extract pairs
    accuracy_pairs = []
    token_pairs = []

    for qid, conditions in by_question.items():
        if "unaware" in conditions and "aware" in conditions:
            unaware = conditions["unaware"]
            aware = conditions["aware"]

            accuracy_pairs.append((unaware["correctness"], aware["correctness"]))
            token_pairs.append(
                (unaware["total_tokens_used"], aware["total_tokens_used"])
            )

    print(f"Number of complete pairs: {len(accuracy_pairs)}")
    print()

    # Accuracy analysis
    print("ACCURACY:")
    print("-" * 40)

    unaware_acc = [u for u, a in accuracy_pairs]
    aware_acc = [a for u, a in accuracy_pairs]

    print(f"  Unaware: {np.mean(unaware_acc):.1%} ± {np.std(unaware_acc):.3f}")
    print(f"  Aware:   {np.mean(aware_acc):.1%} ± {np.std(aware_acc):.3f}")
    print()

    # Paired difference with bootstrap CI
    diff, diff_lo, diff_hi = bootstrap_paired_ci(accuracy_pairs)
    print("  Difference (Unaware - Aware):")
    print(f"    Point estimate: {diff:+.1%}")
    print(f"    95% Bootstrap CI: [{diff_lo:+.1%}, {diff_hi:+.1%}]")

    ci_excludes_zero = diff_lo > 0 or diff_hi < 0
    print(f"    CI excludes zero: {'Yes ***' if ci_excludes_zero else 'No'}")
    print()

    # Paired t-test
    differences = [u - a for u, a in accuracy_pairs]
    t_stat, p_value = stats.ttest_rel(unaware_acc, aware_acc)

    print("  Paired t-test:")
    print(f"    t-statistic: {t_stat:.3f}")
    print(
        f"    p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '†' if p_value < 0.1 else 'ns'}"
    )
    print()

    # Effect size with bootstrap CI
    d, d_lo, d_hi = bootstrap_cohens_d_paired_ci(accuracy_pairs)
    print("  Cohen's d (paired):")
    print(f"    Point estimate: {d:.3f}")
    print(f"    95% Bootstrap CI: [{d_lo:.3f}, {d_hi:.3f}]")
    print(
        f"    Interpretation: {'trivial' if abs(d) < 0.2 else 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'}"
    )
    print()

    # Win/Loss/Tie analysis
    wins = sum(1 for diff in differences if diff > 0.01)  # Unaware wins
    losses = sum(1 for diff in differences if diff < -0.01)  # Aware wins
    ties = len(differences) - wins - losses

    print("  Win/Loss/Tie (Unaware perspective):")
    print(f"    Wins:   {wins} ({wins / len(differences) * 100:.1f}%)")
    print(f"    Losses: {losses} ({losses / len(differences) * 100:.1f}%)")
    print(f"    Ties:   {ties} ({ties / len(differences) * 100:.1f}%)")
    print()

    # Sign test (non-parametric)
    sign_stat = sum(1 for diff in differences if diff > 0)
    sign_p = stats.binom_test(sign_stat, len(differences), 0.5, alternative="two-sided")
    print("  Sign test:")
    print(f"    Positive differences: {sign_stat}/{len(differences)}")
    print(
        f"    p-value: {sign_p:.4f} {'*' if sign_p < 0.05 else '†' if sign_p < 0.1 else 'ns'}"
    )
    print()

    # Token usage analysis
    print()
    print("TOKEN USAGE:")
    print("-" * 40)

    unaware_tokens = [u for u, a in token_pairs]
    aware_tokens = [a for u, a in token_pairs]

    print(f"  Unaware: {np.mean(unaware_tokens):.0f} ± {np.std(unaware_tokens):.0f}")
    print(f"  Aware:   {np.mean(aware_tokens):.0f} ± {np.std(aware_tokens):.0f}")
    print()

    token_diff, token_lo, token_hi = bootstrap_paired_ci(token_pairs)
    print("  Difference (Unaware - Aware):")
    print(f"    Point estimate: {token_diff:+.0f} tokens")
    print(f"    95% Bootstrap CI: [{token_lo:+.0f}, {token_hi:+.0f}]")
    print()


def analyze_budget_interaction(results: list[dict[str, Any]]) -> None:
    """Analyze Budget × Awareness interaction."""
    print("=" * 80)
    print("BUDGET × AWARENESS INTERACTION")
    print("=" * 80)
    print()

    budget_levels = ["tight", "moderate", "comfortable"]

    # Group by budget and question
    by_budget: dict[str, list[tuple[float, float]]] = {b: [] for b in budget_levels}

    by_question: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        if not r["success"]:
            continue

        key = (r["question_id"], r["budget_level"])
        if key not in by_question:
            by_question[key] = {}
        by_question[key][r["condition"]] = r

    # Extract pairs by budget
    for key, conditions in by_question.items():
        qid, budget = key
        if "unaware" in conditions and "aware" in conditions:
            unaware = conditions["unaware"]
            aware = conditions["aware"]
            by_budget[budget].append((unaware["correctness"], aware["correctness"]))

    print(
        f"{'Budget':<15} {'Unaware':<12} {'Aware':<12} {'Diff':<10} {'CI (95%)':<25} {'d':<8} {'p-value'}"
    )
    print("-" * 95)

    for budget in budget_levels:
        pairs = by_budget[budget]

        if len(pairs) < 3:
            print(f"{budget:<15} (insufficient data: n={len(pairs)})")
            continue

        unaware_vals = [u for u, a in pairs]
        aware_vals = [a for u, a in pairs]

        diff, diff_lo, diff_hi = bootstrap_paired_ci(pairs)
        d, _, _ = bootstrap_cohens_d_paired_ci(pairs)
        t_stat, p_value = stats.ttest_rel(unaware_vals, aware_vals)

        print(
            f"{budget:<15} {np.mean(unaware_vals):<12.1%} {np.mean(aware_vals):<12.1%} "
            f"{diff:+.1%}      [{diff_lo:+.1%}, {diff_hi:+.1%}]    "
            f"{d:<8.3f} {p_value:.4f}"
        )

    print()


def analyze_category_interaction(results: list[dict[str, Any]]) -> None:
    """Analyze Category × Awareness interaction."""
    print("=" * 80)
    print("CATEGORY × AWARENESS INTERACTION")
    print("=" * 80)
    print()

    # Group by category and question
    by_category: dict[str, list[tuple[float, float]]] = {}

    by_question: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        if not r["success"]:
            continue

        key = (r["question_id"], r["category"])
        if key not in by_question:
            by_question[key] = {}
        by_question[key][r["condition"]] = r

    # Extract pairs by category
    for key, conditions in by_question.items():
        qid, category = key
        if "unaware" in conditions and "aware" in conditions:
            if category not in by_category:
                by_category[category] = []

            unaware = conditions["unaware"]
            aware = conditions["aware"]
            by_category[category].append((unaware["correctness"], aware["correctness"]))

    # Filter to categories with sufficient data
    min_n = 3
    viable_categories = {
        cat: pairs for cat, pairs in by_category.items() if len(pairs) >= min_n
    }

    print(f"Categories with n≥{min_n} pairs: {len(viable_categories)}")
    print()

    if not viable_categories:
        print(f"⚠️  No categories have n≥{min_n} complete pairs")
        return

    print(
        f"{'Category':<25} {'Unaware':<10} {'Aware':<10} {'Diff':<8} "
        f"{'d':<8} {'p-value':<10} {'n'}"
    )
    print("-" * 80)

    for category in sorted(viable_categories.keys()):
        pairs = viable_categories[category]

        unaware_vals = [u for u, a in pairs]
        aware_vals = [a for u, a in pairs]

        diff, _, _ = bootstrap_paired_ci(pairs)
        d, _, _ = bootstrap_cohens_d_paired_ci(pairs)
        t_stat, p_value = stats.ttest_rel(unaware_vals, aware_vals)

        print(
            f"{category:<25} {np.mean(unaware_vals):>10.1%} {np.mean(aware_vals):>10.1%} "
            f"{diff:>+7.1%} {d:>8.2f} {p_value:>10.4f} ({len(pairs)})"
        )

    print()


def main() -> None:
    """Run analysis on within-subjects results."""
    # Find most recent results file
    results_dir = Path("experiments/results")
    json_files = list(results_dir.glob("part1_within_subjects_*.json"))

    if not json_files:
        print("❌ No within-subjects results found")
        return

    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing: {latest_file.name}")
    print()

    data = load_results(str(latest_file))
    results = data["results"]

    print(f"Total observations: {len(results)}")
    print(f"Successful: {len([r for r in results if r['success']])}")
    print(f"Design: {data['metadata'].get('design', 'unknown')}")
    print()

    # Run analyses
    analyze_main_effect(results)
    analyze_budget_interaction(results)
    analyze_category_interaction(results)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
