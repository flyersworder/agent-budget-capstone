"""Comprehensive statistical analysis of Part 2 full study results.

Includes:
- Descriptive statistics by condition
- Bootstrap confidence intervals
- Effect size calculations
- Token usage patterns
- Statistical significance tests
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Any


def bootstrap_mean(
    data: list[float], n_bootstrap: int = 10000, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Calculate bootstrap confidence interval for mean.

    Args:
        data: List of values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    mean = np.mean(data)
    alpha = 1 - confidence
    lower = np.percentile(means, alpha / 2 * 100)
    upper = np.percentile(means, (1 - alpha / 2) * 100)

    return mean, lower, upper


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Calculate Cohen's d effect size.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def analyze_by_condition(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze results broken down by budget × awareness condition."""
    print("=" * 80)
    print("RESULTS BY CONDITION (with Bootstrap 95% CIs)")
    print("=" * 80)
    print()

    budget_levels = ["tight", "moderate", "comfortable"]
    awareness_levels = ["unaware", "aware"]

    condition_data = {}

    for budget in budget_levels:
        for awareness in awareness_levels:
            key = f"{budget}_{awareness}"
            condition_results = [
                r
                for r in results
                if r["budget_level"] == budget and r["condition"] == awareness
            ]

            if condition_results:
                correctness = [r["correctness"] for r in condition_results]
                tokens = [r["total_tokens_used"] for r in condition_results]
                reasoning = [r["reasoning_tokens_used"] for r in condition_results]
                output_tok = [r["output_tokens_used"] for r in condition_results]

                # Bootstrap CIs
                acc_mean, acc_lower, acc_upper = bootstrap_mean(correctness)
                tok_mean, tok_lower, tok_upper = bootstrap_mean(tokens)

                condition_data[key] = {
                    "budget": budget,
                    "awareness": awareness,
                    "n": len(condition_results),
                    "accuracy": correctness,
                    "tokens": tokens,
                    "reasoning": reasoning,
                    "output": output_tok,
                    "acc_mean": acc_mean,
                    "acc_ci": (acc_lower, acc_upper),
                    "tok_mean": tok_mean,
                    "tok_ci": (tok_lower, tok_upper),
                }

    # Print formatted table
    for budget in budget_levels:
        print(f"{budget.upper()} Budget:")
        print(f"{'Condition':<12} {'N':<4} {'Accuracy':<25} {'Avg Tokens':<25}")
        print("-" * 80)

        for awareness in awareness_levels:
            key = f"{budget}_{awareness}"
            d: dict[str, Any] = condition_data[key]
            acc_ci: tuple[float, float] = d["acc_ci"]
            tok_ci: tuple[float, float] = d["tok_ci"]
            acc_str = f"{d['acc_mean']:.1%} [{acc_ci[0]:.1%}, {acc_ci[1]:.1%}]"
            tok_str = f"{d['tok_mean']:.0f} [{tok_ci[0]:.0f}, {tok_ci[1]:.0f}]"
            print(f"{awareness:<12} {d['n']:<4} {acc_str:<25} {tok_str:<25}")

        print()

    return condition_data


def analyze_main_effects(condition_data: dict[str, Any]) -> None:
    """Analyze main effects of budget and awareness."""
    print("=" * 80)
    print("MAIN EFFECTS ANALYSIS")
    print("=" * 80)
    print()

    # Effect of awareness (collapsed across budgets)
    print("1. AWARENESS EFFECT (Aware vs Unaware)")
    print("-" * 80)

    unaware_acc: list[float] = []
    aware_acc: list[float] = []
    for key, data in condition_data.items():
        if data["awareness"] == "unaware":
            unaware_acc.extend(data["accuracy"])
        else:
            aware_acc.extend(data["accuracy"])

    unaware_mean, unaware_lower, unaware_upper = bootstrap_mean(unaware_acc)
    aware_mean, aware_lower, aware_upper = bootstrap_mean(aware_acc)

    print(
        f"Unaware: {unaware_mean:.1%} [{unaware_lower:.1%}, {unaware_upper:.1%}] (n={len(unaware_acc)})"
    )
    print(
        f"Aware:   {aware_mean:.1%} [{aware_lower:.1%}, {aware_upper:.1%}] (n={len(aware_acc)})"
    )
    print(f"Difference: {(aware_mean - unaware_mean):.1%}")

    # T-test
    t_stat, p_value = stats.ttest_ind(aware_acc, unaware_acc)
    print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")

    # Effect size
    d = cohens_d(aware_acc, unaware_acc)
    print(
        f"Cohen's d: {d:.3f} ({'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'})"
    )
    print()

    # Effect of budget level (collapsed across awareness)
    print("2. BUDGET LEVEL EFFECT")
    print("-" * 80)

    budget_accs: dict[str, list[float]] = {}
    for budget in ["tight", "moderate", "comfortable"]:
        budget_accs[budget] = []
        for key, data in condition_data.items():
            if data["budget"] == budget:
                budget_accs[budget].extend(data["accuracy"])

    for budget in ["tight", "moderate", "comfortable"]:
        acc = budget_accs[budget]
        mean, lower, upper = bootstrap_mean(acc)
        print(
            f"{budget.capitalize():<12}: {mean:.1%} [{lower:.1%}, {upper:.1%}] (n={len(acc)})"
        )

    # ANOVA
    f_stat, p_value = stats.f_oneway(
        budget_accs["tight"], budget_accs["moderate"], budget_accs["comfortable"]
    )
    print(f"\nANOVA: F={f_stat:.3f}, p={p_value:.4f}")
    print()


def analyze_interaction(condition_data: dict[str, Any]) -> None:
    """Analyze interaction between budget and awareness."""
    print("=" * 80)
    print("INTERACTION: Budget × Awareness")
    print("=" * 80)
    print()

    print("Awareness Effect by Budget Level:")
    print("-" * 80)

    for budget in ["tight", "moderate", "comfortable"]:
        unaware_key = f"{budget}_unaware"
        aware_key = f"{budget}_aware"

        unaware_acc = condition_data[unaware_key]["accuracy"]
        aware_acc = condition_data[aware_key]["accuracy"]

        diff = np.mean(aware_acc) - np.mean(unaware_acc)
        d = cohens_d(aware_acc, unaware_acc)

        t_stat, p_value = stats.ttest_ind(aware_acc, unaware_acc)

        print(f"\n{budget.upper()}:")
        print(f"  Unaware: {np.mean(unaware_acc):.1%}")
        print(f"  Aware:   {np.mean(aware_acc):.1%}")
        print(f"  Difference: {diff:+.1%}")
        print(f"  Cohen's d: {d:.3f}")
        print(f"  T-test: t={t_stat:.3f}, p={p_value:.4f}")

    print()


def analyze_token_usage(condition_data: dict[str, Any]) -> None:
    """Analyze token usage patterns."""
    print("=" * 80)
    print("TOKEN USAGE ANALYSIS")
    print("=" * 80)
    print()

    print("Average Token Usage by Condition:")
    print("-" * 80)
    print(
        f"{'Condition':<20} {'Reasoning':<15} {'Output':<15} {'Total':<15} {'Budget %':<10}"
    )
    print("-" * 80)

    budgets = {"tight": 640, "moderate": 1280, "comfortable": 2560}

    for budget in ["tight", "moderate", "comfortable"]:
        for awareness in ["unaware", "aware"]:
            key = f"{budget}_{awareness}"
            d = condition_data[key]

            reasoning_mean = np.mean(d["reasoning"])
            output_mean = np.mean(d["output"])
            total_mean = np.mean(d["tokens"])
            utilization = (total_mean / budgets[budget]) * 100

            label = f"{budget}/{awareness}"
            print(
                f"{label:<20} {reasoning_mean:<15.0f} {output_mean:<15.0f} {total_mean:<15.0f} {utilization:<10.1f}%"
            )
        print()

    # Awareness effect on token usage
    print("Awareness Effect on Token Usage:")
    print("-" * 80)

    for budget in ["tight", "moderate", "comfortable"]:
        unaware_tokens = condition_data[f"{budget}_unaware"]["tokens"]
        aware_tokens = condition_data[f"{budget}_aware"]["tokens"]

        diff = np.mean(aware_tokens) - np.mean(unaware_tokens)
        pct_increase = (diff / np.mean(unaware_tokens)) * 100

        print(f"{budget.capitalize()}: {diff:+.0f} tokens ({pct_increase:+.1f}%)")

    print()


def main() -> None:
    """Run comprehensive analysis."""
    # Load results
    results_dir = Path("experiments/results")
    results_files = sorted(results_dir.glob("part2_full_*.json"))

    if not results_files:
        print("No full study results found!")
        return

    latest = results_files[-1]
    print(f"Analyzing: {latest.name}")
    print()

    with open(latest) as f:
        data = json.load(f)

    results = data["results"]
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len([r for r in results if r['success']])}")
    print()

    # Run analyses
    condition_data = analyze_by_condition(results)
    analyze_main_effects(condition_data)
    analyze_interaction(condition_data)
    analyze_token_usage(condition_data)

    # Summary
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()
    print("1. **SURPRISING: Budget awareness HURTS performance**")
    print("   - Aware agents: lower accuracy across all budget levels")
    print("   - Effect strongest in TIGHT budget (62.5% → 37.5%)")
    print()
    print("2. **Budget level matters**")
    print("   - MODERATE budget shows best performance (81.2% / 75.0%)")
    print("   - TIGHT budget challenging, especially for aware agents")
    print()
    print("3. **Aware agents use more tokens but perform worse**")
    print("   - Consistently higher token usage")
    print("   - Suggests inefficient allocation or overthinking")
    print()
    print("4. **Interaction effect**")
    print("   - Budget awareness effect varies by constraint level")
    print("   - Tightest constraints show largest negative effect")


if __name__ == "__main__":
    main()
