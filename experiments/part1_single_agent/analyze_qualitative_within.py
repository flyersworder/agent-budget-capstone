"""Qualitative analysis of within-subjects results.

Extracts concrete examples showing how aware vs unaware agents
differ when answering the SAME questions.
"""

import json
from pathlib import Path
from typing import Any


def load_results() -> dict[str, Any]:
    """Load most recent within-subjects results."""
    results_dir = Path("experiments/results")
    json_files = list(results_dir.glob("part1_within_subjects_*.json"))
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

    with open(latest_file) as f:
        data: dict[str, Any] = json.load(f)
    return data


def group_by_question(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group results by question_id to get pairs."""
    by_question: dict[str, dict[str, Any]] = {}

    for r in results:
        if not r["success"]:
            continue

        qid = r["question_id"]
        condition = r["condition"]

        if qid not in by_question:
            by_question[qid] = {}
        by_question[qid][condition] = r

    return by_question


def find_biggest_wins(
    by_question: dict[str, dict[str, Any]], for_condition: str
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Find questions where specified condition performed much better."""
    wins = []

    for qid, conditions in by_question.items():
        if "unaware" not in conditions or "aware" not in conditions:
            continue

        unaware = conditions["unaware"]
        aware = conditions["aware"]

        if for_condition == "aware":
            diff = aware["correctness"] - unaware["correctness"]
        else:
            diff = unaware["correctness"] - aware["correctness"]

        if diff > 0.4:  # Big win
            wins.append((qid, unaware, aware))

    # Sort by difference
    wins.sort(
        key=lambda x: abs(x[1]["correctness"] - x[2]["correctness"]), reverse=True
    )
    return wins


def analyze_thinking_differences(by_question: dict[str, dict[str, Any]]) -> None:
    """Analyze thinking text patterns."""
    print("=" * 80)
    print("THINKING TEXT PATTERNS")
    print("=" * 80)
    print()

    thinking_diffs = []
    reasoning_token_diffs = []

    for qid, conditions in by_question.items():
        if "unaware" not in conditions or "aware" not in conditions:
            continue

        unaware = conditions["unaware"]
        aware = conditions["aware"]

        thinking_diff = len(aware["thinking_text"]) - len(unaware["thinking_text"])
        reasoning_diff = (
            aware["reasoning_tokens_used"] - unaware["reasoning_tokens_used"]
        )

        thinking_diffs.append(thinking_diff)
        reasoning_token_diffs.append(reasoning_diff)

    import numpy as np

    print("Thinking text length difference (aware - unaware):")
    print(f"  Mean: {np.mean(thinking_diffs):+.0f} chars")
    print(f"  Median: {np.median(thinking_diffs):+.0f} chars")
    print(f"  Std: {np.std(thinking_diffs):.0f} chars")
    print()

    print("Reasoning tokens difference (aware - unaware):")
    print(f"  Mean: {np.mean(reasoning_token_diffs):+.0f} tokens")
    print(f"  Median: {np.median(reasoning_token_diffs):+.0f} tokens")
    print(f"  Std: {np.std(reasoning_token_diffs):.0f} tokens")
    print()

    # Count meta-cognitive markers
    meta_markers = [
        "budget",
        "token",
        "strategic",
        "prioritize",
        "allocate",
        "conserve",
        "efficient",
    ]

    unaware_with_meta = 0
    aware_with_meta = 0

    for qid, conditions in by_question.items():
        if "unaware" not in conditions or "aware" not in conditions:
            continue

        unaware = conditions["unaware"]
        aware = conditions["aware"]

        unaware_has = any(
            marker in unaware["thinking_text"].lower() for marker in meta_markers
        )
        aware_has = any(
            marker in aware["thinking_text"].lower() for marker in meta_markers
        )

        if unaware_has:
            unaware_with_meta += 1
        if aware_has:
            aware_with_meta += 1

    print("Meta-cognitive markers in thinking:")
    print(
        f"  Unaware: {unaware_with_meta}/{len(by_question)} ({unaware_with_meta / len(by_question) * 100:.1f}%)"
    )
    print(
        f"  Aware:   {aware_with_meta}/{len(by_question)} ({aware_with_meta / len(by_question) * 100:.1f}%)"
    )
    print()


def show_aware_wins(by_question: dict[str, dict[str, Any]]) -> None:
    """Show examples where aware agent won."""
    print("=" * 80)
    print("AWARE AGENT WINS (Same Question, Better Performance)")
    print("=" * 80)
    print()

    wins = find_biggest_wins(by_question, "aware")

    if not wins:
        print("No major aware wins found (diff > 0.4)")
        return

    print(f"Found {len(wins)} questions where aware significantly outperformed unaware")
    print()

    # Show top 3 examples
    for i, (qid, unaware, aware) in enumerate(wins[:3], 1):
        print(f"{'=' * 80}")
        print(f"EXAMPLE {i}: {qid}")
        print(f"{'=' * 80}")
        print()

        print(f"Question: {unaware['question']}")
        print(f"Category: {unaware['category']}")
        print(f"Budget: {unaware['budget_level']}")
        print()

        print("UNAWARE AGENT:")
        print(f"  Correctness: {unaware['correctness']:.1%}")
        print(
            f"  Tokens: {unaware['total_tokens_used']} (reasoning: {unaware['reasoning_tokens_used']})"
        )
        print(f"  Response: {unaware['response'][:200]}...")
        print()

        print("AWARE AGENT:")
        print(f"  Correctness: {aware['correctness']:.1%}")
        print(
            f"  Tokens: {aware['total_tokens_used']} (reasoning: {aware['reasoning_tokens_used']})"
        )
        print(f"  Response: {aware['response'][:200]}...")
        print()

        print("DIFFERENCE:")
        acc_diff = aware["correctness"] - unaware["correctness"]
        token_diff = aware["total_tokens_used"] - unaware["total_tokens_used"]
        print(f"  Accuracy: {acc_diff:+.1%} (aware advantage)")
        print(f"  Tokens: {token_diff:+d}")
        print()


def show_unaware_wins(by_question: dict[str, dict[str, Any]]) -> None:
    """Show examples where unaware agent won."""
    print("=" * 80)
    print("UNAWARE AGENT WINS (Rare Cases)")
    print("=" * 80)
    print()

    wins = find_biggest_wins(by_question, "unaware")

    if not wins:
        print("No major unaware wins found (diff > 0.4)")
        return

    print(f"Found {len(wins)} questions where unaware outperformed aware")
    print()

    # Show all examples (should be rare)
    for i, (qid, unaware, aware) in enumerate(wins, 1):
        print(f"{'=' * 80}")
        print(f"EXAMPLE {i}: {qid}")
        print(f"{'=' * 80}")
        print()

        print(f"Question: {unaware['question']}")
        print(f"Category: {unaware['category']}")
        print()

        print(
            f"UNAWARE: {unaware['correctness']:.1%} correct ({unaware['total_tokens_used']} tokens)"
        )
        print(
            f"AWARE:   {aware['correctness']:.1%} correct ({aware['total_tokens_used']} tokens)"
        )
        print()

        acc_diff = unaware["correctness"] - aware["correctness"]
        print(f"  → Unaware advantage: {acc_diff:+.1%}")
        print()


def analyze_token_efficiency(by_question: dict[str, dict[str, Any]]) -> None:
    """Analyze token efficiency vs accuracy trade-off."""
    print("=" * 80)
    print("TOKEN EFFICIENCY ANALYSIS")
    print("=" * 80)
    print()

    # Categorize pairs by outcome
    aware_worth_it = []  # Aware uses more tokens AND is more accurate
    aware_wasteful = []  # Aware uses more tokens but LESS accurate
    aware_efficient = []  # Aware uses fewer tokens AND more accurate
    tie_cases = []  # Same accuracy

    for qid, conditions in by_question.items():
        if "unaware" not in conditions or "aware" not in conditions:
            continue

        unaware = conditions["unaware"]
        aware = conditions["aware"]

        acc_diff = aware["correctness"] - unaware["correctness"]
        token_diff = aware["total_tokens_used"] - unaware["total_tokens_used"]

        if abs(acc_diff) < 0.01:  # Tie
            tie_cases.append((qid, token_diff))
        elif acc_diff > 0.01 and token_diff > 0:  # Aware better but costs more
            aware_worth_it.append((qid, acc_diff, token_diff))
        elif acc_diff < -0.01 and token_diff > 0:  # Aware worse and costs more
            aware_wasteful.append((qid, acc_diff, token_diff))
        elif acc_diff > 0.01 and token_diff < 0:  # Aware better AND cheaper!
            aware_efficient.append((qid, acc_diff, token_diff))

    print(f"Aware worth it (better + costs more): {len(aware_worth_it)}")
    if aware_worth_it:
        import numpy as np

        avg_acc = np.mean([d[1] for d in aware_worth_it])
        avg_tokens = np.mean([d[2] for d in aware_worth_it])
        print(f"  Avg gain: {avg_acc:+.1%} for {avg_tokens:+.0f} tokens")

    print()
    print(f"Aware efficient (better + cheaper): {len(aware_efficient)}")
    if aware_efficient:
        import numpy as np

        avg_acc = np.mean([d[1] for d in aware_efficient])
        avg_tokens = np.mean([d[2] for d in aware_efficient])
        print(f"  Avg gain: {avg_acc:+.1%} with {avg_tokens:.0f} token savings!")

    print()
    print(f"Aware wasteful (worse + costs more): {len(aware_wasteful)}")
    if aware_wasteful:
        import numpy as np

        avg_acc = np.mean([d[1] for d in aware_wasteful])
        avg_tokens = np.mean([d[2] for d in aware_wasteful])
        print(f"  Avg loss: {avg_acc:-.1%} for {avg_tokens:+.0f} wasted tokens")

    print()
    print(f"Ties (same accuracy): {len(tie_cases)}")
    if tie_cases:
        import numpy as np

        avg_tokens = np.mean([d[1] for d in tie_cases])
        print(f"  Avg token difference: {avg_tokens:+.0f}")

    print()
    print(
        f"**Interpretation**: Aware agents are 'worth it' in {len(aware_worth_it)}/50 cases"
    )
    print(
        f"  ({len(aware_worth_it) / len(by_question) * 100:.1f}% of questions benefit from awareness)"
    )
    print()


def main() -> None:
    """Run qualitative analysis."""
    data = load_results()
    results = data["results"]

    print(f"Analyzing: {len(results)} observations")
    print(f"Design: {data['metadata']['design']}")
    print()

    by_question = group_by_question(results)
    print(f"Complete pairs: {len(by_question)}")
    print()

    # Run analyses
    analyze_thinking_differences(by_question)
    analyze_token_efficiency(by_question)
    show_aware_wins(by_question)
    show_unaware_wins(by_question)

    print("=" * 80)
    print("QUALITATIVE ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
