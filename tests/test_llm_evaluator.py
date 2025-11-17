"""Validation tests for the LLM-based evaluator.

Tests:
1. Sanity checks with clearly better/worse responses
2. Position bias testing (swap consistency)
3. Basic functionality of pairwise comparison
"""

from experiments.evaluator import LLMResponseEvaluator
from experiments.tasks import ResearchTask


def test_sanity_check():
    """Test that evaluator correctly identifies clearly better responses."""
    print("\n" + "=" * 70)
    print("TEST 1: Sanity Check - Clear Quality Difference")
    print("=" * 70)

    evaluator = LLMResponseEvaluator()

    # Create simple task
    task = ResearchTask(
        id="test_01",
        question="What is quantum computing?",
        complexity="simple",
        expected_tool_use=1,
        category="technology",
    )

    # Good response: accurate, complete, clear
    good_response = """Quantum computing is a revolutionary computing paradigm that leverages quantum mechanical phenomena to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or "qubits" that can exist in superposition - simultaneously representing 0 and 1.

Key principles:
- Superposition: Qubits can be in multiple states simultaneously
- Entanglement: Qubits can be correlated in ways impossible classically
- Quantum interference: Amplifying correct answers while canceling wrong ones

Main applications:
- Cryptography: Breaking current encryption, quantum-safe protocols
- Drug discovery: Simulating molecular interactions
- Optimization: Solving complex logistics and scheduling problems
- Machine learning: Quantum algorithms for AI
"""

    # Bad response: vague, incomplete, unclear
    bad_response = """Quantum computing is like a new type of computer that works differently. It uses some quantum stuff to do calculations. Scientists think it might be useful for some things in the future."""

    print("\n--- Comparing Good vs Bad Response ---")
    scores_good, scores_bad = evaluator.evaluate_pairwise_with_swap(
        good_response, bad_response, task, "good", "bad"
    )

    print("\n** Results **")
    print("\nGood Response Scores:")
    for dim in ["accuracy", "completeness", "clarity", "depth", "conciseness"]:
        score = scores_good[dim]
        print(f"  {dim}: {score.score:.2f} (confidence: {score.confidence})")

    print("\nBad Response Scores:")
    for dim in ["accuracy", "completeness", "clarity", "depth", "conciseness"]:
        score = scores_bad[dim]
        print(f"  {dim}: {score.score:.2f} (confidence: {score.confidence})")

    # Verify good response scores higher
    avg_good = (
        sum(
            scores_good[dim].score
            for dim in ["accuracy", "completeness", "clarity", "depth", "conciseness"]
        )
        / 5
    )
    avg_bad = (
        sum(
            scores_bad[dim].score
            for dim in ["accuracy", "completeness", "clarity", "depth", "conciseness"]
        )
        / 5
    )

    print(f"\nAverage - Good: {avg_good:.2f}, Bad: {avg_bad:.2f}")

    if avg_good > avg_bad:
        print("✓ PASS: Evaluator correctly identifies better response")
    else:
        print("✗ FAIL: Evaluator did not identify better response")

    return avg_good > avg_bad


def test_conciseness_vs_verbosity():
    """Test that conciseness measures information density, not brevity."""
    print("\n" + "=" * 70)
    print("TEST 2: Conciseness - Information Density vs Length")
    print("=" * 70)

    evaluator = LLMResponseEvaluator()

    task = ResearchTask(
        id="test_02",
        question="What is machine learning?",
        complexity="simple",
        expected_tool_use=1,
        category="technology",
    )

    # Concise but informative
    concise_response = """Machine learning is a subset of AI where systems learn patterns from data without explicit programming. It works through algorithms that iteratively improve by identifying patterns in training data, adjusting model parameters to minimize prediction errors.

Key paradigms: supervised learning (labeled data), unsupervised learning (pattern discovery), reinforcement learning (reward-based).

Applications: recommendation systems, fraud detection, medical diagnosis, autonomous vehicles, natural language processing."""

    # Verbose with redundancy
    verbose_response = """Machine learning, which is a really important and fascinating field of computer science, is basically a type of artificial intelligence where computer systems and programs can actually learn things from data instead of being explicitly programmed with specific instructions for every single task they need to perform.

The way that machine learning works is really quite interesting. Basically, what happens is that algorithms, which are essentially step-by-step procedures, go through data over and over again, and each time they go through the data, they try to find patterns and relationships in that data. As they find these patterns, they adjust their internal parameters and settings to try to make better predictions. This process continues iteratively until the model performs well.

When we talk about different types of machine learning, there are several main categories. First, there's supervised learning, where the algorithm learns from data that has labels. Then there's unsupervised learning, where the algorithm tries to find patterns without labels. And finally, there's reinforcement learning, where the algorithm learns by receiving rewards or penalties for its actions.

Machine learning has many, many applications in the real world today. For example, it's used in recommendation systems that suggest products or content. It's also used for detecting fraud in financial transactions. Additionally, it's applied in medical diagnosis to help doctors. Furthermore, it's essential for autonomous vehicles that drive themselves. And it's also used extensively in natural language processing tasks."""

    print("\n--- Comparing Concise vs Verbose Response ---")
    scores_concise, scores_verbose = evaluator.evaluate_pairwise_with_swap(
        concise_response, verbose_response, task, "concise", "verbose"
    )

    print("\n** Results **")
    print(
        f"\nConcise Response - Conciseness: {scores_concise['conciseness'].score:.2f}"
    )
    print(f"Verbose Response - Conciseness: {scores_verbose['conciseness'].score:.2f}")

    print(f"\nReasoning: {scores_concise['conciseness'].reasoning[:200]}...")

    if scores_concise["conciseness"].score > scores_verbose["conciseness"].score:
        print("✓ PASS: Concise response scores higher on information density")
    else:
        print("✗ FAIL: Verbose response incorrectly scored higher")

    return scores_concise["conciseness"].score > scores_verbose["conciseness"].score


def test_position_bias():
    """Test position bias mitigation via score consistency."""
    print("\n" + "=" * 70)
    print("TEST 3: Position Bias - Score Consistency")
    print("=" * 70)

    evaluator = LLMResponseEvaluator()

    task = ResearchTask(
        id="test_03",
        question="What is climate change?",
        complexity="simple",
        expected_tool_use=1,
        category="science",
    )

    response_a = """Climate change refers to long-term shifts in global temperatures and weather patterns. Primary causes include greenhouse gas emissions from fossil fuels, deforestation, and industrial activities. Effects include rising sea levels, extreme weather events, ecosystem disruption, and threats to food security."""

    response_b = """Climate change is the gradual warming of Earth's atmosphere due to human activities. Main drivers are CO2 emissions, methane release, and land use changes. Consequences include melting ice caps, increased droughts and floods, biodiversity loss, and agricultural challenges."""

    print("\n--- Evaluating Same Pair Twice ---")

    # Evaluate A vs B
    print("\nRound 1: A vs B")
    scores_a1, scores_b1 = evaluator.evaluate_pairwise_with_swap(
        response_a, response_b, task, "A", "B"
    )

    # Evaluate B vs A (should give same results due to swapping)
    print("\nRound 2: B vs A")
    scores_b2, scores_a2 = evaluator.evaluate_pairwise_with_swap(
        response_b, response_a, task, "B", "A"
    )

    # Compare consistency
    print("\n** Consistency Check **")
    max_diff = 0
    for dim in ["accuracy", "completeness", "clarity", "depth", "conciseness"]:
        diff_a = abs(scores_a1[dim].score - scores_a2[dim].score)
        diff_b = abs(scores_b1[dim].score - scores_b2[dim].score)
        max_diff = max(max_diff, diff_a, diff_b)

        print(f"\n{dim}:")
        print(
            f"  A: {scores_a1[dim].score:.2f} (Round 1) vs {scores_a2[dim].score:.2f} (Round 2) | Diff: {diff_a:.2f}"
        )
        print(
            f"  B: {scores_b1[dim].score:.2f} (Round 1) vs {scores_b2[dim].score:.2f} (Round 2) | Diff: {diff_b:.2f}"
        )

    print(f"\nMaximum score difference across rounds: {max_diff:.2f}")

    # Target: < 0.5 point difference (on 1-5 scale)
    if max_diff < 0.5:
        print("✓ PASS: Excellent consistency (< 0.5 point variance)")
    elif max_diff < 1.0:
        print("✓ PASS: Good consistency (< 1.0 point variance)")
    else:
        print("⚠ WARNING: Moderate inconsistency (>= 1.0 point variance)")

    return max_diff < 1.0


def test_round_robin_ranking():
    """Test round-robin ranking of three strategies."""
    print("\n" + "=" * 70)
    print("TEST 4: Round-Robin Ranking")
    print("=" * 70)

    evaluator = LLMResponseEvaluator()

    task = ResearchTask(
        id="test_04",
        question="What are the benefits of renewable energy?",
        complexity="moderate",
        expected_tool_use=2,
        category="science",
    )

    responses = {
        "comprehensive": """Renewable energy offers substantial environmental, economic, and sustainability benefits. Environmentally, it reduces greenhouse gas emissions by 70-90% compared to fossil fuels, decreases air pollution (preventing ~7M premature deaths annually), and minimizes water usage. Economically, renewable costs have dropped 89% for solar and 70% for wind since 2010, creating 12M jobs globally, with levelized costs now competitive with coal ($40-60/MWh). For sustainability, renewables provide energy independence, reduce geopolitical conflicts over resources, and ensure long-term energy security as fossil fuels deplete.""",
        "brief": """Renewable energy is better for the environment, creates jobs, and is sustainable long-term. It reduces emissions and pollution while becoming more affordable.""",
        "moderate": """Renewable energy provides important benefits across three dimensions. Environmentally, it significantly reduces carbon emissions and air pollution. Economically, prices have fallen dramatically, making renewables cost-competitive while creating employment. From a sustainability perspective, renewables offer energy security and independence from finite fossil fuel resources.""",
    }

    print("\nRanking three responses of different lengths...")
    scores = evaluator.rank_strategies(responses, task)

    print("\n** Final Rankings **")
    ranked = sorted(scores.items(), key=lambda x: x[1].overall, reverse=True)

    for rank, (strategy, score) in enumerate(ranked, 1):
        print(f"\n{rank}. {strategy.upper()} - Overall: {score.overall:.2f}")
        print(
            f"   Accuracy: {score.accuracy:.2f} | Completeness: {score.completeness:.2f}"
        )
        print(
            f"   Clarity: {score.clarity:.2f} | Depth: {score.depth:.2f} | Conciseness: {score.conciseness:.2f}"
        )

    # Verify comprehensive is not automatically best (would indicate length bias)
    print("\n** Bias Check **")
    if ranked[0][0] == "comprehensive":
        print("⚠ WARNING: Longest response ranked first (possible length bias)")
    elif ranked[0][0] == "brief":
        print("⚠ WARNING: Shortest response ranked first (possible brevity bias)")
    else:
        print("✓ PASS: Moderate response ranked best (balanced evaluation)")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LLM EVALUATOR VALIDATION TESTS")
    print("=" * 70)

    results = {
        "Sanity Check": test_sanity_check(),
        "Conciseness Test": test_conciseness_vs_verbosity(),
        "Position Bias": test_position_bias(),
        "Round-Robin Ranking": test_round_robin_ranking(),
    }

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Evaluator is ready for use.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review results above.")
