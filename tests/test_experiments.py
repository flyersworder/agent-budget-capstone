"""Test script for experiment infrastructure verification."""

from experiments import ResponseEvaluator, QualityScore
from experiments.tasks import (
    ResearchTask,
    get_all_tasks,
    get_tasks_by_complexity,
    get_task_by_id,
)


def test_task_loading() -> None:
    """Test research task loading and filtering."""
    print("Testing Task Loading...")

    # Test getting all tasks
    all_tasks = get_all_tasks()
    assert len(all_tasks) == 9, f"Expected 9 tasks, got {len(all_tasks)}"
    print(f"  ✓ Loaded {len(all_tasks)} total tasks")

    # Test filtering by complexity
    simple_tasks = get_tasks_by_complexity("simple")
    assert len(simple_tasks) == 3, f"Expected 3 simple tasks, got {len(simple_tasks)}"
    print(f"  ✓ Found {len(simple_tasks)} simple tasks")

    moderate_tasks = get_tasks_by_complexity("moderate")
    assert len(moderate_tasks) == 3
    print(f"  ✓ Found {len(moderate_tasks)} moderate tasks")

    complex_tasks = get_tasks_by_complexity("complex")
    assert len(complex_tasks) == 3
    print(f"  ✓ Found {len(complex_tasks)} complex tasks")

    # Test getting specific task by ID
    task = get_task_by_id("simple_01")
    assert task is not None
    assert task.id == "simple_01"
    assert task.complexity == "simple"
    print(f"  ✓ Retrieved task by ID: {task.id}")

    # Test task properties
    print("\n  Sample task details:")
    print(f"    ID: {task.id}")
    print(f"    Category: {task.category}")
    print(f"    Complexity: {task.complexity}")
    print(f"    Expected tool use: {task.expected_tool_use}")
    print(f"    Question: {task.question[:50]}...")


def test_evaluator() -> None:
    """Test response evaluator with sample responses."""
    print("\nTesting Response Evaluator...")

    evaluator = ResponseEvaluator()
    print("  ✓ Evaluator created")

    # Create a test task
    task = ResearchTask(
        id="test_01",
        question="What is quantum computing and what are its main applications?",
        complexity="simple",
        expected_tool_use=1,
        category="technology",
    )

    # Test with a good response
    good_response = """
Quantum computing is a revolutionary computing paradigm that leverages quantum mechanical
phenomena to process information. Unlike classical computers that use bits (0 or 1),
quantum computers use quantum bits or qubits that can exist in superposition states.

Key applications include:

1. Cryptography: Breaking current encryption and developing quantum-safe alternatives
2. Drug Discovery: Simulating molecular interactions for pharmaceutical research
3. Optimization: Solving complex logistical and financial optimization problems
4. Machine Learning: Accelerating AI training and pattern recognition

Research shows that quantum computers could solve certain problems exponentially faster
than classical computers. However, current quantum systems face challenges with error
correction and maintaining quantum coherence.
"""

    score = evaluator.evaluate_response(good_response, task)
    print("\n  Good response scores:")
    print(f"    Completeness: {score.completeness:.1f}/10")
    print(f"    Clarity: {score.clarity:.1f}/10")
    print(f"    Depth: {score.depth:.1f}/10")
    print(f"    Evidence: {score.evidence:.1f}/10")
    print(f"    Overall: {score.overall:.1f}/10")

    assert score.overall > 5.0, "Good response should score above 5.0"
    print("  ✓ Good response evaluated correctly")

    # Test with a poor response
    poor_response = "Quantum computing is fast computing."

    poor_score = evaluator.evaluate_response(poor_response, task)
    print("\n  Poor response scores:")
    print(f"    Overall: {poor_score.overall:.1f}/10")

    assert poor_score.overall < score.overall, "Poor response should score lower"
    print("  ✓ Poor response scored lower than good response")

    # Test comparison
    responses = {
        "deep": good_response,
        "verbose": poor_response,
        "balanced": good_response,
    }

    comparison = evaluator.compare_responses(responses, task)
    print("\n  Strategy comparison:")
    print(f"    Best for completeness: {comparison['best_performers']['completeness']}")
    print(f"    Best overall: {comparison['best_performers']['overall']}")
    print("  ✓ Response comparison works")


def test_quality_score() -> None:
    """Test QualityScore dataclass."""
    print("\nTesting QualityScore...")

    score = QualityScore(completeness=8.0, clarity=7.5, depth=9.0, evidence=6.0)

    # Test overall calculation
    expected_overall = (8.0 + 7.5 + 9.0 + 6.0) / 4
    assert abs(score.overall - expected_overall) < 0.01
    print(f"  ✓ Overall score calculated correctly: {score.overall:.2f}")

    # Test serialization
    score_dict = score.to_dict()
    assert "completeness" in score_dict
    assert "overall" in score_dict
    assert score_dict["overall"] == score.overall
    print("  ✓ Score serialization works")


def test_task_categories() -> None:
    """Test task organization by category."""
    print("\nTesting Task Categories...")

    all_tasks = get_all_tasks()
    categories = {task.category for task in all_tasks}

    print(f"  Available categories: {categories}")
    assert len(categories) > 0, "Should have at least one category"
    print(f"  ✓ Found {len(categories)} unique categories")

    # Count tasks per complexity
    complexity_counts = {}
    for task in all_tasks:
        complexity_counts[task.complexity] = (
            complexity_counts.get(task.complexity, 0) + 1
        )

    print("\n  Tasks per complexity level:")
    for complexity, count in complexity_counts.items():
        print(f"    {complexity}: {count} tasks")


def main() -> None:
    """Run all tests."""
    print("=" * 60)
    print("Experiment Infrastructure Test Suite")
    print("=" * 60)

    try:
        test_task_loading()
        test_evaluator()
        test_quality_score()
        test_task_categories()

        print("\n" + "=" * 60)
        print("✅ All experiment infrastructure tests passed!")
        print("=" * 60)

        print("\nNext steps:")
        print("  1. Ensure Google ADK dependencies are installed")
        print("  2. Set up GOOGLE_API_KEY in .env")
        print("  3. Run actual experiments with: python -m experiments.run_part1")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
