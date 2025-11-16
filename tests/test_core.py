"""Test script for core framework verification."""

from agent_budget import (
    AllocationStrategy,
    TokenBudget,
    AgentFactory,
    AgentMetrics,
    ToolUsageMetrics,
    UsageMonitor,
)


def test_token_budget() -> None:
    """Test TokenBudget calculations."""
    print("Testing TokenBudget...")

    # Test basic budget
    budget = TokenBudget(reasoning_tokens=8000, output_tokens=2000)
    assert budget.total == 10000, f"Expected total 10000, got {budget.total}"
    assert budget.validate(), "Budget should be valid"
    print(f"  ✓ Basic budget: {budget}")

    # Test invalid budget
    invalid = TokenBudget(reasoning_tokens=-100, output_tokens=2000)
    assert not invalid.validate(), "Negative budget should be invalid"
    print("  ✓ Validation works correctly")


def test_allocation_strategies() -> None:
    """Test all allocation strategies."""
    print("\nTesting AllocationStrategy...")

    total_budget = 10000

    # Test Deep Thinker (80/20)
    deep = AllocationStrategy.DEEP_THINKER.create_budget(total_budget)
    assert deep.reasoning_tokens == 8000, f"Expected 8000, got {deep.reasoning_tokens}"
    assert deep.output_tokens == 2000, f"Expected 2000, got {deep.output_tokens}"
    print(f"  ✓ Deep Thinker: {deep}")
    print(f"    {AllocationStrategy.DEEP_THINKER.description()}")

    # Test Balanced (50/50)
    balanced = AllocationStrategy.BALANCED.create_budget(total_budget)
    assert balanced.reasoning_tokens == 5000, (
        f"Expected 5000, got {balanced.reasoning_tokens}"
    )
    assert balanced.output_tokens == 5000, (
        f"Expected 5000, got {balanced.output_tokens}"
    )
    print(f"  ✓ Balanced: {balanced}")

    # Test Verbose (20/80)
    verbose = AllocationStrategy.VERBOSE.create_budget(total_budget)
    assert verbose.reasoning_tokens == 2000, (
        f"Expected 2000, got {verbose.reasoning_tokens}"
    )
    assert verbose.output_tokens == 8000, f"Expected 8000, got {verbose.output_tokens}"
    print(f"  ✓ Verbose: {verbose}")


def test_agent_factory() -> None:
    """Test AgentFactory initialization and budget info."""
    print("\nTesting AgentFactory...")

    factory = AgentFactory(model="gemini-2.5-flash-lite")
    print(f"  ✓ Factory created with model: {factory.model}")

    # Test budget info for each strategy
    for strategy in AllocationStrategy:
        info = factory.get_budget_info(strategy, total_budget=10000)
        print(f"  ✓ {strategy.value} budget info:")
        print(
            f"    - Reasoning: {info['reasoning_tokens']} ({info['reasoning_percentage']:.0f}%)"
        )
        print(
            f"    - Output: {info['output_tokens']} ({info['output_percentage']:.0f}%)"
        )


def test_metrics() -> None:
    """Test metrics classes."""
    print("\nTesting Metrics...")

    # Test ToolUsageMetrics
    tool_metrics = ToolUsageMetrics(
        tool_name="google_search", call_count=3, total_tokens=150
    )
    tool_dict = tool_metrics.to_dict()
    assert tool_dict["tool"] == "google_search"
    assert tool_dict["calls"] == 3
    print(f"  ✓ ToolUsageMetrics: {tool_metrics}")

    # Test AgentMetrics
    agent_metrics = AgentMetrics(
        strategy="deep",
        reasoning_tokens_used=7500,
        output_tokens_used=1800,
        total_tokens_used=9300,
        tool_usage=[tool_metrics],
        duration_seconds=5.2,
    )
    assert agent_metrics.total_tool_calls == 3
    print(f"  ✓ AgentMetrics: {agent_metrics}")

    metrics_dict = agent_metrics.to_dict()
    assert metrics_dict["strategy"] == "deep"
    assert metrics_dict["total_tool_calls"] == 3
    print("  ✓ Metrics serialization works")


def test_usage_monitor() -> None:
    """Test UsageMonitor."""
    print("\nTesting UsageMonitor...")

    monitor = UsageMonitor()
    print("  ✓ UsageMonitor created")

    # Test comparison with sample metrics
    metrics_list = [
        AgentMetrics("deep", 8000, 2000, 10000, [], 5.0),
        AgentMetrics("balanced", 5000, 5000, 10000, [], 4.5),
        AgentMetrics("verbose", 2000, 8000, 10000, [], 6.0),
    ]

    comparison = monitor.compare_strategies(metrics_list)
    assert len(comparison["strategies"]) == 3
    assert comparison["avg_tokens"] == 10000
    print(f"  ✓ Strategy comparison: {comparison['strategies']}")
    print(f"    Average duration: {comparison['avg_duration']:.2f}s")


def main() -> None:
    """Run all tests."""
    print("=" * 60)
    print("Core Framework Test Suite")
    print("=" * 60)

    try:
        test_token_budget()
        test_allocation_strategies()
        test_agent_factory()
        test_metrics()
        test_usage_monitor()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure all dependencies are installed")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
