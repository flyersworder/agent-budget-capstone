"""Validation tests for AgentFactory.

Tests factory functionality for both Part 1 and Part 2 configurations.
"""

from agent_budget.agent_factory import AgentFactory
from agent_budget.awareness import AwarenessCondition, BUDGET_MODERATE
from agent_budget.core import (
    AgentRole,
    MultiAgentAwarenessCondition,
    MultiAgentBudgetConfig,
)


def test_part1_single_agents():
    """Test Part 1 single agent creation."""
    print("=" * 80)
    print("TEST: Part 1 Single Agent Creation")
    print("=" * 80)

    factory = AgentFactory()

    # Test both awareness conditions
    for condition in [AwarenessCondition.UNAWARE, AwarenessCondition.AWARE]:
        agent = factory.create_single_agent(
            condition=condition, budget_config=BUDGET_MODERATE
        )

        print(f"\n✓ Created {condition.value} agent:")
        print(f"  Name: {agent.name}")
        print(f"  Model: {agent.model}")
        print(f"  Budget: {BUDGET_MODERATE.total} tokens")
        print(f"  Instruction length: {len(agent.instruction)} chars")

        # Verify config
        assert agent.name == f"{condition.value}_agent"
        assert agent.model == "gemini-2.5-flash-lite"
        assert hasattr(agent, "planner")


def test_part2_multiagent_teams():
    """Test Part 2 multi-agent team creation."""
    print("\n" + "=" * 80)
    print("TEST: Part 2 Multi-Agent Team Creation")
    print("=" * 80)

    factory = AgentFactory()
    total_budget = 1280

    # Test all 4 awareness conditions
    test_configs = [
        (
            "Condition A: No Awareness",
            MultiAgentBudgetConfig.create_equal_split(
                total_budget,
                awareness_condition=MultiAgentAwarenessCondition.NO_AWARENESS,
            ),
        ),
        (
            "Condition B: Overall Only",
            MultiAgentBudgetConfig.create_equal_split(
                total_budget,
                awareness_condition=MultiAgentAwarenessCondition.OVERALL_ONLY,
            ),
        ),
        (
            "Condition C: Overall + Individual",
            MultiAgentBudgetConfig.create_role_based(total_budget),
        ),
        (
            "Condition D: With Negotiation",
            MultiAgentBudgetConfig.create_with_negotiation(total_budget),
        ),
    ]

    for label, config in test_configs:
        print(f"\n{label}:")
        print(f"  Total budget: {config.total_budget} tokens")
        print(f"  Allocated: {config.allocated_budget} tokens")
        print(f"  Reserve pool: {config.reserve_pool} tokens")

        # Validate config
        assert config.validate(), f"Invalid config for {label}"

        # Create team
        team = factory.create_multiagent_team(config)

        print(f"  ✓ Team name: {team.name}")
        print(f"  ✓ Sub-agents: {len(team.sub_agents)}")

        # Verify team structure
        assert len(team.sub_agents) == 3, (
            f"Expected 3 agents, got {len(team.sub_agents)}"
        )

        # Check roles
        roles = [agent.name for agent in team.sub_agents]
        expected_roles = ["researcher", "analyzer", "synthesizer"]
        assert roles == expected_roles, f"Role mismatch: {roles} vs {expected_roles}"

        # Print agent details
        for agent in team.sub_agents:
            alloc = config.allocations[AgentRole(agent.name)]
            print(
                f"    - {agent.name}: {alloc.budget.total} tokens ({alloc.percentage:.0f}%)"
            )


def test_budget_configurations():
    """Test budget configuration methods."""
    print("\n" + "=" * 80)
    print("TEST: Budget Configurations")
    print("=" * 80)

    total = 1280

    # Test equal split
    equal_config = MultiAgentBudgetConfig.create_equal_split(total)
    print(f"\nEqual Split (total={total}):")
    for role, alloc in equal_config.allocations.items():
        print(f"  {role.value}: {alloc.budget.total} tokens ({alloc.percentage:.0f}%)")
    assert equal_config.validate(), "Equal split config invalid"

    # Test role-based
    role_config = MultiAgentBudgetConfig.create_role_based(total)
    print(f"\nRole-Based (total={total}):")
    for role, alloc in role_config.allocations.items():
        print(f"  {role.value}: {alloc.budget.total} tokens ({alloc.percentage:.0f}%)")
    assert role_config.validate(), "Role-based config invalid"

    # Test with negotiation
    neg_config = MultiAgentBudgetConfig.create_with_negotiation(total)
    print(f"\nWith Negotiation (total={total}):")
    print(f"  Allocated: {neg_config.allocated_budget} tokens")
    print(f"  Reserve pool: {neg_config.reserve_pool} tokens")
    for role, alloc in neg_config.allocations.items():
        print(f"  {role.value}: {alloc.budget.total} tokens ({alloc.percentage:.0f}%)")
    assert neg_config.validate(), "Negotiation config invalid"


def test_legacy_support():
    """Test legacy allocation strategy support."""
    print("\n" + "=" * 80)
    print("TEST: Legacy Allocation Strategies (Archived Part 1)")
    print("=" * 80)

    from agent_budget.core import AllocationStrategy

    factory = AgentFactory()

    for strategy in AllocationStrategy:
        agent = factory.create_agent(strategy, total_budget=3000)
        print(f"\n✓ Created {strategy.value} agent (legacy):")
        print(f"  Name: {agent.name}")

        budget_info = factory.get_budget_info(strategy, total_budget=3000)
        print(
            f"  Reasoning: {budget_info['reasoning_tokens']} tokens ({budget_info['reasoning_percentage']:.0f}%)"
        )
        print(
            f"  Output: {budget_info['output_tokens']} tokens ({budget_info['output_percentage']:.0f}%)"
        )


def main():
    """Run all validation tests."""
    print("\n" + "=" * 80)
    print("AGENT FACTORY VALIDATION SUITE")
    print("=" * 80)

    try:
        test_part1_single_agents()
        test_part2_multiagent_teams()
        test_budget_configurations()
        test_legacy_support()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
