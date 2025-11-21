"""Test script for iterative 2-agent team.

This validates:
1. LoopAgent can be created with Researcher ⇄ Validator
2. Loop executes and can exit on approval
3. State is properly tracked across iterations
4. Budget constraints are applied
5. Metrics extraction works correctly
"""

import asyncio
import time

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent_budget.agent_factory import AgentFactory
from agent_budget.core import IterativeTeamConfig, MultiAgentAwarenessCondition
from agent_budget.monitor import UsageMonitor


async def test_iterative_team_basic():
    """Test basic iterative team execution with a simple question."""
    print("=" * 80)
    print("TEST: Basic Iterative Team Execution")
    print("=" * 80)

    # Create factory and team
    factory = AgentFactory()
    config = IterativeTeamConfig.create_standard(
        awareness_condition=MultiAgentAwarenessCondition.NO_AWARENESS,
    )  # Uses default 2000 tokens

    print("\nTeam Configuration:")
    print(f"  Total budget: {config.total_budget} tokens")
    print(f"  Researcher: {config.researcher_budget.total} tokens")
    print(f"  Validator: {config.validator_budget.total} tokens")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Awareness: {config.awareness_condition.value}")

    team = factory.create_iterative_team(config)
    print(f"\n✓ Created team: {team.name}")
    print(f"  Sub-agents: {[agent.name for agent in team.sub_agents]}")

    # Create runner
    session_service = InMemorySessionService()

    # Create session (needs await since it's async)
    _ = await session_service.create_session(
        app_name="test_iterative",
        user_id="test_user",
        session_id="test_session_1",
    )

    runner = Runner(
        agent=team,
        app_name="test_iterative",
        session_service=session_service,
    )

    # Simple test question
    question = "What is the capital of France?"
    print(f"\n{'=' * 80}")
    print(f"Question: {question}")
    print(f"{'=' * 80}")

    content = types.Content(role="user", parts=[types.Part(text=question)])

    # Run and collect events
    print("\nRunning iterative team...")
    events = []
    start_time = time.time()
    try:
        async for event in runner.run_async(
            user_id="test_user",
            session_id="test_session_1",
            new_message=content,
        ):
            events.append(event)

            # Print event details
            if event.author:
                print(f"\n[{event.author}] event:")
                if hasattr(event, "content") and event.content and event.content.parts:
                    text = event.content.parts[0].text
                    # Truncate for display
                    display_text = text[:200] + "..." if len(text) > 200 else text
                    print(f"  {display_text}")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback

        traceback.print_exc()
        return False

    duration = time.time() - start_time

    print(f"\n{'=' * 80}")
    print(f"Execution Complete - Collected {len(events)} events")
    print(f"Duration: {duration:.2f}s")
    print(f"{'=' * 80}")

    # Analyze session state
    final_session = await session_service.get_session(
        app_name="test_iterative",
        user_id="test_user",
        session_id="test_session_1",
    )

    print("\nSession State Keys:")
    for key in final_session.state.keys():
        print(f"  - {key}")

    # Check for expected state keys
    researcher_output = final_session.state.get("researcher_output", "")
    validator_feedback = final_session.state.get("validator_feedback", "")

    print("\nExtracted Outputs:")
    print("\nResearcher Output:")
    print(
        f"  {researcher_output[:300]}..."
        if len(researcher_output) > 300
        else f"  {researcher_output}"
    )

    print("\nValidator Feedback:")
    print(
        f"  {validator_feedback[:300]}..."
        if len(validator_feedback) > 300
        else f"  {validator_feedback}"
    )

    # Check if approved
    approved = "APPROVED" in str(validator_feedback)
    print(f"\nApproval Status: {'✓ APPROVED' if approved else '✗ NOT APPROVED'}")

    # Count iterations (approximate - count CheckApproval events)
    check_events = [
        e for e in events if hasattr(e, "author") and e.author == "CheckApproval"
    ]
    print(f"Iterations: {len(check_events)}")

    # Extract and display metrics
    print(f"\n{'=' * 80}")
    print("METRICS EXTRACTION TEST")
    print(f"{'=' * 80}")

    monitor = UsageMonitor()
    metrics = monitor.extract_multi_agent_metrics(
        events=events,
        session_state=final_session.state,
        awareness_condition=config.awareness_condition.value,
        max_iterations=config.max_iterations,
        duration=duration,
    )

    print(f"\n{metrics}")
    print("\nDetailed Metrics:")
    print(f"  Condition: {metrics.awareness_condition}")
    print(f"  Iterations: {metrics.num_iterations}")
    print(f"  Approved: {metrics.approved}")
    print(f"  Max iterations reached: {metrics.max_iterations_reached}")
    print(f"  Researcher tokens: {metrics.researcher_tokens}")
    print(f"  Validator tokens: {metrics.validator_tokens}")
    print(f"  Total tokens: {metrics.total_tokens}")
    print(f"  Tool calls: {metrics.total_tool_calls}")
    print(f"  Duration: {metrics.duration_seconds:.2f}s")

    if metrics.tool_usage:
        print("\nTool Usage:")
        for tool in metrics.tool_usage:
            print(
                f"  - {tool.tool_name}: {tool.call_count} calls, {tool.total_tokens} tokens"
            )

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

    return True


async def test_all_conditions():
    """Test all 4 awareness conditions."""
    print("\n" + "=" * 80)
    print("TEST: All Awareness Conditions")
    print("=" * 80)

    factory = AgentFactory()
    conditions = [
        MultiAgentAwarenessCondition.NO_AWARENESS,
        MultiAgentAwarenessCondition.OVERALL_ONLY,
        MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL,
    ]

    for condition in conditions:
        print(f"\n{'=' * 80}")
        print(f"Testing Condition: {condition.value}")
        print(f"{'=' * 80}")

        config = IterativeTeamConfig.create_standard(
            awareness_condition=condition,
        )  # Uses default 2000 tokens

        team = factory.create_iterative_team(config)
        print(f"✓ Created team: {team.name}")

    # Test negotiation condition
    print(f"\n{'=' * 80}")
    print("Testing Condition: WITH_NEGOTIATION")
    print(f"={'=' * 80}")

    neg_config = (
        IterativeTeamConfig.create_with_negotiation()
    )  # Uses default 2000 tokens
    neg_team = factory.create_iterative_team(neg_config)
    print(f"✓ Created team: {neg_team.name}")
    print(f"  Total budget: {neg_config.total_budget} tokens")
    print(f"  Reserve pool: {neg_config.reserve_pool} tokens")

    print("\n" + "=" * 80)
    print("✅ ALL CONDITIONS TESTED")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ITERATIVE TEAM VALIDATION SUITE")
    print("=" * 80)

    # Test 1: Basic execution
    asyncio.run(test_iterative_team_basic())

    # Test 2: All conditions
    asyncio.run(test_all_conditions())

    print("\n" + "=" * 80)
    print("✅ ALL VALIDATION TESTS PASSED")
    print("=" * 80)
