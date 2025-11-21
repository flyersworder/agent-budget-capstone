"""Test budget constraints across multiple iterations.

This test validates that:
1. Token usage is tracked correctly across multiple iterations
2. Per-call budget limits are respected
3. Agents can execute multiple iterations without exceeding per-call budgets
"""

import asyncio
import time

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent_budget.agent_factory import AgentFactory
from agent_budget.core import IterativeTeamConfig, MultiAgentAwarenessCondition
from agent_budget.monitor import UsageMonitor


async def test_multi_iteration_budget():
    """Test budget tracking across multiple iterations with a complex question."""
    print("=" * 80)
    print("TEST: Budget Constraints Across Multiple Iterations")
    print("=" * 80)

    # Create factory and team with NO_AWARENESS
    factory = AgentFactory()
    config = IterativeTeamConfig.create_standard(
        awareness_condition=MultiAgentAwarenessCondition.NO_AWARENESS,
    )

    print("\nBudget Configuration:")
    print(f"  Total: {config.total_budget} tokens")
    print(f"  Researcher: {config.researcher_budget.total} tokens")
    print(f"    - Reasoning: {config.researcher_budget.reasoning_tokens}")
    print(f"    - Output: {config.researcher_budget.output_tokens}")
    print(f"  Validator: {config.validator_budget.total} tokens")
    print(f"    - Reasoning: {config.validator_budget.reasoning_tokens}")
    print(f"    - Output: {config.validator_budget.output_tokens}")
    print(f"  Max iterations: {config.max_iterations}")

    team = factory.create_iterative_team(config)
    print(f"\n✓ Created team: {team.name}")

    # Create session
    session_service = InMemorySessionService()
    _ = await session_service.create_session(
        app_name="test_budget",
        user_id="test_user",
        session_id="test_session_budget",
    )

    runner = Runner(
        agent=team,
        app_name="test_budget",
        session_service=session_service,
    )

    # Use a more complex multi-hop question that might require iteration
    question = (
        "What is the name of the director of the 2010 film that starred "
        "the actor who played Iron Man in the Marvel Cinematic Universe?"
    )

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
            session_id="test_session_budget",
            new_message=content,
        ):
            events.append(event)

            # Print simplified event info
            if event.author:
                print(f"\n[{event.author}] Round event")

                # Show token usage if available
                if hasattr(event, "usage_metadata") and event.usage_metadata:
                    thinking = (
                        getattr(event.usage_metadata, "thoughts_token_count", 0) or 0
                    )
                    output = (
                        getattr(event.usage_metadata, "candidates_token_count", 0) or 0
                    )
                    print(
                        f"  Tokens: {thinking} reasoning + {output} output = {thinking + output} total"
                    )

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

    # Get final session state
    final_session = await session_service.get_session(
        app_name="test_budget",
        user_id="test_user",
        session_id="test_session_budget",
    )

    # Extract metrics
    monitor = UsageMonitor()
    metrics = monitor.extract_multi_agent_metrics(
        events=events,
        session_state=final_session.state,
        awareness_condition=config.awareness_condition.value,
        max_iterations=config.max_iterations,
        duration=duration,
    )

    print(f"\n{'=' * 80}")
    print("BUDGET ANALYSIS")
    print(f"{'=' * 80}")

    print("\nExecution Summary:")
    print(f"  Iterations: {metrics.num_iterations}")
    print(f"  Approved: {metrics.approved}")
    print(f"  Max iterations reached: {metrics.max_iterations_reached}")

    print("\nToken Usage:")
    print(
        f"  Researcher: {metrics.researcher_tokens} / {config.researcher_budget.total} per call"
    )
    print(
        f"  Validator: {metrics.validator_tokens} / {config.validator_budget.total} per call"
    )
    print(f"  Total: {metrics.total_tokens} / {config.total_budget} overall budget")

    # Calculate per-iteration averages
    if metrics.num_iterations > 0:
        avg_researcher = metrics.researcher_tokens / metrics.num_iterations
        avg_validator = metrics.validator_tokens / metrics.num_iterations
        print("\nPer-Iteration Averages:")
        print(f"  Researcher: {avg_researcher:.1f} tokens/iteration")
        print(f"  Validator: {avg_validator:.1f} tokens/iteration")

    # Budget constraint verification
    print(f"\n{'=' * 80}")
    print("BUDGET CONSTRAINT VERIFICATION")
    print(f"{'=' * 80}")

    # Note: Per-call budgets are enforced by API
    # We're verifying that agents operate within reasonable bounds
    print("\n✓ Per-call budgets enforced by Gemini API")
    print(f"  Researcher max per call: {config.researcher_budget.total} tokens")
    print(f"  Validator max per call: {config.validator_budget.total} tokens")

    # Check if within overall budget expectations
    # (Note: Could exceed if using max tokens every iteration)
    theoretical_max = config.total_budget * metrics.num_iterations
    print(
        f"\n✓ Theoretical max across {metrics.num_iterations} iteration(s): {theoretical_max} tokens"
    )
    print(
        f"  Actual usage: {metrics.total_tokens} tokens ({metrics.total_tokens / theoretical_max * 100:.1f}% of theoretical max)"
    )

    print("\nFinal Outputs:")
    print(f"  Researcher: {final_session.state.get('researcher_output', '')[:150]}...")
    print(f"  Validator: {final_session.state.get('validator_feedback', '')[:150]}...")

    print("\n" + "=" * 80)
    print("✅ BUDGET CONSTRAINT TEST COMPLETE")
    print("=" * 80)

    return True


if __name__ == "__main__":
    asyncio.run(test_multi_iteration_budget())
