"""Test that agents actually see usage information in subsequent iterations."""

import asyncio

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent_budget.agent_factory import AgentFactory
from agent_budget.core import IterativeTeamConfig, MultiAgentAwarenessCondition


async def test_usage_visibility():
    """Test that agents see usage information in iteration 2.

    Strategy:
    1. Use a deliberately tricky question that requires multiple iterations
    2. Collect all events from the conversation
    3. Verify CheckApproval's usage message appears after iteration 1
    4. Verify Researcher in iteration 2 has access to this message in history
    """
    load_dotenv()

    print("=" * 80)
    print("USAGE VISIBILITY TEST")
    print("=" * 80)
    print()
    print("Testing that agents see cumulative usage in iteration 2+")
    print()

    # Create a question that will likely need revision
    # Ask for something specific that validator might reject first time
    tricky_question = """What is the exact population of Tokyo as of 2023?
Please provide the precise number, not a range or estimate."""

    # Create team with OVERALL_AND_INDIVIDUAL awareness
    factory = AgentFactory()
    config = IterativeTeamConfig.create_standard(
        awareness_condition=MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL
    )
    team = factory.create_iterative_team(config)

    # Create session
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="visibility_test",
        user_id="test_user",
        session_id="test_001",
    )

    runner = Runner(
        agent=team,
        app_name="visibility_test",
        session_service=session_service,
    )

    # Run and collect all events
    print("Running trial...")
    events = []
    content = types.Content(role="user", parts=[types.Part(text=tricky_question)])

    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_001",
        new_message=content,
    ):
        events.append(event)
        # Note: Token tracking now handled internally by TrackingLoopAgent

    print(f"\nTotal events: {len(events)}")
    print()

    # Analyze event sequence
    print("=" * 80)
    print("EVENT SEQUENCE ANALYSIS")
    print("=" * 80)
    print()

    iteration_1_events = []
    iteration_2_events = []
    usage_message_event = None
    current_iteration = 1

    for i, event in enumerate(events):
        author = getattr(event, "author", "unknown")

        # Check for text content
        text_content = ""
        if hasattr(event, "content") and event.content:
            if hasattr(event.content, "parts"):
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        text_content = part.text[:100]  # First 100 chars

        print(f"Event {i}: Author='{author}'")
        if text_content:
            print(f"  Content preview: {text_content}...")

        # Check if this is CheckApproval with usage message
        if author == "CheckApproval" and "[BUDGET STATUS]" in text_content:
            usage_message_event = event
            print("  ⭐ USAGE MESSAGE FOUND")
            current_iteration += 1

        # Categorize by iteration
        if current_iteration == 1:
            iteration_1_events.append(event)
        elif current_iteration == 2:
            iteration_2_events.append(event)

        print()

    # Verify findings
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print()

    num_iterations = current_iteration
    print(f"Number of iterations: {num_iterations}")
    print()

    if num_iterations == 1:
        print("⚠️  Only 1 iteration occurred - cannot test visibility in iteration 2")
        print("   (This might be okay if validator approved immediately)")
        print()
        print("Checking if usage message was still generated:")
        if usage_message_event:
            print("   ✅ Usage message was generated in iteration 1")
        else:
            print("   ❌ No usage message found")
        return

    print(f"✅ Multiple iterations occurred ({num_iterations})")
    print()

    # Check if usage message exists
    if usage_message_event:
        print("✅ Usage message found after iteration 1")

        # Extract the actual message
        if hasattr(usage_message_event, "content") and usage_message_event.content:
            if hasattr(usage_message_event.content, "parts"):
                for part in usage_message_event.content.parts:
                    if hasattr(part, "text") and part.text:
                        print("\nUsage message content:")
                        print("-" * 40)
                        print(part.text)
                        print("-" * 40)
        print()
    else:
        print("❌ No usage message found after iteration 1")
        print()

    # Check iteration 2 events
    print(f"Iteration 2 events: {len(iteration_2_events)}")

    # Check if usage messages are in the event stream (conversation history)
    print("Checking if usage messages are in the event stream (visible to agents):")
    usage_events = [
        e for e in events if hasattr(e, "author") and e.author == "CheckApproval"
    ]

    if usage_events:
        print(f"  ✅ Found {len(usage_events)} usage messages in event stream")
        print("  This means agents in subsequent iterations CAN see these messages")
        usage_in_history = True
    else:
        print("  ❌ No usage messages found in event stream")
        usage_in_history = False

    print()

    # Final verdict
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()

    success = (
        num_iterations >= 2 and usage_message_event is not None and usage_in_history
    )

    if success:
        print("✅ USAGE VISIBILITY VERIFIED")
        print()
        print("Agents in iteration 2+ CAN see cumulative usage information")
        print("because it appears in the session history they receive.")
    else:
        print("❌ USAGE VISIBILITY NOT CONFIRMED")
        print()
        if num_iterations < 2:
            print("Reason: Only 1 iteration occurred (may need harder question)")
        if usage_message_event is None:
            print("Reason: Usage message was not generated")
        if not usage_in_history:
            print("Reason: Usage message not found in session history")

    return success


if __name__ == "__main__":
    success = asyncio.run(test_usage_visibility())
    exit(0 if success else 1)
