"""Test structured output with built-in tools and ThinkingConfig.

This test verifies whether Pydantic output_schema is compatible with:
1. Built-in google_search tool
2. ThinkingConfig (thinking budget)
"""

import asyncio
import os

from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from pydantic import BaseModel, Field


class ResearchOutput(BaseModel):
    """Structured output for a simple research task."""

    answer: str = Field(description="The answer to the research question")
    budget_note: str | None = Field(
        default=None,
        description="Optional note about resource needs (e.g., 'This question requires additional verification')",
    )


async def test_structured_output_with_tools_and_thinking() -> None:
    """Test if output_schema works with google_search and ThinkingConfig."""

    # Load environment variables
    load_dotenv()

    # Verify API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY not found in environment")
        return False

    print("=" * 80)
    print("TESTING: Structured Output + Built-in Tool + ThinkingConfig")
    print("=" * 80)

    # Create agent with structured output
    agent = LlmAgent(
        model="gemini-2.5-flash-lite",
        name="research_agent",
        description="Research agent with structured output",
        instruction="""You are a research agent. Answer questions using google_search.

Respond with a JSON object:
- answer: Your answer to the question
- budget_note: (optional) If question is complex, note resource needs

Example:
{
  "answer": "Paris is the capital of France",
  "budget_note": "Simple factual question, no additional resources needed"
}
""",
        tools=[google_search],
        planner=BuiltInPlanner(
            thinking_config=types.ThinkingConfig(
                thinking_budget=1024,  # Moderate thinking budget
                include_thoughts=True,
            )
        ),
        output_schema=ResearchOutput,  # This is what we're testing
        output_key="research_result",
    )

    # Create session
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="structured_output_test",
        user_id="tester",
        session_id="test_001",
    )

    runner = Runner(
        agent=agent,
        app_name="structured_output_test",
        session_service=session_service,
    )

    # Test question
    question = "What is the capital of France?"
    print(f"\nQuestion: {question}")
    print("\nRunning agent with structured output...\n")

    try:
        content = types.Content(role="user", parts=[types.Part(text=question)])

        events = []
        async for event in runner.run_async(
            user_id="tester",
            session_id="test_001",
            new_message=content,
        ):
            events.append(event)

            # Print progress
            if hasattr(event, "author") and event.author:
                print(f"  Event from: {event.author}")

        # Get final session state
        final_session = await session_service.get_session(
            app_name="structured_output_test",
            user_id="tester",
            session_id="test_001",
        )

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        # Check if structured output was stored
        if "research_result" in final_session.state:
            result = final_session.state["research_result"]
            print("\n✅ Structured output received!")
            print(f"\nResult type: {type(result)}")
            print("\nResult content:")
            print(
                f"  answer: {result.get('answer', 'N/A') if isinstance(result, dict) else result}"
            )
            print(
                f"  budget_note: {result.get('budget_note', 'N/A') if isinstance(result, dict) else 'N/A'}"
            )
        else:
            print("\n❌ No structured output in session state")
            print(f"\nSession state keys: {list(final_session.state.keys())}")

        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)

        # Return success status
        return "research_result" in final_session.state

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_structured_output_with_tools_and_thinking())

    if success:
        print("\n✅ Structured output works with tools + ThinkingConfig!")
    else:
        print("\n❌ Structured output NOT compatible with tools + ThinkingConfig")
