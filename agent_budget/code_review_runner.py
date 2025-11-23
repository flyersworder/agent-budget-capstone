"""Code Review Study Framework: Iterative Coder-Reviewer Teams with Budget Awareness.

This module implements the experimental framework for studying budget awareness
effects in iterative code review teams using ADK best practices.

Architecture:
- Coder: LlmAgent that writes/revises Python code (with google_search tool)
- Reviewer: LlmAgent that tests and reviews code (with code_execution tool)
- CheckApproval: Custom BaseAgent that escalates when code is approved
- LoopAgent: Manages iterative refinement (max 3 iterations)

Conditions:
- Budget Awareness: YES vs NO
- Task Complexity: MEDIUM vs HARD (LiveCodeBench problems)

Best Practices:
- Uses LoopAgent for iteration management
- Session state for data persistence ({{current_code}}, {{feedback}})
- Escalation pattern for loop termination
- Proper session management with InMemorySessionService
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from agent_budget.agent_factory import AgentFactory
from agent_budget.core import MultiAgentAwarenessCondition


@dataclass
class CodeReviewTrial:
    """Results from a single code review trial."""

    problem_id: str
    problem_title: str
    difficulty: str
    awareness_condition: MultiAgentAwarenessCondition

    # Outcomes
    success: bool  # Final code approved
    num_iterations: int
    final_decision: str  # APPROVE or MAX_ITERATIONS_REACHED

    # Token usage (approximate)
    team_total_tokens: int

    # Iteration details
    iterations: list[dict[str, Any]]

    # Metadata
    test_passed: bool


def execute_python_code(code: str) -> str:
    """Execute Python code and return the output.

    Args:
        code: Python code to execute

    Returns:
        str: Output from code execution including stdout, stderr, and exit code
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True,
            timeout=10,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        output += f"EXIT CODE: {result.returncode}"

        return output

    except subprocess.TimeoutExpired:
        return "ERROR: Code execution timed out after 10 seconds"
    except Exception as e:
        return f"ERROR: {str(e)}"
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def run_code_review_trial(
    problem: dict[str, Any],
    awareness_condition: MultiAgentAwarenessCondition,
    max_iterations: int = 3,
) -> CodeReviewTrial:
    """Run a single code review trial using LoopAgent pattern.

    Args:
        problem: LiveCodeBench problem dict
        awareness_condition: Budget awareness level
        max_iterations: Maximum iterations allowed (default: 3)

    Returns:
        CodeReviewTrial with full results
    """
    # Parse test case
    import json

    public_tests = json.loads(problem["public_test_cases"])
    test_input = public_tests[0]["input"]
    expected_output = public_tests[0]["output"]

    # Create a dedicated test function for the Reviewer
    # IMPORTANT: Takes NO arguments - reads code from session state to save tokens
    def test_code(tool_context: Any = None) -> str:
        """Test the code with the problem's test case.

        Reads code from session state key 'current_code'.

        Args:
            tool_context: Automatically injected by FunctionTool

        Returns:
            str: Test result (PASS or FAIL with details)
        """
        import subprocess
        import tempfile
        import os

        # Get code from session state via tool_context
        if not tool_context or not hasattr(tool_context, "session"):
            return "FAIL: No tool context available"

        code = tool_context.session.state.get("current_code", "")

        if not code:
            return "FAIL: No code found in session state"

        # Strip markdown code block markers if present
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python") :].lstrip()
        if code.startswith("```"):
            code = code[3:].lstrip()
        if code.endswith("```"):
            code = code[:-3].rstrip()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["python", temp_file],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=5,
            )
            actual = result.stdout.strip()
            expected = expected_output.strip()

            if result.returncode != 0:
                return f"FAIL: Code crashed with exit code {result.returncode}\nSTDERR: {result.stderr}"
            elif actual == expected:
                return "PASS: All tests passed!"
            else:
                return f"FAIL: Output mismatch\nExpected: {expected[:200]}\nActual: {actual[:200]}"
        except subprocess.TimeoutExpired:
            return "FAIL: Code execution timed out after 5 seconds"
        except Exception as e:
            return f"FAIL: {str(e)}"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    # Create test_code tool for this specific problem
    test_code_tool = FunctionTool(func=test_code)

    # Use factory to create code review team
    factory = AgentFactory(model="gemini-2.5-flash-lite")
    review_loop = factory.create_code_review_team(
        problem_description=problem["question_content"],
        test_code_tool=test_code_tool,
        awareness_condition=awareness_condition,
        max_iterations=max_iterations,
    )

    # Session setup
    session_service = InMemorySessionService()
    runner = Runner(
        agent=review_loop, app_name="code_review_study", session_service=session_service
    )

    # Create session
    session = await session_service.create_session(
        app_name="code_review_study",
        user_id="study",
    )

    # Initialize session state with empty review_decision for first iteration
    session.state["review_decision"] = ""

    # Run the loop
    initial_message = types.Content(
        role="user",
        parts=[
            types.Part(
                text="Please write the Python code to solve the problem. Return ONLY code, no explanations."
            )
        ],
    )

    iterations = []
    num_iterations = 0
    conversation_log = []

    print("\n" + "=" * 80)
    print("STARTING CODE REVIEW ITERATIONS")
    print("=" * 80)

    async for event in runner.run_async(
        user_id="study", session_id=session.id, new_message=initial_message
    ):
        # Log all events with details
        if hasattr(event, "author") and event.author:
            print(f"\n{'─' * 80}")
            print(f"EVENT FROM: {event.author}")

            # Extract and log content
            if hasattr(event, "content") and event.content:
                content_text = ""
                function_calls = []
                function_responses = []

                # Extract parts from content
                if hasattr(event.content, "parts") and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            content_text += part.text
                        if hasattr(part, "function_call") and part.function_call:
                            function_calls.append(part.function_call)
                        if (
                            hasattr(part, "function_response")
                            and part.function_response
                        ):
                            function_responses.append(part.function_response)

                if content_text:
                    print(
                        f"CONTENT: {content_text[:500]}{'...' if len(content_text) > 500 else ''}"
                    )

                if function_calls:
                    print(f"FUNCTION CALLS: {len(function_calls)} call(s)")
                    for fc in function_calls:
                        print(f"  - {fc.name}")

                if function_responses:
                    print(f"FUNCTION RESPONSES: {len(function_responses)} response(s)")
                    for fr in function_responses:
                        response_preview = (
                            str(fr.response)[:300]
                            if hasattr(fr, "response")
                            else "(no response)"
                        )
                        print(
                            f"  - {fr.name}: {response_preview}{'...' if len(str(fr.response)) > 300 else ''}"
                        )

                conversation_log.append(
                    {
                        "author": event.author,
                        "content": content_text,
                        "function_calls": len(function_calls),
                    }
                )

            # Log token usage if available
            if hasattr(event, "usage_metadata") and event.usage_metadata:
                thinking = getattr(event.usage_metadata, "thoughts_token_count", 0) or 0
                candidates = (
                    getattr(event.usage_metadata, "candidates_token_count", 0) or 0
                )
                if thinking > 0 or candidates > 0:
                    print(f"TOKENS: thinking={thinking}, output={candidates}")

            # Log state changes if available
            if hasattr(event, "actions") and event.actions:
                if hasattr(event.actions, "state_delta") and event.actions.state_delta:
                    print(f"STATE DELTA: {list(event.actions.state_delta.keys())}")
                    for key, value in event.actions.state_delta.items():
                        value_preview = str(value)[:100] if value else "(empty)"
                        print(f"  {key}: {value_preview}")

            # Track iterations by monitoring Coder responses
            if event.author == "Coder":
                num_iterations += 1
                print(f"\n🔄 ITERATION {num_iterations} STARTED")

    print("\n" + "=" * 80)
    print(f"ITERATIONS COMPLETE: {num_iterations} total")
    print("=" * 80)

    # CRITICAL: Fetch updated session to get committed state changes
    # The session object is updated in-place during the run, but we need to ensure we have the latest
    updated_session = await session_service.get_session(
        app_name="code_review_study",
        user_id="study",
        session_id=session.id,
    )

    # Debug: Show all session state keys
    print(f"\nALL SESSION STATE KEYS: {list(updated_session.state.keys())}")
    print("SESSION STATE VALUES:")
    for key in updated_session.state.keys():
        value = updated_session.state[key]
        value_preview = str(value)[:100] if value else "(empty)"
        print(f"  {key}: {value_preview}")

    # Extract final results from session state
    final_code = updated_session.state.get("current_code", "")
    final_decision_text = updated_session.state.get("review_decision", "")

    print("\n" + "=" * 80)
    print("FINAL OUTPUTS")
    print("=" * 80)
    print(f"\nFINAL CODE ({len(final_code)} chars):")
    print("─" * 80)
    print(final_code if final_code else "(no code generated)")
    print("─" * 80)
    print("\nFINAL REVIEW DECISION:")
    print("─" * 80)
    print(final_decision_text if final_decision_text else "(no decision)")
    print("─" * 80)
    print("\nSESSION STATE TOKEN USAGE:")
    print(f"  Coder total: {updated_session.state.get('Coder_total_tokens', 0)}")
    print(f"  Reviewer total: {updated_session.state.get('Reviewer_total_tokens', 0)}")
    print(
        f"  Team total: {updated_session.state.get('Coder_total_tokens', 0) + updated_session.state.get('Reviewer_total_tokens', 0)}"
    )

    # Determine success
    approved = "APPROVE" in final_decision_text.upper()
    if approved:
        final_decision = "APPROVE"
        success = True
    else:
        final_decision = "MAX_ITERATIONS_REACHED"
        success = False

    # Approximate token usage (simple heuristic)
    team_total_tokens = (
        len(problem["question_content"].split()) * num_iterations * 2
        + len(final_code.split()) * 2
        + len(final_decision_text.split()) * 2
    )

    # Build iteration details (simplified - LoopAgent doesn't expose iteration history easily)
    iterations = [
        {"iteration": i + 1, "status": "completed"} for i in range(num_iterations)
    ]

    return CodeReviewTrial(
        problem_id=problem.get("question_id", "unknown"),
        problem_title=problem.get("question_title", "unknown"),
        difficulty=problem.get("difficulty", "unknown"),
        awareness_condition=awareness_condition,
        success=success,
        num_iterations=num_iterations,
        final_decision=final_decision,
        team_total_tokens=team_total_tokens,
        iterations=iterations,
        test_passed=success,
    )
