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
from typing import Any, AsyncGenerator

from google.adk.agents import LlmAgent, BaseAgent, LoopAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, FunctionTool
from google.genai import types

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


class CheckApprovalAgent(BaseAgent):  # type: ignore[misc]
    """Custom agent that checks for APPROVE decision and escalates to exit loop.

    Examines state['review_decision'] for "APPROVE" keyword.
    If found, signals LoopAgent to terminate by escalating.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check state for approval and escalate if found.

        Args:
            ctx: Invocation context with session state

        Yields:
            Event with escalate flag if approved
        """
        # Get review decision from state
        decision = ctx.session.state.get("review_decision", "")

        # Check if approval keyword present
        approved = "APPROVE" in str(decision).upper()

        # Yield event with escalate flag
        # escalate=True signals LoopAgent to exit
        yield Event(
            author=self.name,
            actions=EventActions(escalate=approved),
        )


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

    # Build test script template for reviewer
    test_script_template = f"""
import subprocess
import tempfile
import os

code = \"\"\"{{{{current_code}}}}\"\"\"

test_input = \"\"\"{test_input.replace('"', '\\"')}\"\"\"
expected_output = \"\"\"{expected_output.replace('"', '\\"')}\"\"\"

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(code)
    temp_file = f.name

try:
    result = subprocess.run(['python', temp_file], input=test_input, capture_output=True, text=True, timeout=5)
    actual = result.stdout.strip()
    expected = expected_output.strip()

    if result.returncode != 0:
        print("FAILURE: Code crashed with exit code " + str(result.returncode))
        print("STDERR: " + str(result.stderr))
    elif actual == expected:
        print("SUCCESS: All tests passed!")
    else:
        print("FAILURE: Output mismatch")
        print("Expected: " + str(expected[:200]))
        print("Actual: " + str(actual[:200]))
except Exception as e:
    print("ERROR: " + str(e))
finally:
    if os.path.exists(temp_file):
        os.unlink(temp_file)
"""

    # Generate budget awareness message
    budget_message = _generate_budget_message(awareness_condition, max_iterations)

    # Create Coder agent with state integration
    # Feedback comes through conversation history, not state variables
    coder = LlmAgent(
        model="gemini-2.5-flash-lite",
        name="Coder",
        description="Writes or revises Python code to solve programming problems",
        instruction=f"""{budget_message}

Problem to solve:
{problem["question_content"]}

Write a complete Python program that reads from stdin and writes to stdout.
If you see previous review feedback in the conversation, incorporate it to improve the code.
Return ONLY the program code, no explanations.""",
        output_key="current_code",  # Automatically saves to session state
        tools=[google_search],
    )

    # Create Reviewer agent with state integration
    code_execution_tool = FunctionTool(func=execute_python_code)
    reviewer = LlmAgent(
        model="gemini-2.5-flash-lite",
        name="Reviewer",
        description="Tests and reviews code using code execution",
        instruction=f"""{budget_message}

Review the following code:

CODE:
{{{{current_code}}}}

TEST SCRIPT:
{test_script_template}

Instructions:
1. Use code execution to run the test script
2. Analyze the results
3. Make your decision

Decision format:
DECISION: APPROVE or REQUEST_REVISION
FEEDBACK: [specific feedback on what passed/failed or what to improve]

Be concise but thorough.""",
        output_key="review_decision",  # Saves decision to state
        tools=[code_execution_tool],
    )

    # Create CheckApproval agent
    approval_checker = CheckApprovalAgent(
        name="ApprovalChecker",
        description="Checks review decision and escalates if approved",
    )

    # Create LoopAgent for iterative refinement
    review_loop = LoopAgent(
        name="CodeReviewLoop",
        description="Iteratively refines code through Coder-Reviewer collaboration",
        max_iterations=max_iterations,
        sub_agents=[coder, reviewer, approval_checker],
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
        role="user", parts=[types.Part(text="Start code review process")]
    )

    iterations = []
    num_iterations = 0

    async for event in runner.run_async(
        user_id="study", session_id=session.id, new_message=initial_message
    ):
        # Track iterations by monitoring agent responses
        if hasattr(event, "author"):
            if event.author == "Coder" and hasattr(event, "content"):
                num_iterations += 1

    # Extract final results from session state
    final_code = session.state.get("current_code", "")
    final_decision_text = session.state.get("review_decision", "")

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


def _generate_budget_message(
    awareness_condition: MultiAgentAwarenessCondition,
    max_iterations: int,
) -> str:
    """Generate budget awareness message for agents.

    Args:
        awareness_condition: Budget awareness level
        max_iterations: Maximum iterations

    Returns:
        str: Budget message (empty for NO_AWARENESS)
    """
    if awareness_condition == MultiAgentAwarenessCondition.NO_AWARENESS:
        return ""

    # For now, simple awareness message
    # TODO: Implement full budget tracking with limits
    return f"""[BUDGET AWARENESS]
You are working in a team with a limited token budget.
Maximum {max_iterations} iterations available.
Use tokens wisely - be concise and focused.
"""
