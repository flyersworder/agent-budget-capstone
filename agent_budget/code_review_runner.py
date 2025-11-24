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

import ast
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from agent_budget.agent_factory import AgentFactory
from agent_budget.core import MultiAgentAwarenessCondition, get_coder_budget


class FailureReason(Enum):
    """Categorizes why a trial failed."""

    NONE = "none"  # Trial succeeded
    TRUNCATION = "truncation"  # Code truncated due to token limit
    WRONG_ANSWER = "wrong_answer"  # Code ran but produced wrong output
    RUNTIME_ERROR = "runtime_error"  # Code crashed during execution
    TIMEOUT = "timeout"  # Code execution timed out
    SYNTAX_ERROR = "syntax_error"  # Code has syntax errors (may indicate truncation)


@dataclass
class TruncationInfo:
    """Detailed truncation detection results."""

    is_truncated: bool
    tokens_at_limit: bool  # tokens_used >= budget - margin
    syntax_valid: bool  # ast.parse() succeeded
    has_syntax_error: bool  # Execution failed with SyntaxError
    thinking_tokens: int
    output_tokens: int
    total_tokens: int
    budget_limit: int

    @property
    def confidence(self) -> str:
        """How confident we are about truncation detection."""
        if self.tokens_at_limit and not self.syntax_valid:
            return "high"  # Hit limit AND invalid syntax
        elif self.tokens_at_limit and self.has_syntax_error:
            return "high"  # Hit limit AND syntax error at runtime
        elif self.tokens_at_limit:
            return "medium"  # Hit limit but syntax ok (might be valid short code)
        elif not self.syntax_valid:
            return "medium"  # Invalid syntax but didn't hit limit (bug vs truncation?)
        else:
            return "low"  # Neither indicator present


@dataclass
class IterationDetail:
    """Details for a single iteration."""

    iteration: int
    coder_thinking_tokens: int
    coder_output_tokens: int
    coder_total_tokens: int
    reviewer_tokens: int
    truncation_info: TruncationInfo | None
    test_result: str  # PASS, FAIL, or error message
    failure_reason: FailureReason


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

    # Token usage (tracked from LLM responses)
    team_total_tokens: int
    coder_tokens: int = 0
    reviewer_tokens: int = 0

    # Failure analysis
    failure_reason: FailureReason = FailureReason.NONE
    any_truncation: bool = False  # True if ANY iteration had truncation

    # Iteration details (now with truncation info)
    iteration_details: list[IterationDetail] = field(default_factory=list)

    # Legacy field for compatibility
    iterations: list[dict[str, Any]] | None = None

    # Metadata
    test_passed: bool = False


def check_syntax_valid(code: str) -> bool:
    """Check if Python code has valid syntax using ast.parse().

    Args:
        code: Python code to check

    Returns:
        True if syntax is valid, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def detect_truncation(
    code: str,
    thinking_tokens: int,
    output_tokens: int,
    budget_limit: int,
    execution_stderr: str = "",
    margin: int = 10,
) -> TruncationInfo:
    """Detect if code was truncated due to token limits.

    Uses multiple signals for reliable detection:
    1. Token usage at/near limit
    2. Syntax validation with ast.parse()
    3. SyntaxError in execution stderr

    Args:
        code: The generated code
        thinking_tokens: Tokens used for thinking
        output_tokens: Tokens used for output
        budget_limit: Maximum allowed tokens (max_output_tokens)
        execution_stderr: Stderr from code execution (if available)
        margin: Token margin for "at limit" detection

    Returns:
        TruncationInfo with detection results
    """
    total_tokens = thinking_tokens + output_tokens
    tokens_at_limit = total_tokens >= (budget_limit - margin)
    syntax_valid = check_syntax_valid(code)
    has_syntax_error = (
        "SyntaxError" in execution_stderr or "IndentationError" in execution_stderr
    )

    # Truncation = hit token limit AND (invalid syntax OR syntax error at runtime)
    is_truncated = tokens_at_limit and (not syntax_valid or has_syntax_error)

    return TruncationInfo(
        is_truncated=is_truncated,
        tokens_at_limit=tokens_at_limit,
        syntax_valid=syntax_valid,
        has_syntax_error=has_syntax_error,
        thinking_tokens=thinking_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        budget_limit=budget_limit,
    )


def classify_failure(
    test_result: str,
    truncation_info: TruncationInfo | None,
) -> FailureReason:
    """Classify the reason for test failure.

    Args:
        test_result: Result string from test_code()
        truncation_info: Truncation detection results

    Returns:
        FailureReason classification
    """
    if "PASS" in test_result:
        return FailureReason.NONE

    # Check truncation first (highest priority)
    if truncation_info and truncation_info.is_truncated:
        return FailureReason.TRUNCATION

    # Check for specific error types
    if "timed out" in test_result.lower():
        return FailureReason.TIMEOUT

    if "SyntaxError" in test_result or "IndentationError" in test_result:
        # Syntax error but NOT at token limit = likely a bug, not truncation
        if truncation_info and truncation_info.tokens_at_limit:
            return FailureReason.TRUNCATION
        return FailureReason.SYNTAX_ERROR

    if "crashed" in test_result.lower() or "exit code" in test_result.lower():
        return FailureReason.RUNTIME_ERROR

    if "mismatch" in test_result.lower() or "Expected:" in test_result:
        return FailureReason.WRONG_ANSWER

    # Default to wrong answer for other failures
    return FailureReason.WRONG_ANSWER


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
    difficulty: str = "medium",
) -> CodeReviewTrial:
    """Run a single code review trial using LoopAgent pattern.

    Args:
        problem: LiveCodeBench problem dict
        awareness_condition: Budget awareness level
        max_iterations: Maximum iterations allowed (default: 3)
        difficulty: Problem difficulty ("easy" or "medium") for budget scaling

    Returns:
        CodeReviewTrial with full results
    """
    # Parse test case
    import json

    public_tests = json.loads(problem["public_test_cases"])
    test_input = public_tests[0]["input"]
    expected_output = public_tests[0]["output"]

    # Storage for tracking iteration data (captured by closure)
    iteration_data: dict[str, Any] = {
        "test_results": [],  # List of (test_result, stderr) tuples
        "codes": [],  # List of code strings per iteration
    }

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
            result = "FAIL: No tool context available"
            iteration_data["test_results"].append((result, ""))
            return result

        code = tool_context.session.state.get("current_code", "")

        if not code:
            result = "FAIL: No code found in session state"
            iteration_data["test_results"].append((result, ""))
            return result

        # Strip markdown code block markers if present
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python") :].lstrip()
        if code.startswith("```"):
            code = code[3:].lstrip()
        if code.endswith("```"):
            code = code[:-3].rstrip()

        # Store code for truncation analysis
        iteration_data["codes"].append(code)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        stderr_output = ""
        try:
            proc = subprocess.run(
                ["python", temp_file],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=5,
            )
            actual = proc.stdout.strip()
            expected = expected_output.strip()
            stderr_output = proc.stderr

            if proc.returncode != 0:
                test_result = f"FAIL: Code crashed with exit code {proc.returncode}\nSTDERR: {proc.stderr}"
            elif actual == expected:
                test_result = "PASS: All tests passed!"
            else:
                test_result = f"FAIL: Output mismatch\nExpected: {expected[:200]}\nActual: {actual[:200]}"
        except subprocess.TimeoutExpired:
            test_result = "FAIL: Code execution timed out after 5 seconds"
        except Exception as e:
            test_result = f"FAIL: {str(e)}"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        # Store test result for later analysis
        iteration_data["test_results"].append((test_result, stderr_output))
        return test_result

    # Create test_code tool for this specific problem
    test_code_tool = FunctionTool(func=test_code)

    # Use factory to create code review team with difficulty-based budgets
    factory = AgentFactory(model="gemini-2.5-flash-lite")
    review_loop = factory.create_code_review_team(
        problem_description=problem["question_content"],
        test_code_tool=test_code_tool,
        awareness_condition=awareness_condition,
        max_iterations=max_iterations,
        difficulty=difficulty,
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

    num_iterations = 0
    coder_tokens = 0
    reviewer_tokens = 0

    # Per-iteration token tracking
    coder_token_details: list[
        tuple[int, int]
    ] = []  # List of (thinking, output) per iteration
    reviewer_token_counts: list[int] = []

    async for event in runner.run_async(
        user_id="study", session_id=session.id, new_message=initial_message
    ):
        # Track iterations and token usage
        if hasattr(event, "author") and event.author:
            # Track token usage from LLM responses
            if hasattr(event, "usage_metadata") and event.usage_metadata:
                thinking = getattr(event.usage_metadata, "thoughts_token_count", 0) or 0
                output = getattr(event.usage_metadata, "candidates_token_count", 0) or 0
                total = thinking + output

                if event.author == "Coder":
                    coder_tokens += total
                    coder_token_details.append((thinking, output))
                elif event.author == "Reviewer":
                    reviewer_tokens += total
                    reviewer_token_counts.append(total)

            # Count iterations (each Coder response = 1 iteration)
            if event.author == "Coder":
                num_iterations += 1

    # Fetch final session state
    updated_session = await session_service.get_session(
        app_name="code_review_study",
        user_id="study",
        session_id=session.id,
    )

    # Extract results
    final_decision_text = updated_session.state.get("review_decision", "")

    # Determine success
    approved = "APPROVE" in final_decision_text.upper()
    if approved:
        final_decision = "APPROVE"
        success = True
    else:
        final_decision = "MAX_ITERATIONS_REACHED"
        success = False

    # Use tracked token counts
    team_total_tokens = coder_tokens + reviewer_tokens

    # Build detailed iteration analysis with truncation detection
    iteration_details: list[IterationDetail] = []
    any_truncation = False
    budget_limit = get_coder_budget(difficulty).total

    for i in range(num_iterations):
        # Get token details for this iteration
        thinking, output = (
            coder_token_details[i] if i < len(coder_token_details) else (0, 0)
        )
        reviewer_tok = reviewer_token_counts[i] if i < len(reviewer_token_counts) else 0

        # Get code and test result for this iteration
        code = iteration_data["codes"][i] if i < len(iteration_data["codes"]) else ""
        test_result, stderr = (
            iteration_data["test_results"][i]
            if i < len(iteration_data["test_results"])
            else ("FAIL: No test result", "")
        )

        # Detect truncation
        truncation_info = detect_truncation(
            code=code,
            thinking_tokens=thinking,
            output_tokens=output,
            budget_limit=budget_limit,
            execution_stderr=stderr,
        )

        if truncation_info.is_truncated:
            any_truncation = True

        # Classify failure reason
        failure_reason = classify_failure(test_result, truncation_info)

        iteration_details.append(
            IterationDetail(
                iteration=i + 1,
                coder_thinking_tokens=thinking,
                coder_output_tokens=output,
                coder_total_tokens=thinking + output,
                reviewer_tokens=reviewer_tok,
                truncation_info=truncation_info,
                test_result=test_result[:200],  # Truncate for storage
                failure_reason=failure_reason,
            )
        )

    # Determine overall failure reason (from last iteration if failed)
    overall_failure_reason = FailureReason.NONE
    if not success and iteration_details:
        overall_failure_reason = iteration_details[-1].failure_reason

    # Build legacy iteration details for compatibility
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
        coder_tokens=coder_tokens,
        reviewer_tokens=reviewer_tokens,
        failure_reason=overall_failure_reason,
        any_truncation=any_truncation,
        iteration_details=iteration_details,
        iterations=iterations,
        test_passed=success,
    )
