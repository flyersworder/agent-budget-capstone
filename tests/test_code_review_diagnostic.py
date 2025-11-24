"""Diagnostic test for code review loop with full traceability.

This test runs a few HARD problems with detailed logging to verify:
1. Coder is producing complete code (not truncated)
2. Reviewer is calling test_code() function
3. Test results are being properly interpreted
4. Feedback loop is working correctly

Run with: python -m tests.test_code_review_diagnostic
"""

import asyncio
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

# Load .env manually
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.adk.tools import FunctionTool  # noqa: E402
from google.genai import types  # noqa: E402

from agent_budget.agent_factory import AgentFactory  # noqa: E402
from agent_budget.core import MultiAgentAwarenessCondition  # noqa: E402


# Global counters for diagnostics
diagnostic_data: dict = {
    "test_code_calls": 0,
    "coder_outputs": [],
    "reviewer_outputs": [],
    "test_results": [],
}


def create_diagnostic_test_tool(test_input: str, expected_output: str) -> FunctionTool:
    """Create a test_code tool with full diagnostic logging."""

    def test_code(tool_context=None) -> str:
        """Test the code with diagnostic output."""
        diagnostic_data["test_code_calls"] += 1
        call_num = diagnostic_data["test_code_calls"]

        print(f"\n{'=' * 60}")
        print(f">>> test_code() CALLED (call #{call_num})")
        print("=" * 60)

        if not tool_context or not hasattr(tool_context, "session"):
            result = "FAIL: No tool context available"
            diagnostic_data["test_results"].append(
                {"call": call_num, "result": result, "error": "no_context"}
            )
            print(f"Result: {result}")
            return result

        code = tool_context.session.state.get("current_code", "")
        if not code:
            result = "FAIL: No code found in session state"
            diagnostic_data["test_results"].append(
                {"call": call_num, "result": result, "error": "no_code"}
            )
            print(f"Result: {result}")
            return result

        # Strip markdown
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python") :].lstrip()
        if code.startswith("```"):
            code = code[3:].lstrip()
        if code.endswith("```"):
            code = code[:-3].rstrip()

        # Log code details
        code_lines = code.split("\n")
        print(f"Code length: {len(code)} chars, {len(code_lines)} lines")
        print("First 10 lines:")
        for i, line in enumerate(code_lines[:10]):
            print(f"  {i + 1:3d}: {line[:80]}")
        if len(code_lines) > 10:
            print(f"  ... ({len(code_lines) - 10} more lines)")
        print("Last 5 lines:")
        for i, line in enumerate(code_lines[-5:]):
            print(f"  {len(code_lines) - 4 + i:3d}: {line[:80]}")

        # Check for obvious truncation signs
        truncation_signs = []
        if code.rstrip().endswith("..."):
            truncation_signs.append("ends with '...'")
        if "def " in code and code.count("def ") > code.count("return "):
            truncation_signs.append("more 'def' than 'return' statements")
        if code.count("(") != code.count(")"):
            truncation_signs.append(
                f"unbalanced parens: {code.count('(')} ( vs {code.count(')')} )"
            )
        if code.count("{") != code.count("}"):
            truncation_signs.append(
                f"unbalanced braces: {code.count('{')} {{ vs {code.count('}')} }}"
            )

        if truncation_signs:
            print("\n⚠️  POTENTIAL TRUNCATION SIGNS:")
            for sign in truncation_signs:
                print(f"    - {sign}")

        # Execute the code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result_obj = subprocess.run(
                ["python", temp_file],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=5,
            )
            actual = result_obj.stdout.strip()
            expected = expected_output.strip()

            print("\nExecution result:")
            print(f"  Exit code: {result_obj.returncode}")
            print(
                f"  Expected output: {expected[:100]}{'...' if len(expected) > 100 else ''}"
            )
            print(
                f"  Actual output:   {actual[:100]}{'...' if len(actual) > 100 else ''}"
            )
            if result_obj.stderr:
                print(f"  Stderr: {result_obj.stderr[:200]}")

            if result_obj.returncode != 0:
                result = f"FAIL: Code crashed with exit code {result_obj.returncode}\nSTDERR: {result_obj.stderr[:500]}"
            elif actual == expected:
                result = "PASS: All tests passed!"
            else:
                result = f"FAIL: Output mismatch\nExpected: {expected[:200]}\nActual: {actual[:200]}"

            diagnostic_data["test_results"].append(
                {
                    "call": call_num,
                    "result": "PASS" if actual == expected else "FAIL",
                    "exit_code": result_obj.returncode,
                    "expected": expected[:100],
                    "actual": actual[:100],
                    "code_lines": len(code_lines),
                    "code_chars": len(code),
                    "truncation_signs": truncation_signs,
                }
            )

        except subprocess.TimeoutExpired:
            result = "FAIL: Code execution timed out after 5 seconds"
            diagnostic_data["test_results"].append(
                {"call": call_num, "result": result, "error": "timeout"}
            )
        except Exception as e:
            result = f"FAIL: {str(e)}"
            diagnostic_data["test_results"].append(
                {"call": call_num, "result": result, "error": str(e)}
            )
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        print(f"\nFinal result: {result[:100]}")
        print("=" * 60)
        return result

    return FunctionTool(func=test_code)


async def run_diagnostic_trial(
    problem: dict,
    awareness_condition: MultiAgentAwarenessCondition,
    max_iterations: int = 3,
) -> dict:
    """Run a single trial with full diagnostic output."""

    # Reset diagnostics
    diagnostic_data["test_code_calls"] = 0
    diagnostic_data["coder_outputs"] = []
    diagnostic_data["reviewer_outputs"] = []
    diagnostic_data["test_results"] = []

    # Parse test case
    public_tests = json.loads(problem["public_test_cases"])
    test_input = public_tests[0]["input"]
    expected_output = public_tests[0]["output"]

    print("\nTest case:")
    print(f"  Input: {test_input[:100]}{'...' if len(test_input) > 100 else ''}")
    print(
        f"  Expected: {expected_output[:100]}{'...' if len(expected_output) > 100 else ''}"
    )

    # Create diagnostic test tool
    test_code_tool = create_diagnostic_test_tool(test_input, expected_output)

    # Create team
    factory = AgentFactory(model="gemini-2.5-flash-lite")
    review_loop = factory.create_code_review_team(
        problem_description=problem["question_content"],
        test_code_tool=test_code_tool,
        awareness_condition=awareness_condition,
        max_iterations=max_iterations,
    )

    # Run
    session_service = InMemorySessionService()
    runner = Runner(
        agent=review_loop, app_name="diagnostic_test", session_service=session_service
    )

    session = await session_service.create_session(
        app_name="diagnostic_test",
        user_id="test",
    )
    session.state["review_decision"] = ""

    initial_message = types.Content(
        role="user",
        parts=[
            types.Part(
                text="Please write the Python code to solve the problem. Return ONLY code, no explanations."
            )
        ],
    )

    print("\n" + "-" * 60)
    print("Running code review loop...")
    print("-" * 60)

    iteration = 0
    coder_tokens = 0
    reviewer_tokens = 0

    async for event in runner.run_async(
        user_id="test", session_id=session.id, new_message=initial_message
    ):
        if hasattr(event, "author") and event.author:
            # Track tokens
            if hasattr(event, "usage_metadata") and event.usage_metadata:
                thinking = getattr(event.usage_metadata, "thoughts_token_count", 0) or 0
                output = getattr(event.usage_metadata, "candidates_token_count", 0) or 0
                total = thinking + output

                if event.author == "Coder":
                    coder_tokens += total
                    iteration += 1
                    print(
                        f"\n[Coder - Iteration {iteration}] tokens: {total} (thinking: {thinking}, output: {output})"
                    )
                elif event.author == "Reviewer":
                    reviewer_tokens += total
                    print(f"\n[Reviewer - Iteration {iteration}] tokens: {total}")

            # Log content
            if hasattr(event, "content") and event.content:
                text = ""
                if hasattr(event.content, "parts") and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            text = part.text

                if event.author == "Coder":
                    diagnostic_data["coder_outputs"].append(
                        {
                            "iteration": iteration,
                            "length": len(text),
                            "preview": text[:200],
                        }
                    )
                    print(f"  Output length: {len(text)} chars")
                    if len(text) < 50:
                        print(f"  ⚠️  Very short output: '{text}'")
                elif event.author == "Reviewer":
                    diagnostic_data["reviewer_outputs"].append(
                        {"iteration": iteration, "text": text[:500]}
                    )
                    print(f"  Decision: {text[:200]}...")

    # Get final state
    updated_session = await session_service.get_session(
        app_name="diagnostic_test",
        user_id="test",
        session_id=session.id,
    )

    final_decision = updated_session.state.get("review_decision", "")
    success = "APPROVE" in final_decision.upper()

    return {
        "success": success,
        "iterations": iteration,
        "coder_tokens": coder_tokens,
        "reviewer_tokens": reviewer_tokens,
        "test_code_calls": diagnostic_data["test_code_calls"],
        "test_results": diagnostic_data["test_results"],
        "final_decision": final_decision[:200],
    }


async def main():
    """Run diagnostic tests on hard problems."""

    print("=" * 80)
    print("DIAGNOSTIC TEST: Code Review Loop on HARD Problems")
    print("=" * 80)

    # Load dataset
    print("\nLoading LiveCodeBench dataset...")
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="release_v6",
        trust_remote_code=True,
    )

    # Get problems by difficulty after cutoff
    problems_by_diff: dict[str, list] = {"easy": [], "medium": [], "hard": []}
    for p in dataset:
        contest_date = datetime.fromisoformat(p["contest_date"])
        if contest_date >= datetime(2025, 2, 1):
            diff = p.get("difficulty", "unknown")
            if diff in problems_by_diff:
                problems_by_diff[diff].append(p)

    for diff, probs in problems_by_diff.items():
        print(f"Found {len(probs)} {diff.upper()} problems after cutoff")

    # Select which difficulty to test (can be changed)
    test_difficulty = os.environ.get("TEST_DIFFICULTY", "medium")
    test_problems = problems_by_diff.get(test_difficulty, [])

    # Test first 2 problems with full diagnostics
    num_to_test = min(2, len(test_problems))
    print(
        f"\nTesting {num_to_test} {test_difficulty.upper()} problems with full traceability...\n"
    )

    results = []

    for i, problem in enumerate(test_problems[:num_to_test]):
        print("\n" + "=" * 80)
        print(f"PROBLEM {i + 1}/{num_to_test}: {problem['question_title']}")
        print(f"Platform: {problem['platform']}")
        print(f"Difficulty: {problem['difficulty']}")
        print("=" * 80)

        # Show problem description (truncated)
        desc = problem["question_content"][:500]
        print(f"\nProblem description (first 500 chars):\n{desc}...")

        result = await run_diagnostic_trial(
            problem=problem,
            awareness_condition=MultiAgentAwarenessCondition.NO_AWARENESS,
            max_iterations=3,
        )

        results.append(
            {
                "problem": problem["question_title"],
                "difficulty": problem["difficulty"],
                **result,
            }
        )

        print("\n" + "-" * 60)
        print("TRIAL SUMMARY:")
        print("-" * 60)
        print(f"  Success: {result['success']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Coder tokens: {result['coder_tokens']}")
        print(f"  Reviewer tokens: {result['reviewer_tokens']}")
        print(f"  test_code() calls: {result['test_code_calls']}")
        print(f"  Final decision: {result['final_decision'][:100]}")

    # Final summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    for r in results:
        print(f"\n{r['problem'][:50]}:")
        print(f"  Success: {r['success']}")
        print(
            f"  Iterations: {r['iterations']}, test_code calls: {r['test_code_calls']}"
        )

        # Check for issues
        issues = []
        if r["test_code_calls"] == 0:
            issues.append("⚠️  Reviewer never called test_code()")
        if r["test_code_calls"] < r["iterations"]:
            issues.append(
                f"⚠️  Fewer test_code calls ({r['test_code_calls']}) than iterations ({r['iterations']})"
            )
        for tr in r["test_results"]:
            if tr.get("truncation_signs"):
                issues.append(f"⚠️  Possible code truncation: {tr['truncation_signs']}")

        if issues:
            print("  Issues found:")
            for issue in issues:
                print(f"    {issue}")
        else:
            print("  ✓ No obvious issues detected")


if __name__ == "__main__":
    asyncio.run(main())
