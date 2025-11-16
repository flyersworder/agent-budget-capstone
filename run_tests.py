#!/usr/bin/env python
"""Run all tests for the agent budget capstone project.

Usage:
    python run_tests.py
    uv run python run_tests.py
"""

import subprocess
import sys


def run_test(test_module: str) -> bool:
    """Run a single test module.

    Args:
        test_module: Module name to test (e.g., 'tests.test_core')

    Returns:
        True if test passed, False otherwise
    """
    print(f"\n{'=' * 60}")
    print(f"Running {test_module}...")
    print("=" * 60)

    result = subprocess.run([sys.executable, "-m", test_module], capture_output=False)

    return result.returncode == 0


def main() -> int:
    """Run all tests.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    tests = ["tests.test_core", "tests.test_experiments"]

    all_passed = True

    for test in tests:
        if not run_test(test):
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
