"""Validate that thinking token extraction works correctly.

This script runs 2 questions × 2 conditions = 4 experiments
to verify that both thinking_text and response are captured properly.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from experiments.run_part2_pilot import Part2Runner
from experiments.tasks.truthful_qa_tasks import get_pilot_sample


async def main() -> None:
    """Run validation test for thinking token extraction."""
    print("=" * 80)
    print("THINKING TOKEN EXTRACTION VALIDATION")
    print("=" * 80)
    print()
    print("Testing that both thinking_text and response are captured correctly")
    print("Running 2 questions × 2 conditions = 4 experiments")
    print()
    print("=" * 80)
    print()

    # Load first 2 questions
    tasks = get_pilot_sample(seed=42)[:2]

    # Run experiments
    runner = Part2Runner()
    results = []

    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/2] Testing: {task.question[:60]}...")

        # Test unaware condition
        print("  → unaware...", end=" ", flush=True)
        from agent_budget.awareness import AwarenessCondition

        result_unaware = await runner.run_single_experiment(
            task, AwarenessCondition.UNAWARE
        )
        results.append(result_unaware)

        thinking_len = len(result_unaware.thinking_text)
        output_len = len(result_unaware.response)
        print(f"thinking: {thinking_len} chars, output: {output_len} chars")

        # Test aware condition
        print("  → aware...", end=" ", flush=True)
        result_aware = await runner.run_single_experiment(
            task, AwarenessCondition.AWARE
        )
        results.append(result_aware)

        thinking_len = len(result_aware.thinking_text)
        output_len = len(result_aware.response)
        print(f"thinking: {thinking_len} chars, output: {output_len} chars")

    print()
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()

    # Check all experiments captured both thinking and output
    all_have_thinking = all(len(r.thinking_text) > 0 for r in results if r.success)
    all_have_output = all(len(r.response) > 0 for r in results if r.success)

    successful = [r for r in results if r.success]

    print(f"Successful experiments: {len(successful)}/{len(results)}")
    print()

    if all_have_thinking:
        print("✅ All experiments captured thinking text")
    else:
        print("❌ Some experiments missing thinking text")

    if all_have_output:
        print("✅ All experiments captured output text")
    else:
        print("❌ Some experiments missing output text")

    print()
    print("=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)
    print()

    # Show one example in detail
    if successful:
        example = successful[0]
        print(f"Question: {example.question[:80]}...")
        print()
        print("Thinking text (first 200 chars):")
        print(f"  {example.thinking_text[:200]}...")
        print()
        print("Output text:")
        print(f"  {example.response}")
        print()
        print(f"Thinking length: {len(example.thinking_text)} chars")
        print(f"Output length: {len(example.response)} chars")
        print(f"Correctness: {example.correctness}")

    print()
    print("=" * 80)

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"thinking_validation_{timestamp}.json"

    with open(json_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "total_experiments": len(results),
                    "successful": len(successful),
                    "all_have_thinking": all_have_thinking,
                    "all_have_output": all_have_output,
                    "timestamp": timestamp,
                },
                "results": [r.to_dict() for r in results],
            },
            f,
            indent=2,
        )

    print(f"Results saved to: {json_path}")
    print()

    if all_have_thinking and all_have_output:
        print("🎉 VALIDATION PASSED - Thinking extraction works!")
    else:
        print("⚠️  VALIDATION FAILED - Check implementation")


if __name__ == "__main__":
    asyncio.run(main())
