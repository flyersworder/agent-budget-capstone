"""Pilot study for Part 2: Budget Awareness Experiment.

This script runs a pilot study testing whether explicit budget awareness
improves agent performance on TruthfulQA questions.

Design:
- 2 conditions: budget-aware vs budget-unaware
- 1 budget level: medium (2048 reasoning / 1024 output)
- 30 TruthfulQA questions
- Objective correctness evaluation

Usage:
    python -m experiments.run_part2_pilot
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

from agent_budget.awareness import (
    BUDGET_MEDIUM,
    AwarenessCondition,
    create_planner_config,
)
from agent_budget.monitor import AgentMetrics, UsageMonitor
from experiments.evaluator_objective import ObjectiveEvaluator
from experiments.tasks.truthful_qa_tasks import TruthfulQATask, get_pilot_sample


@dataclass
class Part2Result:
    """Results from a single Part 2 experiment.

    Attributes:
        task_id: TruthfulQA task identifier
        condition: Awareness condition (aware/unaware)
        question: The question asked
        response: Agent's response
        correct_answer: Ground truth answer
        correctness: Correctness score (0.0-1.0)
        justification: Evaluation justification
        metrics: Token usage and performance metrics
        success: Whether experiment completed
        error: Error message if failed
    """

    task_id: str
    condition: str
    question: str
    response: str
    thinking_text: str  # Thinking tokens (internal reasoning)
    correct_answer: str
    correctness: float
    justification: str
    metrics: AgentMetrics
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "task_id": self.task_id,
            "condition": self.condition,
            "question": self.question,
            "response": self.response,
            "thinking_text": self.thinking_text,
            "correct_answer": self.correct_answer,
            "correctness": self.correctness,
            "justification": self.justification,
            "success": self.success,
            "error": self.error,
            **self.metrics.to_dict(),
        }


@dataclass
class Part2Suite:
    """Collection of Part 2 experiment results.

    Attributes:
        results: List of individual results
        started_at: Suite start timestamp
        completed_at: Suite completion timestamp
    """

    results: list[Part2Result] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None

    def get_by_condition(self, condition: str) -> list[Part2Result]:
        """Get results for a specific condition."""
        return [r for r in self.results if r.condition == condition]

    def get_successful(self) -> list[Part2Result]:
        """Get only successful results."""
        return [r for r in self.results if r.success]

    def get_accuracy(self, condition: str | None = None) -> float:
        """Calculate accuracy for a condition or overall.

        Args:
            condition: If specified, calculate for this condition only

        Returns:
            Mean correctness score (0.0-1.0)
        """
        results = self.get_successful()
        if condition:
            results = [r for r in results if r.condition == condition]

        if not results:
            return 0.0

        return sum(r.correctness for r in results) / len(results)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "metadata": {
                "total_experiments": len(self.results),
                "successful": len(self.get_successful()),
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            },
            "results": [r.to_dict() for r in self.results],
        }


class Part2Runner:
    """Runner for Part 2 budget awareness experiments."""

    def __init__(self) -> None:
        """Initialize runner."""
        self.monitor = UsageMonitor()
        self.evaluator = ObjectiveEvaluator()
        self.session_service = InMemorySessionService()

    async def run_single_experiment(
        self,
        task: TruthfulQATask,
        condition: AwarenessCondition,
    ) -> Part2Result:
        """Run a single experiment.

        Args:
            task: TruthfulQA task to run
            condition: Awareness condition to test

        Returns:
            Part2Result with response and evaluation
        """
        try:
            # Create agent config with awareness condition
            instruction, planner, generate_config = create_planner_config(
                condition=condition, budget_config=BUDGET_MEDIUM
            )

            # Create agent
            agent = Agent(
                model="gemini-2.5-flash-lite",
                name=f"{condition.value}_agent",
                instruction=instruction,
                planner=planner,
                generate_content_config=generate_config,
                tools=[google_search],
            )

            # Create runner
            runner = Runner(
                agent=agent,
                app_name="part2_pilot",
                session_service=self.session_service,
            )
            session = await runner.session_service.create_session(
                app_name="part2_pilot", user_id="researcher"
            )

            # Run task
            start_time = time.time()
            events = []

            message = types.Content(role="user", parts=[types.Part(text=task.question)])

            async for event in runner.run_async(
                user_id="researcher", session_id=session.id, new_message=message
            ):
                events.append(event)

            duration = time.time() - start_time

            # Extract both thinking tokens and output tokens separately
            thinking_text = ""
            output_parts = []  # Collect all output parts

            # Iterate through all events and separate thinking from output
            for event in events:
                if hasattr(event, "content") and event.content:
                    if hasattr(event.content, "parts") and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                if hasattr(part, "thought") and part.thought:
                                    # This is thinking/reasoning text
                                    thinking_text += part.text
                                else:
                                    # This is actual output text
                                    output_parts.append(part.text)

            # Join all output parts (usually there's only one, but be safe)
            response = "".join(output_parts) if output_parts else ""

            # Extract metrics
            metrics = self.monitor.extract_metrics_from_events(
                events=events, strategy=condition.value, duration=duration
            )

            # Evaluate correctness
            eval_result = self.evaluator.evaluate_correctness(task, response)

            return Part2Result(
                task_id=task.id,
                condition=condition.value,
                question=task.question,
                response=response,
                thinking_text=thinking_text,
                correct_answer=task.best_answer,
                correctness=eval_result.score,
                justification=eval_result.justification,
                metrics=metrics,
                success=True,
            )

        except Exception as e:
            return Part2Result(
                task_id=task.id,
                condition=condition.value,
                question=task.question,
                response="",
                thinking_text="",
                correct_answer=task.best_answer,
                correctness=0.0,
                justification="",
                metrics=AgentMetrics(
                    strategy=condition.value,
                    reasoning_tokens_used=0,
                    output_tokens_used=0,
                    total_tokens_used=0,
                    duration_seconds=0.0,
                ),
                success=False,
                error=str(e),
            )

    async def run_pilot(self) -> Part2Suite:
        """Run pilot study with 30 questions × 2 conditions.

        Returns:
            Part2Suite with all results
        """
        suite = Part2Suite()
        suite.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Load pilot tasks
        tasks = get_pilot_sample(seed=42)

        # Run all conditions
        conditions = [AwarenessCondition.UNAWARE, AwarenessCondition.AWARE]

        for task in tasks:
            for condition in conditions:
                print(
                    f"Running {task.id} with {condition.value} condition "
                    f"(budget: {BUDGET_MEDIUM.total} tokens)..."
                )
                result = await self.run_single_experiment(task, condition)
                suite.results.append(result)

                if result.success:
                    print(f"  ✓ Completed in {result.metrics.duration_seconds:.2f}s")
                    print(
                        f"    Tokens: {result.metrics.total_tokens_used}/{BUDGET_MEDIUM.total}"
                    )
                    print(f"    Correctness: {result.correctness:.2f}")
                else:
                    print(f"  ✗ Failed: {result.error}")

        suite.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return suite


async def main() -> None:
    """Run pilot study."""
    print("=" * 80)
    print("PART 2 PILOT: Budget Awareness Study")
    print("=" * 80)
    print()
    print("Research Question:")
    print("  Does explicit budget awareness improve agent performance?")
    print()
    print("Design:")
    print("  - Conditions: Budget-Aware vs Budget-Unaware")
    print(
        f"  - Budget: {BUDGET_MEDIUM.reasoning_tokens} reasoning / {BUDGET_MEDIUM.output_tokens} output"
    )
    print("  - Tasks: 30 TruthfulQA questions")
    print("  - Evaluation: Objective correctness (0.0-1.0)")
    print()
    print("=" * 80)
    print()

    # Run pilot
    runner = Part2Runner()
    suite = await runner.run_pilot()

    # Print summary
    print()
    print("=" * 80)
    print("PILOT RESULTS SUMMARY")
    print("=" * 80)
    print()

    successful = suite.get_successful()
    print(f"Total experiments: {len(suite.results)}")
    print(f"Successful: {len(successful)}")
    print()

    # Accuracy by condition
    for condition in [AwarenessCondition.UNAWARE, AwarenessCondition.AWARE]:
        cond_results = [r for r in successful if r.condition == condition.value]
        if cond_results:
            accuracy = sum(r.correctness for r in cond_results) / len(cond_results)
            avg_tokens = sum(r.metrics.total_tokens_used for r in cond_results) / len(
                cond_results
            )
            print(f"{condition.value.upper()}:")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Avg tokens: {avg_tokens:.0f}/{BUDGET_MEDIUM.total}")
            print()

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"part2_pilot_{timestamp}.json"

    with open(json_path, "w") as f:
        json.dump(suite.to_dict(), f, indent=2)

    print(f"✅ Results saved to: {json_path}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
