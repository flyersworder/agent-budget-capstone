"""Part 1: Budget Awareness Experiment (Full Study).

Tests whether explicit budget awareness improves agent performance
across different budget constraints.

Design (Between-Subjects):
- 3 budget levels: tight (640), moderate (1280), comfortable (2560)
- 2 awareness conditions: budget-aware vs budget-unaware
- 6 total conditions × ~17 questions = 100 TruthfulQA questions
- Stratified random assignment to balance difficulty
- Objective correctness evaluation

Usage:
    python -m experiments.run_part1_full
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
    BUDGET_COMFORTABLE,
    BUDGET_MODERATE,
    BUDGET_TIGHT,
    AwarenessCondition,
    create_planner_config,
)
from agent_budget.core import TokenBudget
from agent_budget.monitor import AgentMetrics, UsageMonitor
from experiments.shared.evaluator_truthfulqa import ObjectiveEvaluator
from experiments.tasks.truthful_qa_tasks import TruthfulQATask


@dataclass
class Part1Result:
    """Results from a single Part 1 experiment.

    Attributes:
        task_id: TruthfulQA task identifier
        condition: Awareness condition (aware/unaware)
        budget_level: Budget level (tight/moderate/comfortable)
        category: Question category from TruthfulQA (e.g., Health, Law, Misconceptions)
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
    budget_level: str  # tight, moderate, or comfortable
    category: str  # TruthfulQA category (e.g., Health, Law, Misconceptions)
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
            "budget_level": self.budget_level,
            "category": self.category,
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
class Part1Suite:
    """Collection of Part 1 experiment results.

    Attributes:
        results: List of individual results
        started_at: Suite start timestamp
        completed_at: Suite completion timestamp
    """

    results: list[Part1Result] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None

    def get_by_condition(self, condition: str) -> list[Part1Result]:
        """Get results for a specific condition."""
        return [r for r in self.results if r.condition == condition]

    def get_successful(self) -> list[Part1Result]:
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


class Part1Runner:
    """Runner for Part 1 budget awareness experiments."""

    def __init__(self) -> None:
        """Initialize runner."""
        self.monitor = UsageMonitor()
        self.evaluator = ObjectiveEvaluator()
        self.session_service = InMemorySessionService()

    async def run_single_experiment(
        self,
        task: TruthfulQATask,
        condition: AwarenessCondition,
        budget_config: TokenBudget,
        budget_level: str,
    ) -> Part1Result:
        """Run a single experiment.

        Args:
            task: TruthfulQA task to run
            condition: Awareness condition to test
            budget_config: Token budget configuration
            budget_level: Budget level name (tight/moderate/comfortable)

        Returns:
            Part1Result with response and evaluation
        """
        try:
            # Create agent config with awareness condition
            instruction, planner, generate_config = create_planner_config(
                condition=condition, budget_config=budget_config
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
                app_name="part1_pilot",
                session_service=self.session_service,
            )
            session = await runner.session_service.create_session(
                app_name="part1_pilot", user_id="researcher"
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

            return Part1Result(
                task_id=task.id,
                condition=condition.value,
                budget_level=budget_level,
                category=task.category,
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
            return Part1Result(
                task_id=task.id,
                condition=condition.value,
                budget_level=budget_level,
                category=task.category,
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

    async def run_full_study(
        self, n_questions: int = 100, seed: int = 42
    ) -> Part1Suite:
        """Run full Part 1 study with between-subjects design.

        Args:
            n_questions: Number of questions to test (default 100)
            seed: Random seed for assignment

        Returns:
            Part1Suite with all results
        """
        import random

        random.seed(seed)

        suite = Part1Suite()
        suite.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Load questions
        from experiments.tasks.truthful_qa_tasks import TruthfulQALoader

        loader = TruthfulQALoader()
        tasks = loader.sample_stratified(n=n_questions, seed=seed)

        # Define all 6 conditions (3 budgets × 2 awareness)
        budget_configs = [
            ("tight", BUDGET_TIGHT),
            ("moderate", BUDGET_MODERATE),
            ("comfortable", BUDGET_COMFORTABLE),
        ]
        awareness_conditions = [AwarenessCondition.UNAWARE, AwarenessCondition.AWARE]

        # Create all condition combinations
        conditions = [
            (budget_level, budget_config, awareness)
            for budget_level, budget_config in budget_configs
            for awareness in awareness_conditions
        ]

        # Randomly assign questions to conditions (between-subjects)
        random.shuffle(tasks)
        questions_per_condition = len(tasks) // len(conditions)
        remainder = len(tasks) % len(conditions)

        # Distribute remainder evenly across first few conditions
        assignments = []
        task_idx = 0
        for i, (budget_level, budget_config, awareness) in enumerate(conditions):
            # Give one extra question to first 'remainder' conditions
            n_questions = questions_per_condition + (1 if i < remainder else 0)
            assigned_tasks = tasks[task_idx : task_idx + n_questions]
            task_idx += n_questions

            for task in assigned_tasks:
                assignments.append((task, budget_level, budget_config, awareness))

        # Run experiments
        for i, (task, budget_level, budget_config, awareness) in enumerate(
            assignments, 1
        ):
            print(
                f"[{i}/{len(assignments)}] {task.id} | {awareness.value} | {budget_level} "
                f"({budget_config.total} tokens)"
            )

            result = await self.run_single_experiment(
                task=task,
                condition=awareness,
                budget_config=budget_config,
                budget_level=budget_level,
            )
            suite.results.append(result)

            if result.success:
                print(
                    f"  ✓ {result.metrics.duration_seconds:.1f}s | "
                    f"{result.metrics.total_tokens_used}/{budget_config.total} tokens | "
                    f"score={result.correctness:.2f}"
                )
            else:
                print(f"  ✗ Failed: {result.error}")

        suite.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return suite


async def main() -> None:
    """Run full Part 1 study."""
    print("=" * 80)
    print("PART 1: Budget Awareness Study (Full)")
    print("=" * 80)
    print()
    print("Design: Between-Subjects")
    print("  - 3 budget levels: tight (640), moderate (1280), comfortable (2560)")
    print("  - 2 awareness conditions: aware vs unaware")
    print("  - 6 conditions × ~17 questions = 100 TruthfulQA questions")
    print("  - Stratified random assignment")
    print()
    print("=" * 80)
    print()

    # Run full study
    runner = Part1Runner()
    suite = await runner.run_full_study(n_questions=100, seed=100)

    # Print summary
    print()
    print("=" * 80)
    print("PART 1 RESULTS SUMMARY")
    print("=" * 80)
    print()

    successful = suite.get_successful()
    print(f"Total experiments: {len(suite.results)}")
    print(f"Successful: {len(successful)}")
    print()

    # Accuracy by condition (budget × awareness)
    budget_levels = ["tight", "moderate", "comfortable"]
    awareness_conditions = ["unaware", "aware"]

    print("Results by Condition:")
    print()
    for budget in budget_levels:
        print(f"{budget.upper()}:")
        for awareness in awareness_conditions:
            cond_results = [
                r
                for r in successful
                if r.budget_level == budget and r.condition == awareness
            ]
            if cond_results:
                accuracy = sum(r.correctness for r in cond_results) / len(cond_results)
                avg_tokens = sum(
                    r.metrics.total_tokens_used for r in cond_results
                ) / len(cond_results)
                print(
                    f"  {awareness:10s}: {accuracy:5.1%} accuracy | "
                    f"{avg_tokens:4.0f} avg tokens | n={len(cond_results)}"
                )
        print()

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"part1_full_{timestamp}.json"

    with open(json_path, "w") as f:
        json.dump(suite.to_dict(), f, indent=2)

    print(f"✅ Results saved to: {json_path}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
