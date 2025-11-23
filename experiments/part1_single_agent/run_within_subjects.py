"""Part 1: Budget Awareness Experiment (Within-Subjects Design).

Tests whether explicit budget awareness improves agent performance
using a within-subjects design where SAME questions are tested in both conditions.

Design:
- Sample 50 questions (stratified by category)
- Budget level: Between-subjects (tight/moderate/comfortable)
- Awareness: Within-subjects (unaware vs aware)
- Each question tested TWICE (once unaware, once aware)
- Total: 50 questions × 2 conditions = 100 data points

Advantages over between-subjects:
- Eliminates question difficulty confounding
- Perfect category balance (same questions in both conditions)
- Higher statistical power (paired analysis)
- Can test category × awareness interactions cleanly

Usage:
    python -m experiments.part1_single_agent.run_within_subjects
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
class Part1WithinSubjectsResult:
    """Results from a single Part 1 within-subjects trial.

    Attributes:
        question_id: Unique identifier for this question (same for both conditions)
        task_id: TruthfulQA task identifier
        condition: Awareness condition (aware/unaware)
        budget_level: Budget level (tight/moderate/comfortable)
        category: Question category from TruthfulQA
        question: The question asked
        response: Agent's response
        thinking_text: Thinking tokens (internal reasoning)
        correct_answer: Ground truth answer
        correctness: Correctness score (0.0-1.0)
        justification: Evaluation justification
        metrics: Token usage and performance metrics
        success: Whether experiment completed
        error: Error message if failed
    """

    question_id: str  # NEW: Links paired observations
    task_id: str
    condition: str
    budget_level: str
    category: str
    question: str
    response: str
    thinking_text: str
    correct_answer: str
    correctness: float
    justification: str
    metrics: AgentMetrics
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "question_id": self.question_id,
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
class Part1WithinSubjectsSuite:
    """Collection of Part 1 within-subjects results.

    Attributes:
        results: List of individual results (2 per question: unaware + aware)
        started_at: Suite start timestamp
        completed_at: Suite completion timestamp
    """

    results: list[Part1WithinSubjectsResult] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None

    def get_pairs(
        self,
    ) -> list[tuple[Part1WithinSubjectsResult, Part1WithinSubjectsResult]]:
        """Get paired results (unaware, aware) for each question.

        Returns:
            List of (unaware_result, aware_result) tuples
        """
        # Group by question_id
        by_question: dict[str, list[Part1WithinSubjectsResult]] = {}
        for r in self.results:
            if r.success:
                if r.question_id not in by_question:
                    by_question[r.question_id] = []
                by_question[r.question_id].append(r)

        # Extract pairs
        pairs = []
        for question_id, results in by_question.items():
            if len(results) == 2:
                unaware = next(r for r in results if r.condition == "unaware")
                aware = next(r for r in results if r.condition == "aware")
                pairs.append((unaware, aware))

        return pairs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "metadata": {
                "design": "within-subjects",
                "total_questions": len(set(r.question_id for r in self.results)),
                "total_observations": len(self.results),
                "successful": len([r for r in self.results if r.success]),
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            },
            "results": [r.to_dict() for r in self.results],
        }


class Part1WithinSubjectsRunner:
    """Runner for Part 1 within-subjects budget awareness experiments."""

    def __init__(self) -> None:
        """Initialize runner."""
        self.monitor = UsageMonitor()
        self.evaluator = ObjectiveEvaluator()
        self.session_service = InMemorySessionService()

    async def run_single_trial(
        self,
        question_id: str,
        task: TruthfulQATask,
        condition: AwarenessCondition,
        budget_config: TokenBudget,
        budget_level: str,
    ) -> Part1WithinSubjectsResult:
        """Run a single trial for one condition.

        Args:
            question_id: Unique identifier for this question
            task: TruthfulQA task to run
            condition: Awareness condition to test
            budget_config: Token budget configuration
            budget_level: Budget level name (tight/moderate/comfortable)

        Returns:
            Part1WithinSubjectsResult with response and evaluation
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
                app_name="part1_within_subjects",
                session_service=self.session_service,
            )
            session = await runner.session_service.create_session(
                app_name="part1_within_subjects", user_id="researcher"
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

            # Extract thinking and output
            thinking_text = ""
            output_parts = []

            for event in events:
                if hasattr(event, "content") and event.content:
                    if hasattr(event.content, "parts") and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                if hasattr(part, "thought") and part.thought:
                                    thinking_text += part.text
                                else:
                                    output_parts.append(part.text)

            response = "".join(output_parts) if output_parts else ""

            # Extract metrics
            metrics = self.monitor.extract_metrics_from_events(
                events=events, strategy=condition.value, duration=duration
            )

            # Evaluate correctness
            eval_result = self.evaluator.evaluate_correctness(task, response)

            return Part1WithinSubjectsResult(
                question_id=question_id,
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
            return Part1WithinSubjectsResult(
                question_id=question_id,
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

    async def run_within_subjects_study(
        self, n_questions: int = 50, seed: int = 42
    ) -> Part1WithinSubjectsSuite:
        """Run full Part 1 study with within-subjects design.

        Args:
            n_questions: Number of questions to test (default 50)
            seed: Random seed for sampling

        Returns:
            Part1WithinSubjectsSuite with all results
        """
        import random

        random.seed(seed)

        suite = Part1WithinSubjectsSuite()
        suite.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Load questions
        from experiments.tasks.truthful_qa_tasks import TruthfulQALoader

        loader = TruthfulQALoader()
        tasks = loader.sample_stratified(n=n_questions, seed=seed)

        # Assign budget levels (between-subjects)
        budget_configs = [
            ("tight", BUDGET_TIGHT),
            ("moderate", BUDGET_MODERATE),
            ("comfortable", BUDGET_COMFORTABLE),
        ]

        # Distribute questions across budget levels
        questions_per_budget = n_questions // len(budget_configs)
        remainder = n_questions % len(budget_configs)

        task_idx = 0
        question_assignments = []

        for i, (budget_level, budget_config) in enumerate(budget_configs):
            n_for_budget = questions_per_budget + (1 if i < remainder else 0)
            assigned_tasks = tasks[task_idx : task_idx + n_for_budget]
            task_idx += n_for_budget

            for task in assigned_tasks:
                question_assignments.append((task, budget_level, budget_config))

        print(f"Total questions: {len(question_assignments)}")
        print(
            f"Total trials (questions × 2 conditions): {len(question_assignments) * 2}"
        )
        print()

        # Run experiments: Each question tested in BOTH conditions
        trial_num = 0
        for question_idx, (task, budget_level, budget_config) in enumerate(
            question_assignments, 1
        ):
            question_id = f"q{question_idx:03d}"

            print(f"\n{'=' * 80}")
            print(
                f"Question {question_idx}/{len(question_assignments)}: {task.id} | "
                f"{budget_level} ({budget_config.total} tokens)"
            )
            print(f"Category: {task.category}")
            print(f"Question: {task.question[:80]}...")
            print(f"{'=' * 80}\n")

            # Test with UNAWARE agent first
            trial_num += 1
            print(f"[Trial {trial_num}/100] UNAWARE agent...")
            result_unaware = await self.run_single_trial(
                question_id=question_id,
                task=task,
                condition=AwarenessCondition.UNAWARE,
                budget_config=budget_config,
                budget_level=budget_level,
            )
            suite.results.append(result_unaware)

            if result_unaware.success:
                print(
                    f"  ✓ {result_unaware.metrics.duration_seconds:.1f}s | "
                    f"{result_unaware.metrics.total_tokens_used}/{budget_config.total} tokens | "
                    f"score={result_unaware.correctness:.2f}"
                )
            else:
                print(f"  ✗ Failed: {result_unaware.error}")

            # Test with AWARE agent
            trial_num += 1
            print(f"[Trial {trial_num}/100] AWARE agent...")
            result_aware = await self.run_single_trial(
                question_id=question_id,
                task=task,
                condition=AwarenessCondition.AWARE,
                budget_config=budget_config,
                budget_level=budget_level,
            )
            suite.results.append(result_aware)

            if result_aware.success:
                print(
                    f"  ✓ {result_aware.metrics.duration_seconds:.1f}s | "
                    f"{result_aware.metrics.total_tokens_used}/{budget_config.total} tokens | "
                    f"score={result_aware.correctness:.2f}"
                )
            else:
                print(f"  ✗ Failed: {result_aware.error}")

            # Show paired comparison
            if result_unaware.success and result_aware.success:
                acc_diff = result_unaware.correctness - result_aware.correctness
                token_diff = (
                    result_aware.metrics.total_tokens_used
                    - result_unaware.metrics.total_tokens_used
                )
                print(
                    f"  → Accuracy diff (unaware - aware): {acc_diff:+.2f} | "
                    f"Token diff: {token_diff:+d}"
                )

        suite.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return suite


async def main() -> None:
    """Run full Part 1 within-subjects study."""
    print("=" * 80)
    print("PART 1: Budget Awareness Study (Within-Subjects)")
    print("=" * 80)
    print()
    print("Design:")
    print("  - 100 questions (stratified by category)")
    print("  - Budget level: Between-subjects (tight/moderate/comfortable)")
    print("  - Awareness: Within-subjects (each question tested in BOTH conditions)")
    print("  - Total: 100 questions × 2 conditions = 200 data points")
    print()
    print("Advantages:")
    print("  ✓ Eliminates question difficulty confounding")
    print("  ✓ Perfect category balance")
    print("  ✓ Higher statistical power (paired analysis)")
    print("  ✓ Can test category × awareness interactions")
    print()
    print("=" * 80)
    print()

    # Run study
    runner = Part1WithinSubjectsRunner()
    suite = await runner.run_within_subjects_study(n_questions=100, seed=500)

    # Print summary
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    pairs = suite.get_pairs()
    print(f"Total questions: {len(pairs)}")
    print(f"Total observations: {len(suite.results)}")
    print(f"Successful: {len([r for r in suite.results if r.success])}")
    print()

    # Paired analysis
    if pairs:
        accuracy_diffs = []
        token_diffs = []

        for unaware, aware in pairs:
            acc_diff = unaware.correctness - aware.correctness
            token_diff = (
                aware.metrics.total_tokens_used - unaware.metrics.total_tokens_used
            )
            accuracy_diffs.append(acc_diff)
            token_diffs.append(token_diff)

        import numpy as np

        print("Paired Differences (Unaware - Aware):")
        print(
            f"  Accuracy: {np.mean(accuracy_diffs):+.3f} ± {np.std(accuracy_diffs):.3f}"
        )
        print(f"  Tokens:   {np.mean(token_diffs):+.1f} ± {np.std(token_diffs):.1f}")
        print()

        # Count wins
        unaware_wins = sum(1 for d in accuracy_diffs if d > 0)
        aware_wins = sum(1 for d in accuracy_diffs if d < 0)
        ties = sum(1 for d in accuracy_diffs if d == 0)

        print("Win/Loss/Tie:")
        print(f"  Unaware wins: {unaware_wins}")
        print(f"  Aware wins:   {aware_wins}")
        print(f"  Ties:         {ties}")
        print()

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"part1_within_subjects_{timestamp}.json"

    with open(json_path, "w") as f:
        json.dump(suite.to_dict(), f, indent=2)

    print(f"✅ Results saved to: {json_path}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
