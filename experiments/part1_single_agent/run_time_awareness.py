"""Part 1: Time Awareness Experiment (Within-Subjects Design).

Tests whether explicit time awareness improves agent performance
using a within-subjects design where SAME questions are tested in both conditions.

Design:
- Sample 75 questions (stratified by category)
- Time level: Between-subjects (tight/moderate/comfortable)
- Awareness: Within-subjects (unaware vs time-aware)
- Each question tested TWICE (once unaware, once time-aware)
- Total: 75 questions × 2 conditions = 150 data points

Key Question:
Does time awareness work better than budget awareness for strategic behavior?

Usage:
    python -m experiments.part1_single_agent.run_time_awareness
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

from agent_budget.awareness import (
    TIME_COMFORTABLE,
    TIME_MODERATE,
    TIME_TIGHT,
    AwarenessCondition,
    TimeConstraint,
    create_time_aware_instruction,
    create_unaware_instruction,
)
from agent_budget.core import TokenBudget
from agent_budget.monitor import AgentMetrics, UsageMonitor
from experiments.shared.evaluator_truthfulqa import ObjectiveEvaluator
from experiments.tasks.truthful_qa_tasks import TruthfulQATask


@dataclass
class TimeAwarenessResult:
    """Results from a single time awareness trial.

    Attributes:
        question_id: Unique identifier for this question (same for both conditions)
        task_id: TruthfulQA task identifier
        condition: Awareness condition (aware/unaware)
        time_level: Time constraint level (tight/moderate/comfortable)
        time_limit_seconds: Time constraint in seconds
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

    question_id: str
    task_id: str
    condition: str
    time_level: str
    time_limit_seconds: int
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
            "time_level": self.time_level,
            "time_limit_seconds": self.time_limit_seconds,
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
class TimeAwarenessSuite:
    """Collection of time awareness results.

    Attributes:
        results: List of individual results (2 per question: unaware + time-aware)
        started_at: Suite start timestamp
        completed_at: Suite completion timestamp
    """

    results: list[TimeAwarenessResult] = None
    started_at: str | None = None
    completed_at: str | None = None

    def __post_init__(self):
        if self.results is None:
            self.results = []

    def get_pairs(
        self,
    ) -> list[tuple[TimeAwarenessResult, TimeAwarenessResult]]:
        """Get paired results (unaware, time-aware) for each question.

        Returns:
            List of (unaware_result, time_aware_result) tuples
        """
        # Group by question_id
        by_question: dict[str, list[TimeAwarenessResult]] = {}
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
                "resource_type": "time",
                "total_questions": len(set(r.question_id for r in self.results)),
                "total_observations": len(self.results),
                "successful": len([r for r in self.results if r.success]),
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            },
            "results": [r.to_dict() for r in self.results],
        }


class TimeAwarenessRunner:
    """Runner for time awareness experiments."""

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
        time_constraint: TimeConstraint,
        time_level: str,
    ) -> TimeAwarenessResult:
        """Run a single trial for one condition.

        Args:
            question_id: Unique identifier for this question
            task: TruthfulQA task to run
            condition: Awareness condition to test
            time_constraint: Time constraint configuration
            time_level: Time level name (tight/moderate/comfortable)

        Returns:
            TimeAwarenessResult with response and evaluation
        """
        try:
            # Create instruction based on condition
            if condition == AwarenessCondition.AWARE:
                instruction = create_time_aware_instruction(time_constraint)
            else:
                instruction = create_unaware_instruction()

            # Create agent with standard budget (no token constraints for time test)
            # Use comfortable budget to avoid token limits confounding time test
            budget = TokenBudget(reasoning_tokens=2048, output_tokens=512)

            planner = BuiltInPlanner(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=budget.reasoning_tokens,
                    include_thoughts=True,
                )
            )

            generate_config = types.GenerateContentConfig(
                max_output_tokens=budget.total,
                temperature=0.2,
            )

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
                app_name="time_awareness",
                session_service=self.session_service,
            )
            session = await runner.session_service.create_session(
                app_name="time_awareness", user_id="researcher"
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

            return TimeAwarenessResult(
                question_id=question_id,
                task_id=task.id,
                condition=condition.value,
                time_level=time_level,
                time_limit_seconds=time_constraint.seconds,
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
            return TimeAwarenessResult(
                question_id=question_id,
                task_id=task.id,
                condition=condition.value,
                time_level=time_level,
                time_limit_seconds=time_constraint.seconds,
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

    async def run_time_awareness_study(
        self, n_questions: int = 75, seed: int = 900
    ) -> TimeAwarenessSuite:
        """Run full time awareness study with within-subjects design.

        Args:
            n_questions: Number of questions to test (default 75)
            seed: Random seed for sampling

        Returns:
            TimeAwarenessSuite with all results
        """
        import random

        random.seed(seed)

        suite = TimeAwarenessSuite()
        suite.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Load questions
        from experiments.tasks.truthful_qa_tasks import TruthfulQALoader

        loader = TruthfulQALoader()
        tasks = loader.sample_stratified(n=n_questions, seed=seed)

        # Assign time levels (between-subjects)
        time_configs = [
            ("tight", TIME_TIGHT),
            ("moderate", TIME_MODERATE),
            ("comfortable", TIME_COMFORTABLE),
        ]

        # Distribute questions across time levels
        questions_per_level = n_questions // len(time_configs)
        remainder = n_questions % len(time_configs)

        task_idx = 0
        question_assignments = []

        for i, (time_level, time_constraint) in enumerate(time_configs):
            n_for_level = questions_per_level + (1 if i < remainder else 0)
            assigned_tasks = tasks[task_idx : task_idx + n_for_level]
            task_idx += n_for_level

            for task in assigned_tasks:
                question_assignments.append((task, time_level, time_constraint))

        print(f"Total questions: {len(question_assignments)}")
        print(
            f"Total trials (questions × 2 conditions): {len(question_assignments) * 2}"
        )
        print()

        # Run experiments: Each question tested in BOTH conditions
        trial_num = 0
        for question_idx, (task, time_level, time_constraint) in enumerate(
            question_assignments, 1
        ):
            question_id = f"q{question_idx:03d}"

            print(f"\n{'=' * 80}")
            print(
                f"Question {question_idx}/{len(question_assignments)}: {task.id} | "
                f"{time_level} ({time_constraint.display})"
            )
            print(f"Category: {task.category}")
            print(f"Question: {task.question[:80]}...")
            print(f"{'=' * 80}\n")

            # Test with UNAWARE agent first
            trial_num += 1
            print(f"[Trial {trial_num}/150] UNAWARE agent...")
            result_unaware = await self.run_single_trial(
                question_id=question_id,
                task=task,
                condition=AwarenessCondition.UNAWARE,
                time_constraint=time_constraint,
                time_level=time_level,
            )
            suite.results.append(result_unaware)

            if result_unaware.success:
                print(
                    f"  ✓ {result_unaware.metrics.duration_seconds:.1f}s | "
                    f"{result_unaware.metrics.total_tool_calls} searches | "
                    f"score={result_unaware.correctness:.2f}"
                )
            else:
                print(f"  ✗ Failed: {result_unaware.error}")

            # Test with TIME-AWARE agent
            trial_num += 1
            print(f"[Trial {trial_num}/150] TIME-AWARE agent...")
            result_aware = await self.run_single_trial(
                question_id=question_id,
                task=task,
                condition=AwarenessCondition.AWARE,
                time_constraint=time_constraint,
                time_level=time_level,
            )
            suite.results.append(result_aware)

            if result_aware.success:
                print(
                    f"  ✓ {result_aware.metrics.duration_seconds:.1f}s | "
                    f"{result_aware.metrics.total_tool_calls} searches | "
                    f"score={result_aware.correctness:.2f}"
                )
            else:
                print(f"  ✗ Failed: {result_aware.error}")

            # Show paired comparison
            if result_unaware.success and result_aware.success:
                acc_diff = result_unaware.correctness - result_aware.correctness
                time_diff = (
                    result_aware.metrics.duration_seconds
                    - result_unaware.metrics.duration_seconds
                )
                search_diff = (
                    result_aware.metrics.total_tool_calls
                    - result_unaware.metrics.total_tool_calls
                )
                print(
                    f"  → Accuracy diff (unaware - aware): {acc_diff:+.2f} | "
                    f"Time diff: {time_diff:+.1f}s | Search diff: {search_diff:+d}"
                )

        suite.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return suite


async def main() -> None:
    """Run time awareness experiment."""
    print("=" * 80)
    print("PART 1: TIME AWARENESS EXPERIMENT")
    print("=" * 80)
    print()
    print("Testing: Does time awareness improve strategic resource allocation?")
    print()
    print("Design:")
    print("  - 75 questions (stratified by category)")
    print("  - Time level: Between-subjects (tight/moderate/comfortable)")
    print("  - Awareness: Within-subjects (each question tested in BOTH conditions)")
    print("  - Time-aware condition: Mechanistic explanation of time consumption")
    print("  - Total: 75 questions × 2 conditions = 150 data points")
    print()
    print("Time levels:")
    print("  - Tight: 30 seconds (forces efficiency)")
    print("  - Moderate: 60 seconds (comfortable)")
    print("  - Comfortable: 90 seconds (generous)")
    print()
    print("Hypothesis:")
    print("  Time awareness may work better than budget awareness because:")
    print("  - More concrete and actionable (seconds vs tokens)")
    print("  - Creates urgency (not just scarcity)")
    print("  - Clearer trade-offs for search decisions")
    print()
    print("Power: Can detect effect size d ≥ 0.45 with 80% power")
    print()
    print("=" * 80)
    print()

    # Run study
    runner = TimeAwarenessRunner()
    suite = await runner.run_time_awareness_study(n_questions=75, seed=900)

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
        time_diffs = []
        search_diffs = []

        for unaware, aware in pairs:
            acc_diff = unaware.correctness - aware.correctness
            time_diff = (
                aware.metrics.duration_seconds - unaware.metrics.duration_seconds
            )
            search_diff = aware.metrics.total_tool_calls - unaware.metrics.total_tool_calls
            accuracy_diffs.append(acc_diff)
            time_diffs.append(time_diff)
            search_diffs.append(search_diff)

        import numpy as np

        print("Paired Differences (Unaware - Aware):")
        print(
            f"  Accuracy: {np.mean(accuracy_diffs):+.3f} ± {np.std(accuracy_diffs):.3f}"
        )
        print(f"  Time:     {np.mean(time_diffs):+.1f}s ± {np.std(time_diffs):.1f}s")
        print(
            f"  Searches: {np.mean(search_diffs):+.1f} ± {np.std(search_diffs):.1f}"
        )
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

        # Search behavior
        fewer_searches = sum(1 for d in search_diffs if d < 0)
        more_searches = sum(1 for d in search_diffs if d > 0)
        same_searches = sum(1 for d in search_diffs if d == 0)

        print("Search Behavior (Time-aware vs Unaware):")
        print(f"  Fewer searches: {fewer_searches} ({100*fewer_searches/len(pairs):.1f}%)")
        print(f"  More searches:  {more_searches} ({100*more_searches/len(pairs):.1f}%)")
        print(f"  Same searches:  {same_searches} ({100*same_searches/len(pairs):.1f}%)")
        print()

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"part1_time_awareness_{timestamp}.json"

    with open(json_path, "w") as f:
        json.dump(suite.to_dict(), f, indent=2)

    print(f"✅ Results saved to: {json_path}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
