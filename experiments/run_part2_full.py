"""Part 2: Multi-Agent Budget Coordination - Full Study.

2×4 Factorial Design testing how reasoning complexity moderates budget awareness effects.

Design:
- 2 complexity levels × 4 awareness conditions × 25 questions = 200 trials
- Complexity: SIMPLE (2 supporting facts) vs COMPLEX (3 supporting facts)
- Awareness: NO_AWARENESS, OVERALL_ONLY, OVERALL_AND_INDIVIDUAL, RESERVE_AWARENESS
- Max 3 iterations per trial
- Track: iterations, token usage per agent, approval status, answer correctness

Moderator: Number of supporting facts (ground-truth from HotpotQA dataset)
- SIMPLE: 2-hop reasoning chains (A→B)
- COMPLEX: 3-hop reasoning chains (A→B→C)

Usage:
    python -m experiments.run_part2_full
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent_budget.agent_factory import AgentFactory
from agent_budget.core import IterativeTeamConfig, MultiAgentAwarenessCondition
from agent_budget.monitor import MultiAgentMetrics, UsageMonitor
from experiments.evaluator_hotpotqa import HotpotQAEvaluator, HotpotQAScore
from experiments.tasks.hotpotqa_tasks import HotpotQATask, get_supporting_facts_sample


@dataclass
class Part2TrialResult:
    """Results from a single trial (1 question, 1 condition, 1 complexity level).

    Attributes:
        trial_id: Unique trial identifier
        question_id: HotpotQA question ID
        question: The question text
        question_type: Type of question (bridge or comparison)
        complexity: Complexity level (SIMPLE or COMPLEX)
        ground_truth: Correct answer
        awareness_condition: Budget awareness condition
        researcher_output: Final researcher output
        validator_feedback: Final validator feedback
        approved: Whether validator approved
        num_iterations: Number of iterations completed
        max_iterations_reached: Whether loop hit max iterations
        metrics: Multi-agent execution metrics
        correctness_score: Evaluation score for answer correctness
        duration_seconds: Total execution time
    """

    trial_id: str
    question_id: str
    question: str
    question_type: str
    complexity: str
    ground_truth: str
    awareness_condition: str
    researcher_output: str
    validator_feedback: str
    approved: bool
    num_iterations: int
    max_iterations_reached: bool
    metrics: MultiAgentMetrics
    correctness_score: HotpotQAScore
    duration_seconds: float
    budget_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "trial_id": self.trial_id,
            "question_id": self.question_id,
            "question": self.question,
            "question_type": self.question_type,
            "complexity": self.complexity,
            "ground_truth": self.ground_truth,
            "awareness_condition": self.awareness_condition,
            "researcher_output": self.researcher_output,
            "validator_feedback": self.validator_feedback,
            "approved": self.approved,
            "num_iterations": self.num_iterations,
            "max_iterations_reached": self.max_iterations_reached,
            "metrics": self.metrics.to_dict(),
            "correctness_score": self.correctness_score.to_dict(),
            "duration_seconds": self.duration_seconds,
            "budget_messages": self.budget_messages,
        }

    @property
    def is_correct(self) -> bool:
        """Helper property for correctness."""
        return self.correctness_score.score == 1.0


@dataclass
class Part2FullResults:
    """Results from the full study.

    Attributes:
        trials: List of individual trial results
        started_at: Study start timestamp
        completed_at: Study completion timestamp
        design: Study design description
    """

    trials: list[Part2TrialResult] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    design: str = "2x4 factorial: complexity × awareness"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "design": self.design,
            "total_trials": len(self.trials),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "trials": [t.to_dict() for t in self.trials],
        }


async def run_single_trial(
    task: HotpotQATask,
    complexity: str,
    condition: MultiAgentAwarenessCondition,
    factory: AgentFactory,
    evaluator: HotpotQAEvaluator,
    trial_id: str,
) -> Part2TrialResult:
    """Run a single trial with one question and one awareness condition.

    Args:
        task: HotpotQA question to answer
        complexity: Complexity level (SIMPLE or COMPLEX)
        condition: Budget awareness condition
        factory: AgentFactory for creating teams
        evaluator: Evaluator for scoring answers
        trial_id: Unique trial identifier

    Returns:
        Part2TrialResult with all metrics and evaluation
    """
    print(f"\n{'=' * 80}")
    print(f"Trial: {trial_id}")
    print(f"Complexity: {complexity}")
    print(f"Question: {task.question[:60]}...")
    print(f"Condition: {condition.value}")
    print(f"{'=' * 80}")

    # Create team configuration
    if condition == MultiAgentAwarenessCondition.RESERVE_AWARENESS:
        config = IterativeTeamConfig.create_reserve_awareness()
    else:
        config = IterativeTeamConfig.create_standard(awareness_condition=condition)

    # Create iterative team
    team = factory.create_iterative_team(config)

    # Create session
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="part2_full",
        user_id="researcher",
        session_id=trial_id,
    )

    runner = Runner(
        agent=team,
        app_name="part2_full",
        session_service=session_service,
    )

    # Run the trial
    start_time = time.time()
    events = []
    budget_messages = []

    try:
        content = types.Content(role="user", parts=[types.Part(text=task.question)])

        async for event in runner.run_async(
            user_id="researcher",
            session_id=trial_id,
            new_message=content,
        ):
            events.append(event)

            # Collect budget messages from CheckApproval
            if hasattr(event, "author") and event.author == "CheckApproval":
                if hasattr(event, "content") and event.content:
                    if hasattr(event.content, "parts"):
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                budget_messages.append(part.text)

            # Print progress
            if hasattr(event, "author") and event.author:
                if event.author in ["Researcher", "Validator"]:
                    print(f"  [{event.author}] generated response")

        duration = time.time() - start_time

        # Get final session state
        final_session = await session_service.get_session(
            app_name="part2_full",
            user_id="researcher",
            session_id=trial_id,
        )

        # Extract metrics
        monitor = UsageMonitor()
        metrics = monitor.extract_multi_agent_metrics(
            events=events,
            session_state=final_session.state,
            awareness_condition=condition.value,
            max_iterations=config.max_iterations,
            duration=duration,
        )

        # Extract final outputs
        researcher_output = final_session.state.get("researcher_output", "")
        validator_feedback = final_session.state.get("validator_feedback", "")

        # Evaluate answer correctness
        answer_to_evaluate = (
            validator_feedback if metrics.approved else researcher_output
        )
        correctness_score = evaluator.evaluate_correctness(task, answer_to_evaluate)

        print("\n  Results:")
        print(f"    Iterations: {metrics.num_iterations}")
        print(f"    Approved: {metrics.approved}")
        print(f"    Correct: {correctness_score.score == 1.0}")
        print(f"    Total tokens: {metrics.total_tokens}")
        print(f"    Duration: {duration:.2f}s")

        return Part2TrialResult(
            trial_id=trial_id,
            question_id=task.id,
            question=task.question,
            question_type=task.question_type,
            complexity=complexity,
            ground_truth=task.answer,
            awareness_condition=condition.value,
            researcher_output=researcher_output,
            validator_feedback=validator_feedback,
            approved=metrics.approved,
            num_iterations=metrics.num_iterations,
            max_iterations_reached=metrics.max_iterations_reached,
            metrics=metrics,
            correctness_score=correctness_score,
            duration_seconds=duration,
            budget_messages=budget_messages,
        )

    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback

        traceback.print_exc()

        # Return failed trial
        duration = time.time() - start_time
        empty_metrics = MultiAgentMetrics(
            awareness_condition=condition.value,
            num_iterations=0,
            researcher_output="",
            validator_feedback="",
            researcher_tokens=0,
            validator_tokens=0,
            approved=False,
            max_iterations_reached=False,
            duration_seconds=duration,
        )
        empty_score = HotpotQAScore(
            score=0.0,
            justification=f"Trial failed: {str(e)}",
            question=task.question,
            correct_answer=task.answer,
            agent_response="",
            question_type=task.question_type,
        )

        return Part2TrialResult(
            trial_id=trial_id,
            question_id=task.id,
            question=task.question,
            question_type=task.question_type,
            complexity=complexity,
            ground_truth=task.answer,
            awareness_condition=condition.value,
            researcher_output="",
            validator_feedback="",
            approved=False,
            num_iterations=0,
            max_iterations_reached=False,
            metrics=empty_metrics,
            correctness_score=empty_score,
            duration_seconds=duration,
            budget_messages=[],
        )


async def run_full_study() -> Part2FullResults:
    """Run the full Part 2 study with 2×4 factorial design.

    Returns:
        Part2FullResults with all trial results
    """
    print("=" * 80)
    print("PART 2 FULL STUDY: Multi-Agent Budget Coordination")
    print("2×4 Factorial Design: Complexity × Awareness")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - 2 complexity levels (SIMPLE, COMPLEX)")
    print("  - 4 awareness conditions")
    print("  - 25 questions per complexity level")
    print("  - Total: 200 trials (2 × 4 × 25)")
    print("  - Max 3 iterations per trial")
    print()
    print("Complexity Levels (based on # of supporting facts):")
    print("  SIMPLE: 2 supporting facts (2-hop reasoning: A→B)")
    print("  COMPLEX: 3 supporting facts (3-hop reasoning: A→B→C)")
    print()
    print("Awareness Conditions:")
    print("  A. NO_AWARENESS: No budget information")
    print("  B. OVERALL_ONLY: Team budget only")
    print("  C. OVERALL_AND_INDIVIDUAL: Team + individual budgets")
    print("  D. RESERVE_AWARENESS: Team + individual + reserve pool")
    print()
    print("=" * 80)

    # Load environment variables
    load_dotenv()

    # Load questions stratified by supporting facts
    print("\nLoading HotpotQA questions...")
    questions_by_complexity = get_supporting_facts_sample(n_per_complexity=25, seed=42)
    print(f"Loaded {len(questions_by_complexity['SIMPLE'])} SIMPLE questions (2 facts)")
    print(
        f"Loaded {len(questions_by_complexity['COMPLEX'])} COMPLEX questions (3 facts)"
    )

    # Initialize components
    factory = AgentFactory()
    evaluator = HotpotQAEvaluator()

    # Define conditions
    conditions = [
        MultiAgentAwarenessCondition.NO_AWARENESS,
        MultiAgentAwarenessCondition.OVERALL_ONLY,
        MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL,
        MultiAgentAwarenessCondition.RESERVE_AWARENESS,
    ]

    # Run all trials: 2 complexity × 4 awareness × 25 questions = 200 trials
    results = Part2FullResults()
    results.started_at = datetime.now().isoformat()

    trial_num = 0
    total_trials = 2 * len(conditions) * 25

    for complexity in ["SIMPLE", "COMPLEX"]:
        questions = questions_by_complexity[complexity]
        for condition in conditions:
            for question in questions:
                trial_num += 1
                trial_id = (
                    f"full_{trial_num:03d}_{complexity}_{condition.value}_{question.id}"
                )

                print(f"\n[Progress: {trial_num}/{total_trials}]")

                result = await run_single_trial(
                    task=question,
                    complexity=complexity,
                    condition=condition,
                    factory=factory,
                    evaluator=evaluator,
                    trial_id=trial_id,
                )

                results.trials.append(result)

    results.completed_at = datetime.now().isoformat()

    return results


def print_summary(results: Part2FullResults) -> None:
    """Print summary of full study results."""
    print("\n" + "=" * 80)
    print("FULL STUDY SUMMARY")
    print("=" * 80)

    print(f"\nTotal trials: {len(results.trials)}")
    print(f"Started: {results.started_at}")
    print(f"Completed: {results.completed_at}")

    # Overall statistics
    total_correct = sum(1 for t in results.trials if t.is_correct)
    avg_iterations = sum(t.num_iterations for t in results.trials) / len(results.trials)
    avg_tokens = sum(t.metrics.total_tokens for t in results.trials) / len(
        results.trials
    )

    print("\nOverall Performance:")
    print(
        f"  Correct answers: {total_correct}/{len(results.trials)} ({total_correct / len(results.trials) * 100:.1f}%)"
    )
    print(f"  Avg iterations: {avg_iterations:.2f}")
    print(f"  Avg total tokens: {avg_tokens:.0f}")

    # By complexity
    print("\nPerformance by Complexity:")
    for complexity in ["SIMPLE", "COMPLEX"]:
        trials = [t for t in results.trials if t.complexity == complexity]
        correct = sum(1 for t in trials if t.is_correct)
        avg_iter = sum(t.num_iterations for t in trials) / len(trials)
        avg_tok = sum(t.metrics.total_tokens for t in trials) / len(trials)

        print(f"\n  {complexity}:")
        print(
            f"    Correct: {correct}/{len(trials)} ({correct / len(trials) * 100:.1f}%)"
        )
        print(f"    Avg iterations: {avg_iter:.2f}")
        print(f"    Avg tokens: {avg_tok:.0f}")

    # By awareness condition
    print("\nPerformance by Awareness Condition:")
    for condition in [
        "no_awareness",
        "overall_only",
        "overall_and_individual",
        "reserve_awareness",
    ]:
        trials = [t for t in results.trials if t.awareness_condition == condition]
        if not trials:
            continue

        correct = sum(1 for t in trials if t.is_correct)
        avg_iter = sum(t.num_iterations for t in trials) / len(trials)
        avg_tok = sum(t.metrics.total_tokens for t in trials) / len(trials)

        print(f"\n  {condition.upper()}:")
        print(
            f"    Correct: {correct}/{len(trials)} ({correct / len(trials) * 100:.1f}%)"
        )
        print(f"    Avg iterations: {avg_iter:.2f}")
        print(f"    Avg tokens: {avg_tok:.0f}")

    # 2×4 Interaction Table
    print("\n2×4 Factorial Results (Accuracy %):")
    print(f"{'Condition':<30} {'SIMPLE':>10} {'COMPLEX':>10}")
    print("-" * 52)

    for condition in [
        "no_awareness",
        "overall_only",
        "overall_and_individual",
        "reserve_awareness",
    ]:
        row = [condition.replace("_", " ").title()]
        for complexity in ["SIMPLE", "COMPLEX"]:
            trials = [
                t
                for t in results.trials
                if t.awareness_condition == condition and t.complexity == complexity
            ]
            if trials:
                correct = sum(1 for t in trials if t.is_correct)
                pct = correct / len(trials) * 100
                row.append(f"{pct:.1f}%")
            else:
                row.append("N/A")

        print(f"{row[0]:<30} {row[1]:>10} {row[2]:>10}")

    print("\n" + "=" * 80)


def save_results(results: Part2FullResults) -> None:
    """Save full study results to JSON file."""
    output_dir = Path("experiments/results/part2_full")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"full_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_file}")


async def main() -> None:
    """Main entry point."""
    results = await run_full_study()
    print_summary(results)
    save_results(results)

    print("\n" + "=" * 80)
    print("✅ FULL STUDY COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review results in experiments/results/part2_full/")
    print("  2. Run factorial analysis with bootstrap (analyze_part2_full.py)")
    print("  3. Generate publication-ready figures")


if __name__ == "__main__":
    asyncio.run(main())
