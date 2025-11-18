"""Experiment runner for executing and measuring agent performance."""

import time
from dataclasses import dataclass, field
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent_budget import AgentFactory, AllocationStrategy, UsageMonitor
from agent_budget.monitor import AgentMetrics
from experiments.tasks import ResearchTask


@dataclass
class ExperimentResult:
    """Results from a single experiment run.

    Attributes:
        task_id: ID of the research task
        strategy: Allocation strategy used
        question: The research question asked
        response: Agent's response
        metrics: Performance metrics
        budget_limit: Token budget for this experiment
        success: Whether the experiment completed successfully
        error: Error message if failed
    """

    task_id: str
    strategy: str
    question: str
    response: str
    metrics: AgentMetrics
    budget_limit: int
    success: bool = True
    error: str | None = None

    @property
    def within_budget(self) -> bool:
        """Check if experiment stayed within budget (±10% tolerance).

        Returns:
            True if within budget, False if exceeded by >10%
        """
        if not self.success or not self.metrics:
            return False
        tolerance = 0.10  # 10% tolerance
        max_allowed = self.budget_limit * (1 + tolerance)
        return self.metrics.total_tokens_used <= max_allowed

    @property
    def budget_utilization(self) -> float:
        """Calculate budget utilization percentage.

        Returns:
            Percentage of budget used (can exceed 100%)
        """
        if not self.success or not self.metrics:
            return 0.0
        return (self.metrics.total_tokens_used / self.budget_limit) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation with all experiment data
        """
        return {
            "task_id": self.task_id,
            "strategy": self.strategy,
            "question": self.question,
            "response": self.response,
            "budget_limit": self.budget_limit,
            "budget_utilization": self.budget_utilization,
            "within_budget": self.within_budget,
            "success": self.success,
            "error": self.error,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


@dataclass
class ExperimentSuite:
    """Collection of experiment results.

    Attributes:
        results: List of individual experiment results
        started_at: Experiment suite start timestamp
        completed_at: Experiment suite completion timestamp
    """

    results: list[ExperimentResult] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None

    def add_result(self, result: ExperimentResult) -> None:
        """Add an experiment result to the suite.

        Args:
            result: Experiment result to add
        """
        self.results.append(result)

    def get_results_by_strategy(self, strategy: str) -> list[ExperimentResult]:
        """Get all results for a specific strategy.

        Args:
            strategy: Strategy name to filter by

        Returns:
            List of results for the specified strategy
        """
        return [r for r in self.results if r.strategy == strategy]

    def get_successful_results(self) -> list[ExperimentResult]:
        """Get only successful experiment results.

        Returns:
            List of successful results
        """
        return [r for r in self.results if r.success]

    def get_within_budget_results(self) -> list[ExperimentResult]:
        """Get only results that stayed within budget (±10% tolerance).

        Returns:
            List of results within budget
        """
        return [r for r in self.results if r.within_budget]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary with all suite data
        """
        return {
            "total_experiments": len(self.results),
            "successful": len(self.get_successful_results()),
            "failed": len(self.results) - len(self.get_successful_results()),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": [r.to_dict() for r in self.results],
        }


class ExperimentRunner:
    """Runner for executing token allocation experiments.

    This class orchestrates the execution of research tasks using different
    allocation strategies, collecting metrics and responses for analysis.
    """

    def __init__(
        self, factory: AgentFactory | None = None, monitor: UsageMonitor | None = None
    ):
        """Initialize the experiment runner.

        Args:
            factory: AgentFactory instance (creates default if None)
            monitor: UsageMonitor instance (creates default if None)
        """
        self.factory = factory or AgentFactory()
        self.monitor = monitor or UsageMonitor()
        self.session_service = InMemorySessionService()

    async def run_single_experiment(
        self,
        task: ResearchTask,
        strategy: AllocationStrategy,
    ) -> ExperimentResult:
        """Run a single experiment with one task and strategy.

        Args:
            task: Research task to execute (includes recommended_budget)
            strategy: Allocation strategy to use

        Returns:
            ExperimentResult with response and metrics
        """
        try:
            # Use task's recommended budget for fair comparison
            total_budget = task.recommended_budget

            # Create agent for this strategy
            agent = self.factory.create_agent(strategy, total_budget)

            # Create runner and session
            runner = Runner(
                agent=agent, app_name="experiment", session_service=self.session_service
            )
            session = await runner.session_service.create_session(
                app_name="experiment", user_id="researcher"
            )

            # Execute task and collect events
            start_time = time.time()
            events = []

            # Create message content
            message = types.Content(role="user", parts=[types.Part(text=task.question)])

            async for event in runner.run_async(
                user_id="researcher", session_id=session.id, new_message=message
            ):
                events.append(event)

            duration = time.time() - start_time

            # Extract final response
            response = ""
            for event in reversed(events):
                if hasattr(event, "content") and event.content:
                    if hasattr(event.content, "parts") and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                response = part.text
                                break
                if response:
                    break

            # Extract metrics from events
            metrics = self.monitor.extract_metrics_from_events(
                events=events, strategy=strategy.value, duration=duration
            )

            return ExperimentResult(
                task_id=task.id,
                strategy=strategy.value,
                question=task.question,
                response=response,
                metrics=metrics,
                budget_limit=total_budget,
                success=True,
            )

        except Exception as e:
            # Return failed result with error information
            return ExperimentResult(
                task_id=task.id,
                strategy=strategy.value,
                question=task.question,
                response="",
                metrics=AgentMetrics(
                    strategy=strategy.value,
                    reasoning_tokens_used=0,
                    output_tokens_used=0,
                    total_tokens_used=0,
                    duration_seconds=0.0,
                ),
                budget_limit=task.recommended_budget,
                success=False,
                error=str(e),
            )

    async def run_experiment_suite(
        self,
        tasks: list[ResearchTask],
        strategies: list[AllocationStrategy] | None = None,
    ) -> ExperimentSuite:
        """Run a complete suite of experiments.

        Args:
            tasks: List of research tasks to execute (each has recommended_budget)
            strategies: List of strategies to test (defaults to all)

        Returns:
            ExperimentSuite with all results
        """
        if strategies is None:
            strategies = list(AllocationStrategy)

        suite = ExperimentSuite()
        suite.started_at = time.strftime("%Y-%m-%d %H:%M:%S")

        # Run each task with each strategy
        for task in tasks:
            for strategy in strategies:
                print(
                    f"Running {task.id} with {strategy.value} strategy "
                    f"(budget: {task.recommended_budget} tokens)..."
                )
                result = await self.run_single_experiment(task, strategy)
                suite.add_result(result)

                if result.success:
                    print(f"  ✓ Completed in {result.metrics.duration_seconds:.2f}s")
                    print(
                        f"    Tokens used: {result.metrics.total_tokens_used}/{task.recommended_budget}"
                    )

                    # Warn if budget exceeded
                    if result.metrics.total_tokens_used > task.recommended_budget:
                        excess = (
                            result.metrics.total_tokens_used - task.recommended_budget
                        )
                        excess_pct = (excess / task.recommended_budget) * 100
                        print(
                            f"    ⚠️  Exceeded budget by {excess} tokens ({excess_pct:.1f}%)"
                        )
                else:
                    print(f"  ✗ Failed: {result.error}")

        suite.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return suite
