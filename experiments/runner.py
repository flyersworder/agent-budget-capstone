"""Experiment runner for executing and measuring agent performance."""

import time
from dataclasses import dataclass, field
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

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
        success: Whether the experiment completed successfully
        error: Error message if failed
    """

    task_id: str
    strategy: str
    question: str
    response: str
    metrics: AgentMetrics
    success: bool = True
    error: str | None = None

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
        total_budget: int = 10000,
    ) -> ExperimentResult:
        """Run a single experiment with one task and strategy.

        Args:
            task: Research task to execute
            strategy: Allocation strategy to use
            total_budget: Total token budget

        Returns:
            ExperimentResult with response and metrics
        """
        try:
            # Create agent for this strategy
            agent = self.factory.create_agent(strategy, total_budget)

            # Create runner and session
            runner = Runner(agent=agent, session_service=self.session_service)
            session = await runner.session_service.create_session()

            # Execute task and collect events
            start_time = time.time()
            events = []

            async for event in runner.run_async(
                user_message=task.question, session_id=session.id
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
                success=False,
                error=str(e),
            )

    async def run_experiment_suite(
        self,
        tasks: list[ResearchTask],
        strategies: list[AllocationStrategy] | None = None,
        total_budget: int = 10000,
    ) -> ExperimentSuite:
        """Run a complete suite of experiments.

        Args:
            tasks: List of research tasks to execute
            strategies: List of strategies to test (defaults to all)
            total_budget: Total token budget per experiment

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
                print(f"Running {task.id} with {strategy.value} strategy...")
                result = await self.run_single_experiment(task, strategy, total_budget)
                suite.add_result(result)

                if result.success:
                    print(f"  ✓ Completed in {result.metrics.duration_seconds:.2f}s")
                    print(f"    Tokens used: {result.metrics.total_tokens_used}")
                else:
                    print(f"  ✗ Failed: {result.error}")

        suite.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return suite
