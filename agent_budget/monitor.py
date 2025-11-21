"""Usage monitoring and analysis for agent execution.

This module provides classes for tracking token usage and tool calls
during agent execution, including multi-agent iterative workflows.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolUsageMetrics:
    """Track tool usage patterns.

    Attributes:
        tool_name: Name of the tool
        call_count: Number of times the tool was called
        total_tokens: Total tokens consumed by tool calls (if available)
    """

    tool_name: str
    call_count: int
    total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            "tool": self.tool_name,
            "calls": self.call_count,
            "tokens": self.total_tokens,
        }


@dataclass
class AgentMetrics:
    """Comprehensive agent execution metrics.

    Attributes:
        strategy: Allocation strategy name
        reasoning_tokens_used: Actual reasoning tokens consumed
        output_tokens_used: Actual output tokens consumed
        total_tokens_used: Total tokens consumed
        tool_usage: List of tool usage metrics
        duration_seconds: Execution duration in seconds
    """

    strategy: str
    reasoning_tokens_used: int
    output_tokens_used: int
    total_tokens_used: int
    tool_usage: list[ToolUsageMetrics] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def total_tool_calls(self) -> int:
        """Total number of tool calls across all tools.

        Returns:
            Sum of all tool call counts
        """
        return sum(t.call_count for t in self.tool_usage)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation with all metrics
        """
        return {
            "strategy": self.strategy,
            "reasoning_tokens_used": self.reasoning_tokens_used,
            "output_tokens_used": self.output_tokens_used,
            "total_tokens_used": self.total_tokens_used,
            "total_tool_calls": self.total_tool_calls,
            "tool_usage_details": [t.to_dict() for t in self.tool_usage],
            "duration_seconds": self.duration_seconds,
        }

    def __repr__(self) -> str:
        """String representation of metrics."""
        return (
            f"AgentMetrics(strategy={self.strategy}, "
            f"tokens={self.total_tokens_used}, "
            f"tool_calls={self.total_tool_calls}, "
            f"duration={self.duration_seconds:.2f}s)"
        )


@dataclass
class MultiAgentMetrics:
    """Comprehensive metrics for multi-agent iterative execution.

    Tracks execution across multiple agents and iterations, including
    token usage per agent and iteration control metrics.

    Attributes:
        awareness_condition: Budget awareness condition name
        num_iterations: Number of iterations completed
        researcher_output: Final researcher output
        validator_feedback: Final validator feedback
        researcher_tokens: Total tokens used by Researcher
        validator_tokens: Total tokens used by Validator
        approved: Whether Validator approved the final answer
        max_iterations_reached: Whether loop hit max iteration limit
        tool_usage: List of tool usage metrics across all agents
        duration_seconds: Total execution duration
    """

    awareness_condition: str
    num_iterations: int
    researcher_output: str
    validator_feedback: str
    researcher_tokens: int
    validator_tokens: int
    approved: bool
    max_iterations_reached: bool
    tool_usage: list[ToolUsageMetrics] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens used by both agents.

        Returns:
            Sum of researcher and validator tokens
        """
        return self.researcher_tokens + self.validator_tokens

    @property
    def total_tool_calls(self) -> int:
        """Total number of tool calls across all agents.

        Returns:
            Sum of all tool call counts
        """
        return sum(t.call_count for t in self.tool_usage)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation with all metrics
        """
        return {
            "awareness_condition": self.awareness_condition,
            "num_iterations": self.num_iterations,
            "approved": self.approved,
            "max_iterations_reached": self.max_iterations_reached,
            "researcher_tokens": self.researcher_tokens,
            "validator_tokens": self.validator_tokens,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "tool_usage_details": [t.to_dict() for t in self.tool_usage],
            "researcher_output": self.researcher_output,
            "validator_feedback": self.validator_feedback,
            "duration_seconds": self.duration_seconds,
        }

    def __repr__(self) -> str:
        """String representation of metrics."""
        return (
            f"MultiAgentMetrics(condition={self.awareness_condition}, "
            f"iterations={self.num_iterations}, "
            f"tokens={self.total_tokens}, "
            f"approved={self.approved})"
        )


class UsageMonitor:
    """Monitor and analyze agent usage from session events.

    This class extracts usage metrics from Google ADK runner events,
    tracking token consumption and tool usage patterns.
    """

    def extract_metrics_from_events(
        self, events: list[Any], strategy: str, duration: float
    ) -> AgentMetrics:
        """Extract usage metrics from runner events.

        Args:
            events: List of events from runner.run_async()
            strategy: Allocation strategy name
            duration: Execution duration in seconds

        Returns:
            AgentMetrics instance with extracted data
        """
        reasoning_tokens = 0
        output_tokens = 0
        tool_usage: dict[str, dict[str, int]] = {}

        for event in events:
            # Track tool use prompt tokens for this event
            event_tool_tokens = 0

            # Extract token usage from events with usage_metadata
            if hasattr(event, "usage_metadata") and event.usage_metadata:
                # Accumulate reasoning tokens (thinking)
                thinking = getattr(event.usage_metadata, "thoughts_token_count", 0)
                if thinking:
                    reasoning_tokens += thinking

                # Accumulate output tokens (candidates)
                candidates = getattr(event.usage_metadata, "candidates_token_count", 0)
                if candidates:
                    output_tokens += candidates

                # Track tool use prompt tokens (for grounding results)
                event_tool_tokens = (
                    getattr(event.usage_metadata, "tool_use_prompt_token_count", 0) or 0
                )

            # Track tool calls
            # Note: Event structure may vary - this is a best-effort approach
            if hasattr(event, "tool_name") and event.tool_name:
                tool_name = event.tool_name
                if tool_name not in tool_usage:
                    tool_usage[tool_name] = {"count": 0, "tokens": 0}
                tool_usage[tool_name]["count"] += 1

            # Alternative: Check for function calls in content
            if hasattr(event, "content") and event.content:
                if hasattr(event.content, "parts") and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            tool_name = part.function_call.name
                            if tool_name not in tool_usage:
                                tool_usage[tool_name] = {"count": 0, "tokens": 0}
                            tool_usage[tool_name]["count"] += 1

            # Check for grounding metadata (google_search uses this instead of function_call)
            if hasattr(event, "grounding_metadata") and event.grounding_metadata:
                gm = event.grounding_metadata
                # Check if web_search_queries exist (indicates google_search was used)
                if hasattr(gm, "web_search_queries") and gm.web_search_queries:
                    tool_name = "google_search"
                    if tool_name not in tool_usage:
                        tool_usage[tool_name] = {"count": 0, "tokens": 0}
                    # Count number of search queries as separate calls
                    tool_usage[tool_name]["count"] += len(gm.web_search_queries)
                    # Add grounding result tokens (from tool_use_prompt_token_count)
                    if event_tool_tokens:
                        tool_usage[tool_name]["tokens"] += event_tool_tokens

        # Convert tool usage to ToolUsageMetrics objects
        tool_metrics = [
            ToolUsageMetrics(
                tool_name=name, call_count=data["count"], total_tokens=data["tokens"]
            )
            for name, data in tool_usage.items()
        ]

        return AgentMetrics(
            strategy=strategy,
            reasoning_tokens_used=reasoning_tokens,
            output_tokens_used=output_tokens,
            total_tokens_used=reasoning_tokens + output_tokens,
            tool_usage=tool_metrics,
            duration_seconds=duration,
        )

    def extract_multi_agent_metrics(
        self,
        events: list[Any],
        session_state: dict[str, Any],
        awareness_condition: str,
        max_iterations: int,
        duration: float,
    ) -> MultiAgentMetrics:
        """Extract metrics from multi-agent iterative execution.

        Args:
            events: List of events from runner.run_async()
            session_state: Final session state with agent outputs
            awareness_condition: Budget awareness condition name
            max_iterations: Maximum iterations allowed
            duration: Execution duration in seconds

        Returns:
            MultiAgentMetrics with extracted data
        """
        # Extract final outputs from session state
        researcher_output = session_state.get("researcher_output", "")
        validator_feedback = session_state.get("validator_feedback", "")

        # Check if approved
        approved = "APPROVED" in str(validator_feedback)

        # Count iterations by counting CheckApproval events
        # Each CheckApproval event represents one iteration cycle
        check_events = [
            e for e in events if hasattr(e, "author") and e.author == "CheckApproval"
        ]
        num_iterations = len(check_events)

        # Check if max iterations reached
        max_iterations_reached = num_iterations >= max_iterations

        # Extract token usage per agent
        researcher_tokens = 0
        validator_tokens = 0
        tool_usage: dict[str, dict[str, int]] = {}

        for event in events:
            # Track tool use prompt tokens for this event
            event_tool_tokens = 0

            # Extract token usage from events with usage_metadata
            if hasattr(event, "usage_metadata") and event.usage_metadata:
                # Get thinking + output tokens
                thinking = getattr(event.usage_metadata, "thoughts_token_count", 0) or 0
                candidates = (
                    getattr(event.usage_metadata, "candidates_token_count", 0) or 0
                )
                event_tokens = thinking + candidates

                # Track tool use prompt tokens
                event_tool_tokens = (
                    getattr(event.usage_metadata, "tool_use_prompt_token_count", 0) or 0
                )

                # Attribute tokens to correct agent
                if hasattr(event, "author"):
                    if event.author == "Researcher":
                        researcher_tokens += event_tokens
                    elif event.author == "Validator":
                        validator_tokens += event_tokens

            # Track tool calls (same logic as single-agent)
            # Check for grounding metadata (google_search)
            if hasattr(event, "grounding_metadata") and event.grounding_metadata:
                gm = event.grounding_metadata
                if hasattr(gm, "web_search_queries") and gm.web_search_queries:
                    tool_name = "google_search"
                    if tool_name not in tool_usage:
                        tool_usage[tool_name] = {"count": 0, "tokens": 0}
                    tool_usage[tool_name]["count"] += len(gm.web_search_queries)
                    if event_tool_tokens:
                        tool_usage[tool_name]["tokens"] += event_tool_tokens

            # Check for function calls in content
            if hasattr(event, "content") and event.content:
                if hasattr(event.content, "parts") and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            tool_name = part.function_call.name
                            if tool_name not in tool_usage:
                                tool_usage[tool_name] = {"count": 0, "tokens": 0}
                            tool_usage[tool_name]["count"] += 1

        # Convert tool usage to ToolUsageMetrics objects
        tool_metrics = [
            ToolUsageMetrics(
                tool_name=name, call_count=data["count"], total_tokens=data["tokens"]
            )
            for name, data in tool_usage.items()
        ]

        return MultiAgentMetrics(
            awareness_condition=awareness_condition,
            num_iterations=num_iterations,
            researcher_output=researcher_output,
            validator_feedback=validator_feedback,
            researcher_tokens=researcher_tokens,
            validator_tokens=validator_tokens,
            approved=approved,
            max_iterations_reached=max_iterations_reached,
            tool_usage=tool_metrics,
            duration_seconds=duration,
        )

    def compare_strategies(self, metrics_list: list[AgentMetrics]) -> dict[str, Any]:
        """Compare metrics across multiple strategies.

        Args:
            metrics_list: List of AgentMetrics to compare

        Returns:
            Dictionary with comparative analysis
        """
        if not metrics_list:
            return {"error": "No metrics provided"}

        comparison: dict[str, Any] = {
            "strategies": [],
            "total_tokens": {},
            "tool_calls": {},
            "duration": {},
        }

        for metrics in metrics_list:
            strategy = metrics.strategy
            comparison["strategies"].append(strategy)
            comparison["total_tokens"][strategy] = metrics.total_tokens_used
            comparison["tool_calls"][strategy] = metrics.total_tool_calls
            comparison["duration"][strategy] = metrics.duration_seconds

        # Calculate averages and ranges
        comparison["avg_tokens"] = sum(comparison["total_tokens"].values()) / len(
            metrics_list
        )
        comparison["avg_tool_calls"] = sum(comparison["tool_calls"].values()) / len(
            metrics_list
        )
        comparison["avg_duration"] = sum(comparison["duration"].values()) / len(
            metrics_list
        )

        return comparison
