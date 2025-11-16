"""Usage monitoring and analysis for agent execution.

This module provides classes for tracking token usage and tool calls
during agent execution.
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
            # Extract token usage from events with usage_metadata
            if hasattr(event, "usage_metadata") and event.usage_metadata:
                # Accumulate reasoning tokens (thinking)
                thinking = getattr(event.usage_metadata, "thinking_tokens", 0)
                if thinking:
                    reasoning_tokens += thinking

                # Accumulate output tokens (candidates)
                candidates = getattr(event.usage_metadata, "candidates_token_count", 0)
                if candidates:
                    output_tokens += candidates

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
