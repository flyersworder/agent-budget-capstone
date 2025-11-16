"""Core framework for token budget allocation strategies.

This module provides the foundational classes for managing token budgets
across reasoning and output tokens in AI agents.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass
class TokenBudget:
    """Strategic token budget allocation.

    Attributes:
        reasoning_tokens: Tokens allocated for thinking/analysis
        output_tokens: Tokens allocated for response generation
    """

    reasoning_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        """Total token budget (reasoning + output)."""
        return self.reasoning_tokens + self.output_tokens

    def validate(self) -> bool:
        """Ensure non-negative budgets.

        Returns:
            True if both budgets are non-negative, False otherwise
        """
        return self.reasoning_tokens >= 0 and self.output_tokens >= 0

    def __repr__(self) -> str:
        """String representation of budget."""
        return (
            f"TokenBudget(reasoning={self.reasoning_tokens}, "
            f"output={self.output_tokens}, total={self.total})"
        )


class AllocationStrategy(Enum):
    """Single-agent allocation strategies.

    Each strategy represents a different tradeoff between reasoning depth
    and output verbosity.
    """

    DEEP_THINKER = "deep"  # High reasoning, low output (80/20)
    BALANCED = "balanced"  # Equal split (50/50)
    VERBOSE = "verbose"  # Low reasoning, high output (20/80)

    def create_budget(self, total_tokens: int) -> TokenBudget:
        """Create budget for this strategy.

        Args:
            total_tokens: Total token budget to allocate

        Returns:
            TokenBudget instance with strategy-specific allocation

        Raises:
            ValueError: If total_tokens is negative
        """
        if total_tokens < 0:
            raise ValueError("Total tokens must be non-negative")

        ratios = {
            "deep": (0.8, 0.2),  # Deep Thinker
            "balanced": (0.5, 0.5),  # Balanced
            "verbose": (0.2, 0.8),  # Verbose
        }

        reasoning_ratio, output_ratio = ratios[self.value]

        return TokenBudget(
            reasoning_tokens=int(total_tokens * reasoning_ratio),
            output_tokens=int(total_tokens * output_ratio),
        )

    def description(self) -> str:
        """Human-readable description of the strategy.

        Returns:
            Strategy description
        """
        descriptions = {
            "deep": (
                "Deep Thinker: Prioritizes thorough analysis and reasoning "
                "over verbose output. Best for complex analytical tasks."
            ),
            "balanced": (
                "Balanced: Equal allocation between reasoning and output. "
                "Good general-purpose strategy for most tasks."
            ),
            "verbose": (
                "Verbose: Prioritizes detailed, comprehensive output over "
                "deep reasoning. Best for tasks requiring extensive explanation."
            ),
        }
        return descriptions[self.value]
