"""Core framework for token budget allocation strategies.

This module provides the foundational classes for managing token budgets
across reasoning and output tokens in AI agents.

Includes support for both single-agent and multi-agent configurations.
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
    """Single-agent allocation strategies (Part 1 - ARCHIVED).

    Each strategy represents a different tradeoff between reasoning depth
    and output verbosity.

    NOTE: This is from the archived Part 1 (allocation strategies study).
    For current research, see awareness.py and multi-agent configs below.
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


# ============================================================================
# PART 2: Multi-Agent Configurations
# ============================================================================


class AgentRole(Enum):
    """Roles in multi-agent teams (Part 2).

    Each role has specific responsibilities in the sequential workflow.
    """

    RESEARCHER = "researcher"  # Gathers information using tools
    ANALYZER = "analyzer"  # Evaluates information for correctness
    SYNTHESIZER = "synthesizer"  # Produces final answer


class MultiAgentAwarenessCondition(Enum):
    """Budget awareness conditions for multi-agent teams (Part 2)."""

    NO_AWARENESS = "no_awareness"  # Baseline: no budget information
    OVERALL_ONLY = "overall_only"  # Know total team budget only
    OVERALL_AND_INDIVIDUAL = "overall_and_individual"  # Know total + own allocation
    WITH_NEGOTIATION = "with_negotiation"  # Can request additional budget


@dataclass
class RoleAllocation:
    """Budget allocation for a specific agent role.

    Attributes:
        role: Agent role
        budget: Token budget for this role
        percentage: Percentage of total team budget (for reporting)
    """

    role: AgentRole
    budget: TokenBudget
    percentage: float


@dataclass
class MultiAgentBudgetConfig:
    """Budget configuration for multi-agent teams.

    Attributes:
        total_budget: Total team budget (sum of all agent budgets)
        allocations: Budget allocation for each role
        reserve_pool: Tokens held in reserve for negotiation (condition D only)
        awareness_condition: How budget info is communicated to agents
    """

    total_budget: int
    allocations: dict[AgentRole, RoleAllocation]
    reserve_pool: int = 0
    awareness_condition: MultiAgentAwarenessCondition = (
        MultiAgentAwarenessCondition.NO_AWARENESS
    )

    @property
    def allocated_budget(self) -> int:
        """Total allocated across all roles."""
        return sum(alloc.budget.total for alloc in self.allocations.values())

    @property
    def remaining_pool(self) -> int:
        """Remaining budget in reserve pool."""
        return self.reserve_pool

    def validate(self) -> bool:
        """Ensure configuration is valid.

        Returns:
            True if valid, False otherwise
        """
        # Total allocated + reserve should equal total budget
        total_used = self.allocated_budget + self.reserve_pool
        return total_used == self.total_budget

    @staticmethod
    def create_equal_split(
        total_budget: int,
        num_agents: int = 3,
        awareness_condition: MultiAgentAwarenessCondition = MultiAgentAwarenessCondition.NO_AWARENESS,
    ) -> "MultiAgentBudgetConfig":
        """Create config with equal budget split across agents.

        Args:
            total_budget: Total team budget
            num_agents: Number of agents (default: 3)
            awareness_condition: How to communicate budget

        Returns:
            MultiAgentBudgetConfig with equal allocation
        """
        per_agent = total_budget // num_agents
        remainder = total_budget % num_agents  # Handle rounding
        roles = list(AgentRole)[:num_agents]

        allocations = {}
        for i, role in enumerate(roles):
            # Give remainder to last agent
            agent_budget = per_agent + (remainder if i == len(roles) - 1 else 0)

            allocations[role] = RoleAllocation(
                role=role,
                budget=TokenBudget(
                    reasoning_tokens=agent_budget // 2,  # 50/50 split within agent
                    output_tokens=agent_budget // 2,
                ),
                percentage=(agent_budget / total_budget) * 100,
            )

        return MultiAgentBudgetConfig(
            total_budget=total_budget,
            allocations=allocations,
            reserve_pool=0,
            awareness_condition=awareness_condition,
        )

    @staticmethod
    def create_role_based(
        total_budget: int,
        awareness_condition: MultiAgentAwarenessCondition = MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL,
    ) -> "MultiAgentBudgetConfig":
        """Create config with role-based allocation.

        Researcher: 40% (needs tools, thinking)
        Analyzer: 35% (deep reasoning)
        Synthesizer: 25% (concise output)

        Args:
            total_budget: Total team budget
            awareness_condition: How to communicate budget

        Returns:
            MultiAgentBudgetConfig with role-based allocation
        """
        allocations = {}

        # Researcher: 40%, focus on reasoning for tool use
        researcher_total = int(total_budget * 0.40)
        researcher_reasoning = int(researcher_total * 0.7)
        allocations[AgentRole.RESEARCHER] = RoleAllocation(
            role=AgentRole.RESEARCHER,
            budget=TokenBudget(
                reasoning_tokens=researcher_reasoning,
                output_tokens=researcher_total
                - researcher_reasoning,  # Derive to avoid rounding
            ),
            percentage=40.0,
        )

        # Analyzer: 35%, balanced
        analyzer_total = int(total_budget * 0.35)
        analyzer_reasoning = int(analyzer_total * 0.6)
        allocations[AgentRole.ANALYZER] = RoleAllocation(
            role=AgentRole.ANALYZER,
            budget=TokenBudget(
                reasoning_tokens=analyzer_reasoning,
                output_tokens=analyzer_total
                - analyzer_reasoning,  # Derive to avoid rounding
            ),
            percentage=35.0,
        )

        # Synthesizer: 25%, focus on output
        synthesizer_total = int(total_budget * 0.25)
        synthesizer_reasoning = int(synthesizer_total * 0.4)
        allocations[AgentRole.SYNTHESIZER] = RoleAllocation(
            role=AgentRole.SYNTHESIZER,
            budget=TokenBudget(
                reasoning_tokens=synthesizer_reasoning,
                output_tokens=synthesizer_total
                - synthesizer_reasoning,  # Derive to avoid rounding
            ),
            percentage=25.0,
        )

        return MultiAgentBudgetConfig(
            total_budget=total_budget,
            allocations=allocations,
            reserve_pool=0,
            awareness_condition=awareness_condition,
        )

    @staticmethod
    def create_with_negotiation(
        total_budget: int,
        initial_allocation_pct: float = 0.6,
    ) -> "MultiAgentBudgetConfig":
        """Create config with negotiation reserve pool.

        Args:
            total_budget: Total team budget
            initial_allocation_pct: Percentage to allocate initially (default: 60%)

        Returns:
            MultiAgentBudgetConfig with reserve pool for negotiation
        """
        initial_budget = int(total_budget * initial_allocation_pct)
        reserve = total_budget - initial_budget

        # Use equal split for initial allocation
        per_agent = initial_budget // 3
        remainder = initial_budget % 3  # Handle rounding
        roles = list(AgentRole)

        allocations = {}
        for i, role in enumerate(roles):
            # Give remainder to last agent
            agent_budget = per_agent + (remainder if i == len(roles) - 1 else 0)
            reasoning_tokens = agent_budget // 2

            allocations[role] = RoleAllocation(
                role=role,
                budget=TokenBudget(
                    reasoning_tokens=reasoning_tokens,
                    output_tokens=agent_budget
                    - reasoning_tokens,  # Derive to avoid rounding
                ),
                percentage=(agent_budget / total_budget) * 100,
            )

        return MultiAgentBudgetConfig(
            total_budget=total_budget,
            allocations=allocations,
            reserve_pool=reserve,
            awareness_condition=MultiAgentAwarenessCondition.WITH_NEGOTIATION,
        )
