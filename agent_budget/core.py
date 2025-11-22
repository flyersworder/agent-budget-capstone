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


# ============================================================================
# PART 2: Multi-Agent Configurations
# ============================================================================


class AgentRole(Enum):
    """Roles in multi-agent teams (Part 2).

    Each role has specific responsibilities in the workflow.
    """

    # Part 2 Original (Research workflow)
    RESEARCHER = "researcher"  # Gathers information using tools
    ANALYZER = "analyzer"  # Evaluates information for correctness
    SYNTHESIZER = "synthesizer"  # Produces final answer

    # Part 2 Redesign (Code review workflow)
    CODER = "coder"  # Writes/revises code
    REVIEWER = "reviewer"  # Tests and reviews code


class MultiAgentAwarenessCondition(Enum):
    """Budget awareness conditions for multi-agent teams (Part 2)."""

    NO_AWARENESS = "no_awareness"  # Baseline: no budget information
    OVERALL_ONLY = "overall_only"  # Know total team budget only
    OVERALL_AND_INDIVIDUAL = "overall_and_individual"  # Know total + own allocation
    RESERVE_AWARENESS = (
        "reserve_awareness"  # Know total + own allocation + reserve pool
    )


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
    def create_reserve_awareness(
        total_budget: int,
        initial_allocation_pct: float = 0.6,
    ) -> "MultiAgentBudgetConfig":
        """Create config with reserve pool awareness.

        Args:
            total_budget: Total team budget
            initial_allocation_pct: Percentage to allocate initially (default: 60%)

        Returns:
            MultiAgentBudgetConfig with reserve pool
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
            awareness_condition=MultiAgentAwarenessCondition.RESERVE_AWARENESS,
        )


# ============================================================================
# PART 2: Iterative 2-Agent Team Configuration
# ============================================================================


@dataclass
class IterativeTeamConfig:
    """Budget configuration for iterative 2-agent team (Researcher ⇄ Validator).

    Attributes:
        total_budget: Total team budget across all iterations
        researcher_budget: Budget for Researcher agent
        validator_budget: Budget for Validator agent
        max_iterations: Maximum number of refinement rounds (default: 3)
        reserve_pool: Tokens held in reserve for negotiation (Condition D only)
        awareness_condition: How budget info is communicated to agents
    """

    total_budget: int
    researcher_budget: TokenBudget
    validator_budget: TokenBudget
    max_iterations: int = 3
    reserve_pool: int = 0
    awareness_condition: MultiAgentAwarenessCondition = (
        MultiAgentAwarenessCondition.NO_AWARENESS
    )

    @property
    def allocated_budget(self) -> int:
        """Total allocated to both agents."""
        return self.researcher_budget.total + self.validator_budget.total

    def validate(self) -> bool:
        """Ensure configuration is valid.

        Returns:
            True if valid, False otherwise
        """
        # Total allocated + reserve should equal total budget
        total_used = self.allocated_budget + self.reserve_pool
        return total_used == self.total_budget

    @staticmethod
    def create_standard(
        total_budget: int = 2000,
        awareness_condition: MultiAgentAwarenessCondition = MultiAgentAwarenessCondition.NO_AWARENESS,
    ) -> "IterativeTeamConfig":
        """Create standard 60/40 split config.

        Researcher gets 60% (more tool use), Validator gets 40%.

        NOTE: Minimum 2000 tokens required to meet Gemini's 512 token
        thinking budget minimum for both agents.

        Args:
            total_budget: Total team budget (default: 2000)
            awareness_condition: How to communicate budget

        Returns:
            IterativeTeamConfig with 60/40 allocation
        """
        researcher_total = int(total_budget * 0.60)  # 1200
        validator_total = int(total_budget * 0.40)  # 800

        # Researcher: 60% reasoning, 40% output (more tool-heavy)
        # 720 reasoning, 480 output
        researcher_reasoning = int(researcher_total * 0.60)
        researcher_budget = TokenBudget(
            reasoning_tokens=researcher_reasoning,
            output_tokens=researcher_total - researcher_reasoning,
        )

        # Validator: 65% reasoning, 35% output (ensure 512+ thinking budget)
        # 520 reasoning, 280 output
        validator_reasoning = int(validator_total * 0.65)
        validator_budget = TokenBudget(
            reasoning_tokens=validator_reasoning,
            output_tokens=validator_total - validator_reasoning,
        )

        return IterativeTeamConfig(
            total_budget=total_budget,
            researcher_budget=researcher_budget,
            validator_budget=validator_budget,
            max_iterations=3,
            reserve_pool=0,
            awareness_condition=awareness_condition,
        )

    @staticmethod
    def create_reserve_awareness(
        total_budget: int = 2000,
        initial_allocation_pct: float = 0.80,
    ) -> "IterativeTeamConfig":
        """Create config with reserve pool awareness.

        Args:
            total_budget: Total team budget (default: 2000)
            initial_allocation_pct: Percentage to allocate initially (default: 80%)

        Returns:
            IterativeTeamConfig with reserve pool
        """
        initial_budget = int(total_budget * initial_allocation_pct)  # 1600
        reserve = total_budget - initial_budget  # 400

        # 60/40 split of initial allocation
        researcher_total = int(initial_budget * 0.60)  # 960
        validator_total = int(initial_budget * 0.40)  # 640

        # Researcher: 60% reasoning
        researcher_reasoning = int(researcher_total * 0.60)  # 576
        researcher_budget = TokenBudget(
            reasoning_tokens=researcher_reasoning,
            output_tokens=researcher_total - researcher_reasoning,
        )

        # Validator: 80% reasoning to ensure 512+ (640 * 0.80 = 512)
        validator_reasoning = int(validator_total * 0.80)  # 512
        validator_budget = TokenBudget(
            reasoning_tokens=validator_reasoning,
            output_tokens=validator_total - validator_reasoning,
        )

        return IterativeTeamConfig(
            total_budget=total_budget,
            researcher_budget=researcher_budget,
            validator_budget=validator_budget,
            max_iterations=3,
            reserve_pool=reserve,
            awareness_condition=MultiAgentAwarenessCondition.RESERVE_AWARENESS,
        )
