"""Agent Budget Capstone - Token Allocation Framework.

This package provides tools for managing and analyzing token budget
allocations in AI agents, demonstrating tradeoffs between reasoning
depth and output verbosity.
"""

from .agent_factory import AgentFactory
from .core import (
    CODE_REVIEW_CODER_BUDGET,
    CODE_REVIEW_REVIEWER_BUDGET,
    CODE_REVIEW_TEAM_BUDGET,
    IterativeTeamConfig,
    TokenBudget,
)
from .loop_agents import CheckApprovalAgent
from .monitor import AgentMetrics, MultiAgentMetrics, ToolUsageMetrics, UsageMonitor
from .planner import PlannerEstimate, estimate_budget

__version__ = "0.1.0"

__all__ = [
    "TokenBudget",
    "IterativeTeamConfig",
    "CODE_REVIEW_CODER_BUDGET",
    "CODE_REVIEW_REVIEWER_BUDGET",
    "CODE_REVIEW_TEAM_BUDGET",
    "AgentFactory",
    "CheckApprovalAgent",
    "AgentMetrics",
    "MultiAgentMetrics",
    "ToolUsageMetrics",
    "UsageMonitor",
    # Part 2 Extension: Planner
    "PlannerEstimate",
    "estimate_budget",
]
