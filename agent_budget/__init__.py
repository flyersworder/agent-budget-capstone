"""Agent Budget Capstone - Token Allocation Framework.

This package provides tools for managing and analyzing token budget
allocations in AI agents, demonstrating tradeoffs between reasoning
depth and output verbosity.
"""

from .agent_factory import AgentFactory
from .core import IterativeTeamConfig, TokenBudget
from .loop_agents import CheckApprovalAgent
from .monitor import AgentMetrics, MultiAgentMetrics, ToolUsageMetrics, UsageMonitor

__version__ = "0.1.0"

__all__ = [
    "TokenBudget",
    "IterativeTeamConfig",
    "AgentFactory",
    "CheckApprovalAgent",
    "AgentMetrics",
    "MultiAgentMetrics",
    "ToolUsageMetrics",
    "UsageMonitor",
]
