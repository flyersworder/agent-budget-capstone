"""Agent Budget Capstone - Token Allocation Framework.

This package provides tools for managing and analyzing token budget
allocations in AI agents, demonstrating tradeoffs between reasoning
depth and output verbosity.
"""

from .agent_factory import AgentFactory
from .core import TokenBudget
from .monitor import AgentMetrics, ToolUsageMetrics, UsageMonitor

__version__ = "0.1.0"

__all__ = [
    "TokenBudget",
    "AgentFactory",
    "AgentMetrics",
    "ToolUsageMetrics",
    "UsageMonitor",
]
