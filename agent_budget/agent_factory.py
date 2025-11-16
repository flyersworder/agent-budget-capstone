"""Factory for creating agents with different token allocations.

This module provides a factory class for creating Google ADK agents
with specific token budget allocations.
"""

import os
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search
from google.genai import types

from .core import AllocationStrategy

# Load environment variables
load_dotenv()


class AgentFactory:
    """Factory for creating agents with different token allocations.

    This factory creates Google ADK agents configured with specific
    reasoning and output token budgets based on allocation strategies.

    Attributes:
        model: The Gemini model to use (default: gemini-2.5-flash-lite)
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        """Initialize the agent factory.

        Args:
            model: Gemini model identifier
        """
        self.model = model

    def create_agent(
        self,
        strategy: AllocationStrategy,
        total_budget: int = 10000,
        tools: list[Any] | None = None,
    ) -> Agent:
        """Create an agent with specific budget allocation.

        Args:
            strategy: Allocation strategy to use
            total_budget: Total token budget (default: 10000)
            tools: List of tools to provide to agent (default: [google_search])

        Returns:
            Configured Agent instance

        Raises:
            ValueError: If total_budget is negative
        """
        if total_budget < 0:
            raise ValueError("Total budget must be non-negative")

        budget = strategy.create_budget(total_budget)

        if tools is None:
            tools = [google_search]  # Default to built-in Google Search

        # Strategy-specific instructions
        instructions = {
            "deep": (
                "You are a research assistant that thinks deeply before acting. "
                "Carefully plan your tool usage. Use tools only when necessary. "
                "Provide concise, insightful responses based on thorough analysis."
            ),
            "balanced": (
                "You are a research assistant with balanced thinking and output. "
                "Use tools when helpful to gather information. "
                "Provide clear, well-reasoned responses with appropriate detail."
            ),
            "verbose": (
                "You are a research assistant that provides detailed explanations. "
                "Use tools to gather comprehensive information. "
                "Provide thorough, detailed responses with extensive context."
            ),
        }

        # Verify API key is in environment
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )

        return Agent(
            model=self.model,
            name=f"{strategy.value}_agent",
            instruction=instructions[strategy.value],
            description=f"Agent with {strategy.value} token allocation strategy",
            tools=tools,
            planner=BuiltInPlanner(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=budget.reasoning_tokens,
                    include_thoughts=True,  # Enable thought tracking for analysis
                )
            ),
            generate_content_config=types.GenerateContentConfig(
                max_output_tokens=budget.output_tokens,
                temperature=0.0,  # Deterministic for fair comparison
            ),
        )

    def create_all_agents(
        self, total_budget: int = 10000, tools: list[Any] | None = None
    ) -> dict[str, Agent]:
        """Create agents for all allocation strategies.

        Args:
            total_budget: Total token budget for each agent
            tools: List of tools to provide to agents

        Returns:
            Dictionary mapping strategy name to Agent instance
        """
        return {
            strategy.value: self.create_agent(strategy, total_budget, tools)
            for strategy in AllocationStrategy
        }

    def get_budget_info(
        self, strategy: AllocationStrategy, total_budget: int = 10000
    ) -> dict[str, Any]:
        """Get budget allocation details for a strategy.

        Args:
            strategy: Allocation strategy
            total_budget: Total token budget

        Returns:
            Dictionary with budget breakdown
        """
        budget = strategy.create_budget(total_budget)

        return {
            "strategy": strategy.value,
            "total_budget": total_budget,
            "reasoning_tokens": budget.reasoning_tokens,
            "output_tokens": budget.output_tokens,
            "reasoning_percentage": (budget.reasoning_tokens / total_budget) * 100,
            "output_percentage": (budget.output_tokens / total_budget) * 100,
        }
