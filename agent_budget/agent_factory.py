"""Factory for creating agents with budget configurations.

This module provides a unified factory for creating:
- Part 1: Single agents with budget awareness
- Part 2: Multi-agent teams with coordination

Supports all awareness conditions and budget configurations.
"""

import os
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .awareness import AwarenessCondition
from .core import (
    AgentRole,
    AllocationStrategy,
    MultiAgentAwarenessCondition,
    MultiAgentBudgetConfig,
    TokenBudget,
)

# Load environment variables
load_dotenv()


# ============================================================================
# Part 2: Negotiation Tool
# ============================================================================


def request_budget(
    amount: int, justification: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Request additional budget from shared pool (Part 2, Condition D only).

    Args:
        amount: Additional tokens requested
        justification: Reason for additional budget

    Returns:
        Dictionary with approval status and allocated amount
    """
    pool = tool_context.state.get("budget_pool", 0)

    # Simple rule-based approval:
    # Approve if pool has funds AND justification mentions complexity/accuracy
    keywords = ["complex", "difficult", "challenging", "accuracy", "thorough"]
    is_justified = any(word in justification.lower() for word in keywords)

    if pool >= amount and is_justified:
        # Approve request
        tool_context.state["budget_pool"] = pool - amount

        # Log the request
        if "budget_requests" not in tool_context.state:
            tool_context.state["budget_requests"] = []
        tool_context.state["budget_requests"].append(
            {
                "amount": amount,
                "justification": justification,
                "approved": True,
                "allocated": amount,
            }
        )

        return {
            "approved": True,
            "allocated": amount,
            "remaining_pool": pool - amount,
        }
    else:
        # Reject request
        reason = (
            "Insufficient budget in pool"
            if pool < amount
            else "Justification does not meet criteria"
        )

        if "budget_requests" not in tool_context.state:
            tool_context.state["budget_requests"] = []
        tool_context.state["budget_requests"].append(
            {
                "amount": amount,
                "justification": justification,
                "approved": False,
                "allocated": 0,
            }
        )

        return {"approved": False, "allocated": 0, "reason": reason}


# ============================================================================
# Instruction Generators
# ============================================================================


def _create_part1_instruction(
    condition: AwarenessCondition, budget_config: TokenBudget
) -> str:
    """Create instruction for Part 1 single agent.

    Args:
        condition: Budget awareness condition
        budget_config: Token budget configuration

    Returns:
        Instruction string
    """
    if condition == AwarenessCondition.UNAWARE:
        return (
            "You are a helpful research assistant that answers questions "
            "accurately and concisely. Use the google_search tool to find "
            "factual information when needed. Think carefully before responding."
        )
    else:  # AWARE
        return f"""<budget>
You have a computational budget of:
- {budget_config.reasoning_tokens} tokens for internal thinking/reasoning
- {budget_config.total} tokens total (including your response)

These are HARD LIMITS enforced by the system.
</budget>

<strategy>
Given your budget constraints:
1. Use your thinking budget efficiently - plan before searching
2. Make strategic tool calls - search for the most critical information
3. Provide a concise, accurate response within your output budget
</strategy>

<task>
Answer the question accurately and concisely using available tools.
</task>"""


def _create_multiagent_instruction(
    role: AgentRole,
    team_config: MultiAgentBudgetConfig,
) -> str:
    """Create instruction for multi-agent team member.

    Args:
        role: Agent role in the team
        team_config: Multi-agent budget configuration

    Returns:
        Instruction string
    """
    # Role-specific base instructions
    role_instructions = {
        AgentRole.RESEARCHER: (
            "You are the RESEARCHER in a 3-agent team. Your role is to gather "
            "relevant information using the google_search tool. Find factual "
            "evidence to answer the question. Pass your findings to the next agent."
        ),
        AgentRole.ANALYZER: (
            "You are the ANALYZER in a 3-agent team. Your role is to critically "
            "evaluate the information gathered by the researcher. Check for "
            "accuracy, identify contradictions, and determine the correct answer. "
            "Pass your analysis to the next agent."
        ),
        AgentRole.SYNTHESIZER: (
            "You are the SYNTHESIZER in a 3-agent team. Your role is to produce "
            "a final, concise answer based on the research and analysis from "
            "previous agents. Provide a clear, accurate response."
        ),
    }

    base_instruction = role_instructions[role]
    condition = team_config.awareness_condition
    role_alloc = team_config.allocations[role]

    # Add budget awareness based on condition
    if condition == MultiAgentAwarenessCondition.NO_AWARENESS:
        # No budget information
        return base_instruction

    elif condition == MultiAgentAwarenessCondition.OVERALL_ONLY:
        # Only total team budget
        return f"""<team_budget>
Your team has a total budget of {team_config.total_budget} tokens across all 3 agents.
Use resources wisely to ensure the team completes the task within budget.
</team_budget>

<role>
{base_instruction}
</role>"""

    elif condition == MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL:
        # Total + individual allocation
        return f"""<team_budget>
Your team has a total budget of {team_config.total_budget} tokens across all 3 agents.
</team_budget>

<your_allocation>
As the {role.value.upper()}, your individual allocation is:
- {role_alloc.budget.reasoning_tokens} tokens for thinking/reasoning
- {role_alloc.budget.output_tokens} tokens for your output
- {role_alloc.budget.total} tokens total ({role_alloc.percentage:.0f}% of team budget)

Optimize your work within YOUR allocation.
</your_allocation>

<role>
{base_instruction}
</role>"""

    else:  # WITH_NEGOTIATION
        # Can request additional budget
        return f"""<team_budget>
Your team has a total budget of {team_config.total_budget} tokens.
Initial allocation: {team_config.allocated_budget} tokens (distributed)
Reserve pool: {team_config.reserve_pool} tokens (available for requests)
</team_budget>

<your_allocation>
Your initial allocation as {role.value.upper()}:
- {role_alloc.budget.reasoning_tokens} tokens for thinking/reasoning
- {role_alloc.budget.output_tokens} tokens for your output
- {role_alloc.budget.total} tokens total

If you need MORE budget for this task, you can call request_budget(amount, justification).
Provide a clear justification explaining why additional budget is needed.
</your_allocation>

<role>
{base_instruction}
</role>"""


# ============================================================================
# Agent Factory
# ============================================================================


class AgentFactory:
    """Factory for creating agents with budget configurations.

    Supports:
    - Part 1: Single agents with budget awareness
    - Part 2: Multi-agent teams with coordination
    - Legacy: Allocation strategies (archived Part 1)
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        """Initialize the agent factory.

        Args:
            model: Gemini model identifier
        """
        self.model = model

        # Verify API key
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )

    # ========================================================================
    # PART 1: Single Agent Creation
    # ========================================================================

    def create_single_agent(
        self,
        condition: AwarenessCondition,
        budget_config: TokenBudget,
        tools: list[Any] | None = None,
    ) -> Agent:
        """Create single agent for Part 1 budget awareness study.

        Args:
            condition: Budget awareness condition (aware/unaware)
            budget_config: Token budget configuration
            tools: List of tools (default: [google_search])

        Returns:
            Configured Agent instance
        """
        if tools is None:
            tools = [google_search]

        instruction = _create_part1_instruction(condition, budget_config)

        # NOTE: max_output_tokens must accommodate both thinking AND output
        # See Part 1 findings for critical bug fix
        return Agent(
            model=self.model,
            name=f"{condition.value}_agent",
            instruction=instruction,
            description=f"Agent with {condition.value} budget awareness",
            tools=tools,
            planner=BuiltInPlanner(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=budget_config.reasoning_tokens,
                    include_thoughts=True,
                )
            ),
            generate_content_config=types.GenerateContentConfig(
                max_output_tokens=budget_config.total,  # CRITICAL: total, not just output!
                temperature=0.2,  # Slightly stochastic for natural variation
            ),
        )

    # ========================================================================
    # PART 2: Multi-Agent Team Creation
    # ========================================================================

    def create_multiagent_team(
        self,
        team_config: MultiAgentBudgetConfig,
        tools: list[Any] | None = None,
    ) -> SequentialAgent:
        """Create multi-agent team for Part 2 coordination study.

        Creates a SequentialAgent with 3 sub-agents:
        Researcher → Analyzer → Synthesizer

        Args:
            team_config: Multi-agent budget configuration
            tools: Tools for researcher (default: [google_search])

        Returns:
            SequentialAgent with configured sub-agents
        """
        if tools is None:
            tools = [google_search]

        # Add negotiation tool for condition D
        if (
            team_config.awareness_condition
            == MultiAgentAwarenessCondition.WITH_NEGOTIATION
        ):
            tools = tools + [request_budget]

        # Create 3 role-based agents
        agents = []

        for role in [AgentRole.RESEARCHER, AgentRole.ANALYZER, AgentRole.SYNTHESIZER]:
            role_alloc = team_config.allocations[role]
            instruction = _create_multiagent_instruction(role, team_config)

            # Researcher gets tools, others don't need them
            agent_tools = (
                tools
                if role == AgentRole.RESEARCHER
                else (
                    [request_budget]
                    if team_config.awareness_condition
                    == MultiAgentAwarenessCondition.WITH_NEGOTIATION
                    else []
                )
            )

            agent = LlmAgent(
                model=self.model,
                name=role.value,
                instruction=instruction,
                description=f"{role.value.capitalize()} agent in sequential team",
                tools=agent_tools,
                planner=BuiltInPlanner(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=role_alloc.budget.reasoning_tokens,
                        include_thoughts=True,
                    )
                ),
                generate_content_config=types.GenerateContentConfig(
                    max_output_tokens=role_alloc.budget.total,  # Total for this agent
                    temperature=0.2,
                ),
                output_key=f"{role.value}_output",  # Save to state for next agent
            )

            agents.append(agent)

        # Create sequential team
        return SequentialAgent(
            name=f"team_{team_config.awareness_condition.value}",
            description="Multi-agent team for Part 2 budget coordination study",
            sub_agents=agents,
        )

    # ========================================================================
    # LEGACY: Allocation Strategy Support (Archived Part 1)
    # ========================================================================

    def create_agent(
        self,
        strategy: AllocationStrategy,
        total_budget: int = 10000,
        tools: list[Any] | None = None,
    ) -> Agent:
        """Create agent with allocation strategy (LEGACY - Archived Part 1).

        This method is kept for backward compatibility with archived Part 1 code.
        For current research, use create_single_agent() or create_multiagent_team().

        Args:
            strategy: Allocation strategy to use
            total_budget: Total token budget (default: 10000)
            tools: List of tools to provide to agent (default: [google_search])

        Returns:
            Configured Agent instance
        """
        if total_budget < 0:
            raise ValueError("Total budget must be non-negative")

        budget = strategy.create_budget(total_budget)

        if tools is None:
            tools = [google_search]

        # Strategy-specific instructions with budget awareness
        instructions = {
            "deep": (
                f"BUDGET: You have {budget.reasoning_tokens} tokens for internal thinking "
                f"and {budget.output_tokens} tokens for your final response.\n\n"
                "STRATEGY: You are a 'Deep Thinker' agent. Use your large reasoning budget "
                "to thoroughly analyze the problem, explore multiple angles, and plan your "
                "approach carefully. Think deeply before acting. Use tools strategically "
                "when they add value. Then provide a concise, well-reasoned response that "
                "distills your analysis into key insights. Focus on quality over quantity "
                "in your output."
            ),
            "balanced": (
                f"BUDGET: You have {budget.reasoning_tokens} tokens for internal thinking "
                f"and {budget.output_tokens} tokens for your final response.\n\n"
                "STRATEGY: You are a 'Balanced' agent. You have equal budgets for thinking "
                "and output. Allocate your resources proportionally to the task's needs. "
                "Use reasoning to structure your approach and plan tool usage. Then provide "
                "clear, appropriately detailed responses that balance depth and clarity."
            ),
            "verbose": (
                f"BUDGET: You have {budget.reasoning_tokens} tokens for internal thinking "
                f"and {budget.output_tokens} tokens for your final response.\n\n"
                "STRATEGY: You are a 'Verbose' agent. You have a large output budget to "
                "provide comprehensive, detailed explanations. Your thinking budget is more "
                "limited, so be efficient in your reasoning. Use tools to gather comprehensive "
                "information, then focus on thorough, well-explained responses with extensive "
                "context, examples, and supporting details."
            ),
        }

        return Agent(
            model=self.model,
            name=f"{strategy.value}_agent",
            instruction=instructions[strategy.value],
            description=f"Agent with {strategy.value} token allocation strategy",
            tools=tools,
            planner=BuiltInPlanner(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=budget.reasoning_tokens,
                    include_thoughts=True,
                )
            ),
            generate_content_config=types.GenerateContentConfig(
                max_output_tokens=budget.output_tokens,
                temperature=0.0,
            ),
        )

    def create_all_agents(
        self, total_budget: int = 10000, tools: list[Any] | None = None
    ) -> dict[str, Agent]:
        """Create agents for all allocation strategies (LEGACY).

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

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_budget_info(
        self, strategy: AllocationStrategy, total_budget: int = 10000
    ) -> dict[str, Any]:
        """Get budget allocation details for a strategy (LEGACY).

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
