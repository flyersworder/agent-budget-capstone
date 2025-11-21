"""Factory for creating agents with budget configurations.

This module provides a unified factory for creating:
- Part 1: Single agents with budget awareness
- Part 2: Multi-agent teams with coordination

Supports all awareness conditions and budget configurations.
"""

import os
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.planners import BuiltInPlanner
from google.adk.tools import google_search
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .awareness import AwarenessCondition
from .core import (
    AgentRole,
    IterativeTeamConfig,
    MultiAgentAwarenessCondition,
    MultiAgentBudgetConfig,
    TokenBudget,
)
from .loop_agents import CheckApprovalAgent

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
# PART 2 ITERATIVE: Instruction Generators
# ============================================================================


def _create_researcher_instruction(team_config: IterativeTeamConfig) -> str:
    """Create instruction for Researcher in iterative team.

    Args:
        team_config: Iterative team budget configuration

    Returns:
        Instruction string for Researcher
    """
    condition = team_config.awareness_condition
    budget = team_config.researcher_budget

    base = """<role>
You are the RESEARCHER in a 2-agent iterative team.

Your task:
1. Use google_search to gather information and evidence
2. Propose an answer with clear reasoning
3. If you receive feedback from the Validator, revise your answer accordingly

Output format:
- Round 1: "Based on [evidence], I propose the answer is [X] because [reasoning]"
- Round N: "Based on feedback, I revise my answer to [Y] because [new reasoning]"

Important: Provide detailed reasoning so the Validator can verify your work.
</role>"""

    # Add budget awareness
    if condition == MultiAgentAwarenessCondition.NO_AWARENESS:
        return base

    elif condition == MultiAgentAwarenessCondition.OVERALL_ONLY:
        return f"""<team_budget>
Your 2-agent team has {team_config.total_budget} tokens TOTAL across up to {team_config.max_iterations} rounds.
Use resources efficiently - unnecessary iterations waste budget.
</team_budget>

{base}"""

    elif condition == MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL:
        return f"""<team_budget>
Your team has {team_config.total_budget} tokens total across up to {team_config.max_iterations} rounds.
</team_budget>

<your_allocation>
As RESEARCHER, your allocation across ALL rounds:
- {budget.reasoning_tokens} tokens for thinking/reasoning
- {budget.output_tokens} tokens for your output
- {budget.total} tokens total (60% of team budget)

Manage this budget across up to {team_config.max_iterations} iterations.
Each round uses part of your total allocation.
</your_allocation>

{base}"""

    else:  # WITH_NEGOTIATION
        return f"""<team_budget>
Your team has {team_config.total_budget} tokens total.
Initial allocation: {team_config.allocated_budget} tokens
Reserve pool: {team_config.reserve_pool} tokens (available for requests)
</team_budget>

<your_allocation>
Your initial allocation as RESEARCHER:
- {budget.reasoning_tokens} tokens for thinking/reasoning
- {budget.output_tokens} tokens for your output
- {budget.total} tokens total

If you need MORE budget to address Validator feedback, call:
request_budget(amount, justification)

Provide clear justification (e.g., "Need to search additional sources to verify X").
</your_allocation>

{base}"""


def _create_validator_instruction(team_config: IterativeTeamConfig) -> str:
    """Create instruction for Validator in iterative team.

    Args:
        team_config: Iterative team budget configuration

    Returns:
        Instruction string for Validator
    """
    condition = team_config.awareness_condition
    budget = team_config.validator_budget

    base = f"""<role>
You are the VALIDATOR in a 2-agent iterative team.

Your task:
1. Critically evaluate the Researcher's proposal and reasoning
2. You can independently verify facts using google_search if needed
3. Either APPROVE the answer or provide constructive feedback

Output format options:
- APPROVE: "APPROVED. Final answer: [X] because [validation reasoning]"
- Request clarification: "Please verify: Did you check [Y]?"
- Identify error: "Your reasoning has a flaw: [Z]. Please revise."
- Request more info: "You're missing information about [W]."

CRITICAL: If you approve, you MUST start your response with "APPROVED" (exact word).
This signals the system to finalize the answer.

Up to {team_config.max_iterations} rounds available. Approve when answer is good enough.
</role>"""

    # Add budget awareness
    if condition == MultiAgentAwarenessCondition.NO_AWARENESS:
        return base

    elif condition == MultiAgentAwarenessCondition.OVERALL_ONLY:
        return f"""<team_budget>
Your 2-agent team has {team_config.total_budget} tokens TOTAL across up to {team_config.max_iterations} rounds.
Use resources efficiently - approve when answer is good enough.
</team_budget>

{base}"""

    elif condition == MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL:
        return f"""<team_budget>
Your team has {team_config.total_budget} tokens total across up to {team_config.max_iterations} rounds.
</team_budget>

<your_allocation>
As VALIDATOR, your allocation across ALL rounds:
- {budget.reasoning_tokens} tokens for thinking/reasoning
- {budget.output_tokens} tokens for your output
- {budget.total} tokens total (40% of team budget)

Manage this budget across up to {team_config.max_iterations} iterations.
Balance thoroughness with efficiency - approve when answer is good enough.
</your_allocation>

{base}"""

    else:  # WITH_NEGOTIATION
        return f"""<team_budget>
Your team has {team_config.total_budget} tokens total.
Initial allocation: {team_config.allocated_budget} tokens
Reserve pool: {team_config.reserve_pool} tokens (available for requests)
</team_budget>

<your_allocation>
Your initial allocation as VALIDATOR:
- {budget.reasoning_tokens} tokens for thinking/reasoning
- {budget.output_tokens} tokens for your output
- {budget.total} tokens total

If you need MORE budget for thorough verification, call:
request_budget(amount, justification)

Provide clear justification (e.g., "Need to independently verify claim X").
</your_allocation>

{base}"""


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
    # PART 2 ITERATIVE: 2-Agent Iterative Team Creation
    # ========================================================================

    def create_iterative_team(
        self,
        team_config: IterativeTeamConfig,
        tools: list[Any] | None = None,
    ) -> LoopAgent:
        """Create iterative 2-agent team for Part 2 coordination study.

        Creates a LoopAgent with iterative Researcher ⇄ Validator workflow:
        1. Researcher proposes answer with reasoning
        2. Validator evaluates and either approves or provides feedback
        3. CheckApproval agent checks for approval and exits loop if found
        4. Loop repeats up to max_iterations times

        Args:
            team_config: Iterative team budget configuration
            tools: Tools for agents (default: [google_search])

        Returns:
            LoopAgent with configured Researcher, Validator, and CheckApproval agents
        """
        if tools is None:
            tools = [google_search]

        # Add negotiation tool for Condition D
        agent_tools = tools
        if (
            team_config.awareness_condition
            == MultiAgentAwarenessCondition.WITH_NEGOTIATION
        ):
            agent_tools = tools + [request_budget]

        # Create Researcher agent
        researcher_instruction = _create_researcher_instruction(team_config)
        researcher = LlmAgent(
            model=self.model,
            name="Researcher",
            instruction=researcher_instruction,
            description="Gathers information and proposes answers",
            tools=agent_tools,
            planner=BuiltInPlanner(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=team_config.researcher_budget.reasoning_tokens,
                    include_thoughts=True,
                )
            ),
            generate_content_config=types.GenerateContentConfig(
                # IMPORTANT: Per-call budget, not cumulative
                # Instruction tells agent to manage across iterations
                max_output_tokens=team_config.researcher_budget.total,
                temperature=0.2,
            ),
            output_key="researcher_output",  # Saves to state['researcher_output']
        )

        # Create Validator agent
        validator_instruction = _create_validator_instruction(team_config)
        validator = LlmAgent(
            model=self.model,
            name="Validator",
            instruction=validator_instruction,
            description="Validates research and provides feedback or approval",
            tools=agent_tools,  # Can also search to verify
            planner=BuiltInPlanner(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=team_config.validator_budget.reasoning_tokens,
                    include_thoughts=True,
                )
            ),
            generate_content_config=types.GenerateContentConfig(
                max_output_tokens=team_config.validator_budget.total,
                temperature=0.2,
            ),
            output_key="validator_feedback",  # Saves to state['validator_feedback']
        )

        # Create CheckApproval agent
        # This agent checks if validator_feedback contains "APPROVED"
        # If yes, escalates to exit the loop
        check_approval = CheckApprovalAgent(
            name="CheckApproval",
            description="Checks if Validator approved and exits loop",
        )

        # Create LoopAgent
        # Loop runs: Researcher → Validator → CheckApproval
        # Exits when CheckApproval escalates (approval found) or max_iterations reached
        return LoopAgent(
            name=f"iterative_team_{team_config.awareness_condition.value}",
            description="Iterative 2-agent team with Researcher ⇄ Validator loop",
            sub_agents=[researcher, validator, check_approval],
            max_iterations=team_config.max_iterations,
        )
