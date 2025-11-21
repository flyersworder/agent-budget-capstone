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
from google.adk.tools import FunctionTool, google_search
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
from .tracking_loop_agent import TrackingLoopAgent

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


# Wrap request_budget in FunctionTool for ADK compatibility
request_budget_tool = FunctionTool(func=request_budget)


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

    else:  # NEGOTIATION_AWARENESS
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

    # Persona
    persona = """<persona>
You are the RESEARCHER in a 2-agent iterative collaboration team.
Your role is to gather evidence and propose well-reasoned answers.
</persona>"""

    # Task description
    task = """<task>
1. Use google_search to find relevant information and evidence
2. Analyze the evidence and propose an answer with clear reasoning
3. If the Validator provides feedback, carefully consider it and revise your answer
4. Continue iterating until the Validator approves or maximum rounds are reached
</task>"""

    # Iteration awareness
    iteration = f"""<iteration_context>
This is an iterative process with up to {team_config.max_iterations} rounds.
- You can see the round number from the conversation history
- Round 1: Your first proposal
- Round 2+: Revisions based on Validator feedback
- The Validator will respond with either approval (starting with "APPROVED") or constructive feedback
</iteration_context>"""

    # Output format
    output_format = """<output_format>
Provide your response in this structure:

**Round 1 (Initial Proposal):**
"Based on [specific evidence from search], I propose the answer is [X] because [clear reasoning connecting evidence to answer]."

**Round 2+ (Revisions):**
"Based on the Validator's feedback about [specific concern], I have [searched for new evidence / reconsidered my reasoning]. I now propose [revised answer] because [updated reasoning]."

Always include:
- Specific evidence sources
- Clear logical reasoning
- Acknowledgment of Validator feedback (if any)
</output_format>"""

    # Budget awareness (varies by condition)
    if condition == MultiAgentAwarenessCondition.NO_AWARENESS:
        constraints = ""

    elif condition == MultiAgentAwarenessCondition.OVERALL_ONLY:
        constraints = f"""<constraints>
<budget_awareness>
Your 2-agent team has a TOTAL budget of {team_config.total_budget} tokens for this task.
This budget is shared across both agents (Researcher and Validator) and all {team_config.max_iterations} rounds.

Efficiency matters:
- Unnecessary iterations consume team budget
- Aim for accurate answers in fewer rounds when reasonable
</budget_awareness>
</constraints>"""

    elif condition == MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL:
        constraints = f"""<constraints>
<budget_awareness>
Team budget: {team_config.total_budget} tokens total for this task
Your allocation (Researcher): {budget.total} tokens per call (60% of team)
- {budget.reasoning_tokens} tokens available for thinking/reasoning per call
- {budget.output_tokens} tokens available for your response per call

Note: Each agent call (each round) has this same budget limit.
Be concise and efficient within each call to enable multiple rounds if needed.
</budget_awareness>
</constraints>"""

    else:  # RESERVE_AWARENESS
        constraints = f"""<constraints>
<budget_awareness>
Team budget structure:
- Total: {team_config.total_budget} tokens
- Allocated: {team_config.allocated_budget} tokens (distributed to agents)
- Reserve pool: {team_config.reserve_pool} tokens (held for complex questions)

Your current allocation (Researcher): {budget.total} tokens per call
- {budget.reasoning_tokens} tokens for thinking/reasoning per call
- {budget.output_tokens} tokens for your response per call

Note: The reserve pool provides flexibility for unexpectedly challenging tasks.
Focus on providing accurate answers efficiently within your allocation.
</budget_awareness>
</constraints>"""

    # Combine sections in best-practice order
    if constraints:
        return f"{persona}\n\n{constraints}\n\n{task}\n\n{iteration}\n\n{output_format}"
    else:
        return f"{persona}\n\n{task}\n\n{iteration}\n\n{output_format}"


def _create_validator_instruction(team_config: IterativeTeamConfig) -> str:
    """Create instruction for Validator in iterative team.

    Args:
        team_config: Iterative team budget configuration

    Returns:
        Instruction string for Validator
    """
    condition = team_config.awareness_condition
    budget = team_config.validator_budget

    # Persona
    persona = """<persona>
You are the VALIDATOR in a 2-agent iterative collaboration team.
Your role is to critically evaluate the Researcher's work and either approve or provide constructive feedback.
</persona>"""

    # Task description
    task = f"""<task>
Your role: Evaluate the quality of the Researcher's work based ONLY on what they provided.

1. Carefully review the Researcher's proposed answer and reasoning

2. CRITICAL CHECKPOINT - Ask yourself:
   ✓ Does the answer DIRECTLY address what the question asks for?
   ✓ Is the answer SPECIFIC and COMPLETE (not vague or hedged)?
   ✓ Did the Researcher find concrete evidence, or just reason about it?

3. Check for quality issues:
   - Logical flaws or contradictions
   - Claims without supporting evidence
   - Missing key information needed to answer the question
   - Gaps in reasoning chain
   - "Cannot determine" or "Not available" conclusions (be skeptical - different searches might help)

4. Decide whether to APPROVE or REQUEST REVISION

Decision criteria:
- APPROVE if the answer is:
  * DIRECTLY answers what the question asks (not a related but different answer)
  * SPECIFIC and concrete (not "cannot determine" unless truly impossible)
  * Well-supported by evidence Researcher found and cited
  * Addresses all parts of the question
  * Has clear reasoning connecting evidence to conclusion
  * Free of obvious logical errors

- REQUEST REVISION if:
  * Answer doesn't directly address the question asked
  * Answer is "cannot determine" but different search terms might help
  * Key information is missing (specify exactly what to search for)
  * Reasoning has gaps or flaws (specify exactly where)
  * Evidence doesn't support the conclusion
  * Answer seems plausible but lacks specific evidence

Common pitfalls to avoid:
⚠️  Approving logically sound reasoning when the actual answer is still missing
⚠️  Accepting "cannot determine" too quickly - suggest specific alternative searches
⚠️  Approving answers that explain WHY something can't be found instead of finding it

Note: You do NOT have search tools. You can only evaluate what the Researcher provided.
Balance thoroughness with efficiency (up to {team_config.max_iterations} rounds total).
</task>"""

    # Iteration awareness
    iteration = f"""<iteration_context>
This is an iterative process with up to {team_config.max_iterations} rounds.
- You can see the round number from the conversation history
- Round 1: Evaluate initial proposal
- Round 2+: Check if Researcher addressed your previous feedback
- End condition: You approve OR maximum rounds reached
</iteration_context>"""

    # Output format
    output_format = """<output_format>
**If APPROVING:**
"APPROVED. Final answer: [X] because [explain why the evidence supports this answer]."

CRITICAL: You MUST start with the exact word "APPROVED" to signal approval.

**If REQUESTING REVISION:**
Provide SPECIFIC, ACTIONABLE feedback using this template:

"REQUEST REVISION.

**Issue:** [State the specific problem - missing info, logical gap, or unsupported claim]

**What's needed:** [Be very specific about what the Researcher should do next]
Examples:
- "Search for [specific fact X] about [specific topic Y]"
- "Verify that [specific claim A] is consistent with [specific claim B]"
- "Clarify the connection between [evidence C] and [conclusion D]"

**Reason:** [Briefly explain why this matters for answering the question]"

Requirements for effective feedback:
✓ Name specific facts, claims, or search terms
✓ Give clear direction for next round (what to search, verify, or clarify)
✓ Avoid vague requests like "need more info" or "unclear reasoning"
✗ Do NOT search yourself - Researcher will handle searches in next round
</output_format>"""

    # Budget awareness (varies by condition)
    if condition == MultiAgentAwarenessCondition.NO_AWARENESS:
        constraints = ""

    elif condition == MultiAgentAwarenessCondition.OVERALL_ONLY:
        constraints = f"""<constraints>
<budget_awareness>
Your 2-agent team has a TOTAL budget of {team_config.total_budget} tokens for this task.
This budget is shared across both agents (Researcher and Validator) and all {team_config.max_iterations} rounds.

Efficiency considerations:
- Additional iterations consume team budget
- Approve when the answer is good enough, even if not perfect
- Balance quality with resource efficiency
</budget_awareness>
</constraints>"""

    elif condition == MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL:
        constraints = f"""<constraints>
<budget_awareness>
Team budget: {team_config.total_budget} tokens total for this task
Your allocation (Validator): {budget.total} tokens per call (40% of team)
- {budget.reasoning_tokens} tokens available for thinking/reasoning per call
- {budget.output_tokens} tokens available for your response per call

Note: Each agent call (each round) has this same budget limit.
Be thorough but concise to enable multiple rounds if needed.
Balance verification depth with efficiency.
</budget_awareness>
</constraints>"""

    else:  # RESERVE_AWARENESS
        constraints = f"""<constraints>
<budget_awareness>
Team budget structure:
- Total: {team_config.total_budget} tokens
- Allocated: {team_config.allocated_budget} tokens (distributed to agents)
- Reserve pool: {team_config.reserve_pool} tokens (held for complex questions)

Your current allocation (Validator): {budget.total} tokens per call
- {budget.reasoning_tokens} tokens for thinking/reasoning per call
- {budget.output_tokens} tokens for your response per call

Note: The reserve pool provides flexibility for unexpectedly challenging tasks.
Be thorough but efficient - approve when answers are sufficiently accurate.
</budget_awareness>
</constraints>"""

    # Combine sections in best-practice order
    if constraints:
        return f"{persona}\n\n{constraints}\n\n{task}\n\n{iteration}\n\n{output_format}"
    else:
        return f"{persona}\n\n{task}\n\n{iteration}\n\n{output_format}"


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

        # NEGOTIATION_AWARENESS uses instruction-based negotiation (no tool needed)
        # All conditions use the same tools to avoid API conflicts with thinking mode

        # Create 3 role-based agents
        agents = []

        for role in [AgentRole.RESEARCHER, AgentRole.ANALYZER, AgentRole.SYNTHESIZER]:
            role_alloc = team_config.allocations[role]
            instruction = _create_multiagent_instruction(role, team_config)

            # Only Researcher gets tools (google_search)
            # Others don't need tools - they work with text from previous agents
            agent_tools = tools if role == AgentRole.RESEARCHER else []

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

        # NEGOTIATION_AWARENESS uses instruction-based negotiation (no tool needed)
        # All conditions use the same tools to avoid API conflicts with thinking mode
        agent_tools = tools

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
        # NOTE: Validator does NOT get search tools - only evaluates Researcher's work
        validator_instruction = _create_validator_instruction(team_config)
        validator = LlmAgent(
            model=self.model,
            name="Validator",
            instruction=validator_instruction,
            description="Validates research and provides feedback or approval",
            tools=[],  # No tools - validator only evaluates, doesn't search
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
        # Also reports token usage with rich context for awareness conditions
        report_usage = (
            team_config.awareness_condition != MultiAgentAwarenessCondition.NO_AWARENESS
        )
        check_approval = CheckApprovalAgent(
            name="CheckApproval",
            description="Checks if Validator approved and exits loop",
            report_usage=report_usage,
            awareness_condition=team_config.awareness_condition,
            researcher_budget_total=team_config.researcher_budget.total,
            validator_budget_total=team_config.validator_budget.total,
            team_budget_total=team_config.total_budget,
        )

        # Create TrackingLoopAgent
        # Loop runs: Researcher → Validator → CheckApproval
        # Tracks token usage internally before passing to next agent
        # Exits when CheckApproval escalates (approval found) or max_iterations reached
        return TrackingLoopAgent(
            name=f"iterative_team_{team_config.awareness_condition.value}",
            description="Iterative 2-agent team with Researcher ⇄ Validator loop",
            sub_agents=[researcher, validator, check_approval],
            max_iterations=team_config.max_iterations,
        )
