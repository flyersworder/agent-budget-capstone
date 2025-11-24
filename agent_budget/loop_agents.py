"""Custom agents for iterative loop patterns.

This module provides specialized agents for controlling loop execution
in multi-agent iterative workflows.
"""

from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from agent_budget.core import MultiAgentAwarenessCondition


class CheckApprovalAgent(BaseAgent):  # type: ignore[misc]
    """Custom agent that checks for approval and escalates to exit loop.

    This agent examines the validator's feedback in the session state.
    If the feedback contains "APPROVED", it signals the LoopAgent to
    terminate by escalating.

    Also reports cumulative token usage for budget awareness with:
    - Iteration context (e.g., "Iteration 2 of 3 complete")
    - Actionable guidance (what to focus on next)
    - Challenge framing (emphasize capability, not threat)

    Always checks state['validator_feedback'] for "APPROVED" keyword.
    """

    def __init__(
        self,
        name: str,
        description: str,
        report_usage: bool = True,
        awareness_condition: MultiAgentAwarenessCondition = MultiAgentAwarenessCondition.NO_AWARENESS,
        researcher_budget_total: int = 0,
        validator_budget_total: int = 0,
        team_budget_total: int = 0,
        agent1_name: str = "Researcher",
        agent2_name: str = "Validator",
        approval_state_key: str = "validator_feedback",
        approval_keyword: str = "APPROVED",
        max_iterations: int = 3,
    ):
        """Initialize CheckApprovalAgent.

        Args:
            name: Agent name
            description: Agent description
            report_usage: Whether to report token usage (default: True)
            awareness_condition: Level of budget awareness to report
            researcher_budget_total: Researcher's total budget per call (agent1)
            validator_budget_total: Validator's total budget per call (agent2)
            team_budget_total: Team's total budget
            agent1_name: Name of first agent (default: "Researcher")
            agent2_name: Name of second agent (default: "Validator")
            approval_state_key: State key to check for approval (default: "validator_feedback")
            approval_keyword: Keyword indicating approval (default: "APPROVED")
            max_iterations: Maximum iterations allowed (default: 3)
        """
        super().__init__(name=name, description=description)
        # Store as private attributes to avoid Pydantic field validation
        object.__setattr__(self, "_report_usage", report_usage)
        object.__setattr__(self, "_awareness_condition", awareness_condition)
        object.__setattr__(self, "_researcher_budget_total", researcher_budget_total)
        object.__setattr__(self, "_validator_budget_total", validator_budget_total)
        object.__setattr__(self, "_team_budget_total", team_budget_total)
        object.__setattr__(self, "_agent1_name", agent1_name)
        object.__setattr__(self, "_agent2_name", agent2_name)
        object.__setattr__(self, "_approval_state_key", approval_state_key)
        object.__setattr__(self, "_approval_keyword", approval_keyword)
        object.__setattr__(self, "_max_iterations", max_iterations)
        object.__setattr__(self, "_current_iteration", 0)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check state for approval and escalate if found.

        Args:
            ctx: Invocation context with session state

        Yields:
            Event with usage report (if enabled) and escalate flag
        """
        # Increment iteration counter
        current_iter = getattr(self, "_current_iteration", 0) + 1
        object.__setattr__(self, "_current_iteration", current_iter)

        # Report token usage if enabled
        usage_message = ""
        if getattr(self, "_report_usage", True):
            agent1_name = getattr(self, "_agent1_name", "Researcher")
            agent2_name = getattr(self, "_agent2_name", "Validator")

            r_total = ctx.session.state.get(f"{agent1_name.lower()}_total_tokens", 0)
            v_total = ctx.session.state.get(f"{agent2_name.lower()}_total_tokens", 0)
            team_total = r_total + v_total

            if team_total > 0:
                usage_message = self._generate_usage_message(
                    r_total=r_total,
                    v_total=v_total,
                    team_total=team_total,
                    current_iteration=current_iter,
                )

        # Get approval feedback from state using configured key
        approval_state_key = getattr(self, "_approval_state_key", "validator_feedback")
        approval_keyword = getattr(self, "_approval_keyword", "APPROVED")

        feedback = ctx.session.state.get(approval_state_key, "")

        # Check if approval keyword present
        approved = approval_keyword in str(feedback).upper()

        # Yield event with usage message and escalate flag
        # escalate=True signals LoopAgent to exit
        if usage_message:
            content = types.Content(
                role="model", parts=[types.Part(text=usage_message)]
            )
            yield Event(
                author=self.name,
                content=content,
                actions=EventActions(escalate=approved),
            )
        else:
            # No usage message, just escalate flag
            yield Event(
                author=self.name,
                actions=EventActions(escalate=approved),
            )

    def _generate_usage_message(
        self,
        r_total: int,
        v_total: int,
        team_total: int,
        current_iteration: int = 1,
    ) -> str:
        """Generate iteration-aware status message with challenge framing.

        Design principles (from literature review):
        1. Include iteration context (temporal awareness)
        2. Frame remaining resources positively (what you CAN do)
        3. Provide actionable guidance for next iteration
        4. Avoid threatening language (no emojis, no "WARNING/CRITICAL")

        Args:
            r_total: Agent1 cumulative tokens used
            v_total: Agent2 cumulative tokens used
            team_total: Team cumulative tokens used
            current_iteration: Current iteration number (1-indexed)

        Returns:
            Formatted status message with iteration context
        """
        awareness = getattr(
            self, "_awareness_condition", MultiAgentAwarenessCondition.NO_AWARENESS
        )
        r_budget = getattr(self, "_researcher_budget_total", 0)
        v_budget = getattr(self, "_validator_budget_total", 0)
        team_budget = getattr(self, "_team_budget_total", 0)
        max_iterations = getattr(self, "_max_iterations", 3)
        agent1_name = getattr(self, "_agent1_name", "Researcher")
        agent2_name = getattr(self, "_agent2_name", "Validator")

        remaining_iterations = max_iterations - current_iteration

        # Start with iteration context (temporal awareness)
        lines = [
            f"[STATUS: Iteration {current_iteration} of {max_iterations} complete]"
        ]

        # Individual agent details (for OVERALL_AND_INDIVIDUAL)
        if awareness == MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL:
            # Agent 1 - focus on remaining capacity
            r_remaining = max(0, r_budget - r_total)
            r_remaining_pct = (r_remaining / r_budget * 100) if r_budget > 0 else 0
            lines.append(
                f"{agent1_name}: {r_remaining:,} tokens available ({r_remaining_pct:.0f}% of allocation)"
            )

            # Agent 2 - focus on remaining capacity
            v_remaining = max(0, v_budget - v_total)
            v_remaining_pct = (v_remaining / v_budget * 100) if v_budget > 0 else 0
            lines.append(
                f"{agent2_name}: {v_remaining:,} tokens available ({v_remaining_pct:.0f}% of allocation)"
            )

        # Team summary (for OVERALL_ONLY and OVERALL_AND_INDIVIDUAL)
        if awareness in [
            MultiAgentAwarenessCondition.OVERALL_ONLY,
            MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL,
        ]:
            team_remaining = max(0, team_budget - team_total)
            team_remaining_pct = (
                (team_remaining / team_budget * 100) if team_budget > 0 else 0
            )
            lines.append(
                f"Team: {team_remaining:,} tokens available ({team_remaining_pct:.0f}% remaining)"
            )

            # Add actionable guidance based on situation
            guidance = self._get_actionable_guidance(
                team_remaining_pct, remaining_iterations
            )
            if guidance:
                lines.append("")
                lines.append(guidance)

        # Fallback for other conditions (just show usage)
        if awareness not in [
            MultiAgentAwarenessCondition.OVERALL_ONLY,
            MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL,
        ]:
            lines.append(
                f"Usage this iteration: {agent1_name} {r_total:,}, {agent2_name} {v_total:,}"
            )

        lines.append("")  # Blank line at end
        return "\n".join(lines)

    def _get_actionable_guidance(
        self, remaining_pct: float, remaining_iterations: int
    ) -> str:
        """Generate actionable guidance based on resources and iterations remaining.

        Uses challenge framing (what you can do) not threat framing (what you can't).

        Args:
            remaining_pct: Percentage of budget remaining (0-100)
            remaining_iterations: Number of iterations left

        Returns:
            Actionable guidance string
        """
        if remaining_iterations == 0:
            # Final iteration - encourage focus
            return "Final iteration: Focus on the specific issue identified."

        if remaining_pct < 25:
            # Low resources but still iterations left
            return f"{remaining_iterations} iteration(s) left. Focus revisions on the specific failing test."
        elif remaining_pct < 50:
            # Moderate resources
            return f"{remaining_iterations} iteration(s) available. Target the root cause of the failure."
        else:
            # Plenty of resources
            return f"{remaining_iterations} iteration(s) available for refinement."
