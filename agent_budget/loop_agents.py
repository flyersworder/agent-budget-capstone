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

    Also reports cumulative token usage for budget awareness with rich context
    including budget limits, percentages, and soft warnings.

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
    ):
        """Initialize CheckApprovalAgent.

        Args:
            name: Agent name
            description: Agent description
            report_usage: Whether to report token usage (default: True)
            awareness_condition: Level of budget awareness to report
            researcher_budget_total: Researcher's total budget per call
            validator_budget_total: Validator's total budget per call
            team_budget_total: Team's total budget
        """
        super().__init__(name=name, description=description)
        # Store as private attributes to avoid Pydantic field validation
        object.__setattr__(self, "_report_usage", report_usage)
        object.__setattr__(self, "_awareness_condition", awareness_condition)
        object.__setattr__(self, "_researcher_budget_total", researcher_budget_total)
        object.__setattr__(self, "_validator_budget_total", validator_budget_total)
        object.__setattr__(self, "_team_budget_total", team_budget_total)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check state for approval and escalate if found.

        Args:
            ctx: Invocation context with session state

        Yields:
            Event with usage report (if enabled) and escalate flag
        """
        # Report token usage if enabled
        usage_message = ""
        if getattr(self, "_report_usage", True):
            r_total = ctx.session.state.get("researcher_total_tokens", 0)
            v_total = ctx.session.state.get("validator_total_tokens", 0)
            team_total = r_total + v_total

            if team_total > 0:
                usage_message = self._generate_usage_message(
                    r_total=r_total,
                    v_total=v_total,
                    team_total=team_total,
                )

        # Get validator feedback from state
        # Always check 'validator_feedback' key for 'APPROVED' keyword
        feedback = ctx.session.state.get("validator_feedback", "")

        # Check if approval keyword present
        approved = "APPROVED" in str(feedback)

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
    ) -> str:
        """Generate rich usage message with budget context and warnings.

        Args:
            r_total: Researcher cumulative tokens used
            v_total: Validator cumulative tokens used
            team_total: Team cumulative tokens used

        Returns:
            Formatted usage message with context and warnings
        """
        awareness = getattr(
            self, "_awareness_condition", MultiAgentAwarenessCondition.NO_AWARENESS
        )
        r_budget = getattr(self, "_researcher_budget_total", 0)
        v_budget = getattr(self, "_validator_budget_total", 0)
        team_budget = getattr(self, "_team_budget_total", 0)

        lines = ["[BUDGET STATUS]"]

        # Individual agent details (for OVERALL_AND_INDIVIDUAL)
        if awareness == MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL:
            # Researcher
            r_pct = (r_total / r_budget * 100) if r_budget > 0 else 0
            r_remaining = max(0, r_budget - r_total)
            lines.append(
                f"Researcher: {r_total:,} / {r_budget:,} tokens ({r_pct:.0f}% used, {r_remaining:,} remaining)"
            )

            # Validator
            v_pct = (v_total / v_budget * 100) if v_budget > 0 else 0
            v_remaining = max(0, v_budget - v_total)
            lines.append(
                f"Validator: {v_total:,} / {v_budget:,} tokens ({v_pct:.0f}% used, {v_remaining:,} remaining)"
            )
            lines.append("")

        # Team total (for OVERALL_ONLY and OVERALL_AND_INDIVIDUAL)
        if awareness in [
            MultiAgentAwarenessCondition.OVERALL_ONLY,
            MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL,
        ]:
            team_pct = (team_total / team_budget * 100) if team_budget > 0 else 0
            team_remaining = max(0, team_budget - team_total)
            lines.append(
                f"Team total: {team_total:,} / {team_budget:,} tokens ({team_pct:.0f}% used, {team_remaining:,} remaining)"
            )

            # Generate warning based on usage percentage
            warning = self._get_warning_message(team_pct)
            if warning:
                lines.append("")
                lines.append(warning)

        # Fallback for other conditions (just show usage)
        if awareness not in [
            MultiAgentAwarenessCondition.OVERALL_ONLY,
            MultiAgentAwarenessCondition.OVERALL_AND_INDIVIDUAL,
        ]:
            lines.append(
                f"Cumulative usage: Researcher {r_total:,}, Validator {v_total:,}, Team {team_total:,}"
            )

        lines.append("")  # Blank line at end
        return "\n".join(lines)

    def _get_warning_message(self, usage_pct: float) -> str:
        """Generate warning message based on usage percentage.

        Args:
            usage_pct: Percentage of budget used (0-100+)

        Returns:
            Warning message or empty string if no warning needed
        """
        if usage_pct >= 90:
            return "🚨 CRITICAL: Over 90% budget used - be very concise!"
        elif usage_pct >= 75:
            return "⚠️  WARNING: Over 75% budget used - please be concise."
        elif usage_pct >= 50:
            return "ℹ️  NOTE: Over 50% budget used - monitor token usage carefully."
        return ""
