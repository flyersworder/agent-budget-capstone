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
        max_iterations = getattr(self, "_max_iterations", 3)
        agent1_name = getattr(self, "_agent1_name", "Researcher")
        agent2_name = getattr(self, "_agent2_name", "Validator")

        remaining_iterations = max_iterations - current_iteration

        # Status message with both agents' token usage from this iteration
        # - Shows iteration progress
        # - Shows each agent's tokens used (so both can adjust)
        # - Both agents see this message
        lines = [
            f"[STATUS: Iteration {current_iteration} of {max_iterations} complete]",
            f"{agent1_name} tokens used: {r_total:,}",
            f"{agent2_name} tokens used: {v_total:,}",
            f"{remaining_iterations} iteration(s) remaining.",
        ]

        lines.append("")  # Blank line at end
        return "\n".join(lines)

    def _get_actionable_guidance(
        self, remaining_pct: float, remaining_iterations: int
    ) -> str:
        """Generate simple iteration count - no behavioral guidance.

        Keeping this minimal to avoid confounding the budget awareness manipulation.

        Args:
            remaining_pct: Percentage of budget remaining (0-100)
            remaining_iterations: Number of iterations left

        Returns:
            Simple iteration info string
        """
        if remaining_iterations == 0:
            return "Final iteration."
        else:
            return f"{remaining_iterations} iteration(s) remaining."
