"""Custom agents for iterative loop patterns.

This module provides specialized agents for controlling loop execution
in multi-agent iterative workflows.
"""

from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions


class CheckApprovalAgent(BaseAgent):  # type: ignore[misc]
    """Custom agent that checks for approval and escalates to exit loop.

    This agent examines the validator's feedback in the session state.
    If the feedback contains "APPROVED", it signals the LoopAgent to
    terminate by escalating.

    Always checks state['validator_feedback'] for "APPROVED" keyword.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check state for approval and escalate if found.

        Args:
            ctx: Invocation context with session state

        Yields:
            Event with escalate=True if approved, else escalate=False
        """
        # Get validator feedback from state
        # Always check 'validator_feedback' key for 'APPROVED' keyword
        feedback = ctx.session.state.get("validator_feedback", "")

        # Check if approval keyword present
        approved = "APPROVED" in str(feedback)

        # Yield event with escalate flag
        # escalate=True signals LoopAgent to exit
        # Note: content can be None for simple control agents
        yield Event(
            author=self.name,
            actions=EventActions(escalate=approved),
        )
