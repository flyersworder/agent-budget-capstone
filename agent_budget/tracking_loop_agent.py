"""Custom LoopAgent that tracks token usage before passing control to next agent."""

from typing import AsyncGenerator

from google.adk.agents import LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from agent_budget.usage_tracker import extract_token_usage, update_usage_in_dict


class TrackingLoopAgent(LoopAgent):  # type: ignore[misc]
    """LoopAgent that tracks token usage internally.

    This agent intercepts events from sub-agents and updates session state
    with token usage BEFORE yielding the event. This ensures that CheckApproval
    agent sees updated usage when it reads the session state.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Run loop with internal token tracking."""
        if not self.sub_agents:
            return

        # Get agent state for resumption
        from google.adk.agents.loop_agent import LoopAgentState

        agent_state = self._load_agent_state(ctx, LoopAgentState)
        is_resuming_at_current_agent = agent_state is not None
        times_looped, start_index = self._get_start_state(agent_state)

        should_exit = False
        pause_invocation = False

        while (not self.max_iterations or times_looped < self.max_iterations) and not (
            should_exit or pause_invocation
        ):
            for i in range(start_index, len(self.sub_agents)):
                sub_agent = self.sub_agents[i]

                if ctx.is_resumable and not is_resuming_at_current_agent:
                    agent_state = LoopAgentState(
                        current_sub_agent=sub_agent.name,
                        times_looped=times_looped,
                    )
                    ctx.set_agent_state(self.name, agent_state=agent_state)
                    yield self._create_agent_state_event(ctx)

                is_resuming_at_current_agent = False

                # Run sub-agent and track tokens
                from google.adk.utils.context_utils import Aclosing

                async with Aclosing(sub_agent.run_async(ctx)) as agen:
                    async for event in agen:
                        # Track token usage BEFORE yielding event
                        # This ensures state is updated before next agent runs
                        if hasattr(event, "author") and event.author in [
                            "Researcher",
                            "Validator",
                            "Coder",
                            "Reviewer",
                        ]:
                            thinking, output = extract_token_usage(event)
                            if thinking + output > 0:
                                update_usage_in_dict(
                                    state=ctx.session.state,
                                    agent_name=event.author,
                                    thinking_tokens=thinking,
                                    output_tokens=output,
                                )

                        # Now yield the event
                        yield event

                        if event.actions.escalate:
                            should_exit = True
                        if ctx.should_pause_invocation(event):
                            pause_invocation = True

                if should_exit or pause_invocation:
                    break  # break inner for loop

            # Restart from the beginning of the loop.
            start_index = 0
            times_looped += 1
            # Reset the state of all sub-agents in the loop.
            ctx.reset_sub_agent_states(self.name)

        # If the invocation is paused, we should not yield the end of agent event.
        if pause_invocation:
            return

        if ctx.is_resumable:
            ctx.set_agent_state(self.name, end_of_agent=True)
            yield self._create_agent_state_event(ctx)
