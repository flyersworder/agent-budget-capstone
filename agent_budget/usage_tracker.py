"""Helper functions for tracking token usage across iterations.

This module provides utilities for extracting token usage from events
and updating session state with cumulative counts.
"""

from typing import Any

from google.adk.events import Event
from google.adk.sessions import Session


def extract_token_usage(event: Event) -> tuple[int, int]:
    """Extract token usage from an event.

    Args:
        event: Event with potential usage_metadata

    Returns:
        Tuple of (thinking_tokens, output_tokens)
    """
    if not hasattr(event, "usage_metadata") or not event.usage_metadata:
        return (0, 0)

    # Extract thinking tokens (reasoning)
    thinking = getattr(event.usage_metadata, "thoughts_token_count", 0) or 0

    # Extract output tokens (response + tool use)
    candidates = getattr(event.usage_metadata, "candidates_token_count", 0) or 0
    tool_tokens = getattr(event.usage_metadata, "tool_use_prompt_token_count", 0) or 0
    output = candidates + tool_tokens

    return (thinking, output)


def update_usage_in_dict(
    state: dict[str, Any],
    agent_name: str,
    thinking_tokens: int,
    output_tokens: int,
) -> None:
    """Update cumulative token usage in a state dictionary.

    This directly modifies the provided state dict.

    Args:
        state: State dictionary to update
        agent_name: Name of agent (e.g., "Researcher", "Validator")
        thinking_tokens: Thinking tokens used in this call
        output_tokens: Output tokens used in this call
    """
    # Update cumulative counts
    key_thinking = f"{agent_name.lower()}_thinking_tokens"
    key_output = f"{agent_name.lower()}_output_tokens"
    key_total = f"{agent_name.lower()}_total_tokens"

    current_thinking = state.get(key_thinking, 0)
    current_output = state.get(key_output, 0)

    state[key_thinking] = current_thinking + thinking_tokens
    state[key_output] = current_output + output_tokens
    state[key_total] = (
        current_thinking + thinking_tokens + current_output + output_tokens
    )


def format_usage_status(session: Session, agent_name: str, budget_total: int) -> str:
    """Format usage status message for agent instructions.

    Args:
        session: Current session with state
        agent_name: Name of agent (e.g., "Researcher", "Validator")
        budget_total: Total budget per call

    Returns:
        Formatted status string
    """
    key_thinking = f"{agent_name.lower()}_thinking_tokens"
    key_output = f"{agent_name.lower()}_output_tokens"
    key_total = f"{agent_name.lower()}_total_tokens"

    thinking_used = session.state.get(key_thinking, 0)
    output_used = session.state.get(key_output, 0)
    total_used = session.state.get(key_total, 0)

    if total_used == 0:
        return "This is your first iteration - no tokens used yet."

    return f"""Token usage so far across iterations:
- Thinking: {thinking_used} tokens
- Output: {output_used} tokens
- Total: {total_used} tokens
- Budget per call: {budget_total} tokens"""


def report_team_usage_from_state(
    state: dict[str, Any],
    agent1_name: str = "Researcher",
    agent2_name: str = "Validator",
) -> str:
    """Report cumulative team token usage from session state.

    This function is meant to be called by a UsageReporter agent
    to provide budget awareness feedback to the team.

    Args:
        state: Session state dictionary
        agent1_name: Name of first agent (default: "Researcher")
        agent2_name: Name of second agent (default: "Validator")

    Returns:
        Formatted usage report string
    """
    agent1_total = state.get(f"{agent1_name.lower()}_total_tokens", 0)
    agent2_total = state.get(f"{agent2_name.lower()}_total_tokens", 0)
    team_total = agent1_total + agent2_total

    if team_total == 0:
        return "[BUDGET USAGE] This is iteration 1 - no tokens used yet."

    return f"""[BUDGET USAGE UPDATE]
Cumulative token usage across all iterations so far:
- {agent1_name}: {agent1_total} tokens
- {agent2_name}: {agent2_total} tokens
- Team total: {team_total} tokens

This information is for your awareness. Continue with your task."""
