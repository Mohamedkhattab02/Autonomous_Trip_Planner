"""Tools for the Travel Manager Agent (plan.md agent #1).

The manager does not carry out work itself; it inspects what has been done
and decides who runs next. `read_trip_state` is bound to the live state at
graph-build time, so the manager sees the same state the graph does.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.tools import tool

from trip_planner.schemas import AgentName
from trip_planner.state import TripState

# The order stages must run in. The manager may not skip ahead, because each
# stage consumes the previous one's output.
AGENT_SEQUENCE: tuple[AgentName, ...] = (
    AgentName.INTAKE,
    AgentName.DESTINATION_RESEARCH,
)


def make_read_trip_state(state: TripState) -> Callable:
    """Build a `read_trip_state` tool bound to one graph invocation's state.

    Args:
        state: The state as seen by the manager node on this turn.

    Returns:
        A LangChain tool the manager can call to inspect progress.
    """

    @tool
    def read_trip_state() -> dict:
        """Read what the trip planner knows so far and which agents have run.

        Returns:
            The user's request, the agents already completed, and a summary
            of each agent's output.
        """
        intake = state.get("intake")
        research = state.get("research")
        return {
            "user_request": state.get("user_request", ""),
            "completed_agents": [
                str(name) for name in state.get("completed_agents", [])
            ],
            "intake": intake.model_dump(mode="json") if intake else None,
            "research_done": research is not None,
        }

    return read_trip_state


@tool
def delegate_to_agent(agent: str, reason: str) -> str:
    """Record the decision to hand the next step to a specific agent.

    Args:
        agent: The agent to run next. One of "intake", "destination_research".
        reason: Why this agent should run now.

    Returns:
        Confirmation of the delegation, or an error if the agent is unknown.
    """
    valid = {str(name) for name in AGENT_SEQUENCE}
    if agent not in valid:
        return f"Unknown agent '{agent}'. Valid agents are: {', '.join(sorted(valid))}."
    return f"Delegated to {agent}. Reason: {reason}"
