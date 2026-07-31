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
    AgentName.FLIGHTS,
    AgentName.LODGING,
    AgentName.ATTRACTIONS,
    AgentName.ROUTING,
    AgentName.BUDGET,
    AgentName.CRITIC,
    AgentName.ITINERARY,
    AgentName.CALENDAR,
)


def _summarize(state: TripState) -> dict:
    """Summarize every agent's output without dumping it all into context.

    The manager only needs to know what exists and whether it succeeded, not
    the full contents - those belong to the agent that consumes them.

    Args:
        state: The state as seen by the manager on this turn.

    Returns:
        One short entry per stage that has produced something.
    """
    summary: dict = {}

    intake = state.get("intake")
    if intake is not None:
        summary["intake"] = {
            "is_complete": intake.is_complete,
            "missing_fields": intake.missing_fields,
            "destination": intake.profile.destination,
        }

    if state.get("research") is not None:
        summary["destination_research"] = {"done": True}

    flights = state.get("flights")
    if flights is not None:
        summary["flights"] = {
            "option_count": len(flights.options),
            "recommended": flights.recommended_option_id,
        }

    lodging = state.get("lodging")
    if lodging is not None:
        summary["lodging"] = {
            "option_count": len(lodging.options),
            "recommended": lodging.recommended_option_id,
        }

    attractions = state.get("attractions")
    if attractions is not None:
        summary["attractions"] = {"place_count": len(attractions.places)}

    routing = state.get("routing")
    if routing is not None:
        summary["routing"] = {
            "day_count": len(routing.days),
            "stop_count": sum(len(day.stops) for day in routing.days),
        }

    budget = state.get("budget")
    if budget is not None:
        summary["budget"] = {
            "total_cost": budget.total_cost,
            "currency": budget.currency,
            "within_budget": budget.within_budget,
        }

    critic = state.get("critic")
    if critic is not None:
        summary["critic"] = {
            "approved": critic.approved,
            "blocker_count": len(critic.blockers),
            "issue_count": len(critic.issues),
        }

    itinerary = state.get("itinerary")
    if itinerary is not None:
        summary["itinerary"] = {"title": itinerary.title, "day_count": len(itinerary.days)}

    calendar = state.get("calendar")
    if calendar is not None:
        summary["calendar"] = {
            "event_count": len(calendar.events),
            "ics_path": calendar.ics_path,
        }

    return summary


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
            The user's request, the agents already completed, how many times
            the plan has been revised, and a summary of each agent's output.
        """
        return {
            "user_request": state.get("user_request", ""),
            "completed_agents": [
                str(name) for name in state.get("completed_agents", [])
            ],
            "revision_count": state.get("revision_count", 0),
            "results": _summarize(state),
        }

    return read_trip_state


@tool
def delegate_to_agent(agent: str, reason: str) -> str:
    """Record the decision to hand the next step to a specific agent.

    Args:
        agent: The agent to run next, e.g. "intake", "flights", "critic".
        reason: Why this agent should run now.

    Returns:
        Confirmation of the delegation, or an error if the agent is unknown.
    """
    valid = {str(name) for name in AGENT_SEQUENCE}
    if agent not in valid:
        return f"Unknown agent '{agent}'. Valid agents are: {', '.join(sorted(valid))}."
    return f"Delegated to {agent}. Reason: {reason}"
