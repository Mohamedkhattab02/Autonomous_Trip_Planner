"""Routing Agent (plan.md agent #7).

Takes the pool of places from the Attractions Agent and distributes it across
the trip's days, so each day is geographically tight and nothing is scheduled
when it is closed.

This is also the agent the Critic sends work back to: when a plan is rejected,
the graph returns here with the issues attached.
"""

from __future__ import annotations

from langchain.agents import create_agent

from trip_planner.llm import get_model
from trip_planner.schemas import (
    AgentName,
    AttractionsResult,
    CriticResult,
    RoutingResult,
    TravelerProfile,
)
from trip_planner.state import TripState
from trip_planner.tools import ROUTING_TOOLS

# The traveler's day. Stops are scheduled inside this window.
DAY_START = "09:00"
DAY_END = "21:00"

SYSTEM_PROMPT = """You are the Routing Agent of an autonomous trip planner.

Your job is to turn a pool of places into a day-by-day plan that a real person
could actually follow.

Follow these steps:
1. Call `cluster_locations` with every place and the number of days. Each
   cluster becomes one day, so no day criss-crosses the city.
2. Assign each cluster to a date. Before fixing a day, call
   `check_opening_hours` for each place in it - if a place is closed that day,
   swap it with a place from another day.
3. Order the stops within a day geographically. Use `calculate_distance`
   between consecutive stops and `calculate_travel_time` to get the gap,
   and put the result in `travel_minutes_from_previous` and `travel_mode`.
4. Assign start and end times using each place's visit duration plus the
   travel time to reach it.

Rules:
- The day runs from {day_start} to {day_end}. Never schedule outside it.
- Never overlap two stops: a stop starts only after the previous one ends plus
  the travel time between them.
- Schedule a meal stop around midday and another in the evening when the pool
  has restaurants.
- Four to five stops a day is realistic. More than six is not - leave the
  extras in `unscheduled_place_ids` rather than cramming them in.
- Use only `place_id` values from the pool you were given. Never invent a place.
- Every place must end up either scheduled or in `unscheduled_place_ids`, so
  nothing is silently dropped.
- If you are given critic feedback, fixing those issues takes priority over
  everything else. Address each one and say how in `reasoning`.
""".format(day_start=DAY_START, day_end=DAY_END)


def build_routing_agent():
    """Build the Routing Agent runnable."""
    return create_agent(
        model=get_model(),
        tools=ROUTING_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=RoutingResult,
        name="routing_agent",
    )


def _places_table(attractions: AttractionsResult) -> str:
    """Render the places pool as a compact table for the prompt.

    Args:
        attractions: The pool from the Attractions Agent.

    Returns:
        One line per place, with the fields routing needs.
    """
    lines = []
    for place in attractions.places:
        coordinates = (
            f"{place.coordinates.latitude:.5f},{place.coordinates.longitude:.5f}"
            if place.coordinates
            else "NO COORDINATES"
        )
        closed = ", ".join(place.closed_days) if place.closed_days else "never"
        lines.append(
            f"- {place.place_id} | {place.name} | {place.category} | "
            f"coords: {coordinates} | visit: {place.visit_duration_minutes} min | "
            f"closed: {closed} | hours: {place.opening_hours}"
        )
    return "\n".join(lines) or "- (the attractions pool is empty)"


def _critic_feedback(critic: CriticResult | None) -> str:
    """Render the Critic's issues as instructions for a revision.

    Args:
        critic: The Critic's verdict, or None on the first routing pass.

    Returns:
        The feedback block for the prompt, or an empty string on a first pass.
    """
    if critic is None or critic.approved:
        return ""

    issues = "\n".join(
        f"- [{issue.severity}] {issue.location or 'plan'}: {issue.description}"
        f" -> fix: {issue.suggested_fix or 'reschedule as needed'}"
        for issue in critic.issues
    )
    return (
        "\n\nTHIS IS A REVISION. The Critic rejected the previous plan for "
        f"these reasons - fix every one of them:\n{issues}"
    )


def _routing_brief(
    profile: TravelerProfile,
    attractions: AttractionsResult,
    critic: CriticResult | None,
    days: int,
) -> str:
    """Build the routing request.

    Args:
        profile: The traveler profile.
        attractions: The pool of candidate places.
        critic: The Critic's verdict, when this is a revision.
        days: Number of days to fill.

    Returns:
        A prompt describing the days to build and the places to use.
    """
    constraints = (
        "\n".join(f"- {item}" for item in profile.constraints)
        if profile.constraints
        else "- none stated"
    )
    return (
        f"Build a day-by-day plan for this trip.\n"
        f"Destination: {profile.destination}\n"
        f"Dates: {profile.start_date} to {profile.end_date}\n"
        f"Days to fill: {days} (day 1 is {profile.start_date})\n"
        f"Travelers: {profile.travelers or 1}\n"
        f"Constraints to respect:\n{constraints}\n\n"
        f"Places available:\n{_places_table(attractions)}"
        f"{_critic_feedback(critic)}"
    )


def routing_node(state: TripState) -> dict:
    """Graph node: run the Routing Agent.

    On a revision the Critic's issues are passed in, and the previous critic
    verdict is cleared so the next critic pass judges the new plan.

    Args:
        state: The shared trip state; reads `intake`, `attractions`, `critic`.

    Returns:
        A partial state update with the day-by-day plan.
    """
    profile = state["intake"].profile
    attractions = state["attractions"]
    critic = state.get("critic")
    is_revision = critic is not None and not critic.approved

    days = (
        (profile.end_date - profile.start_date).days + 1
        if profile.start_date and profile.end_date
        else 1
    )

    agent = build_routing_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": _routing_brief(profile, attractions, critic, days),
                }
            ]
        }
    )
    routing: RoutingResult = result["structured_response"]

    update: dict = {"routing": routing}
    if is_revision:
        # Count the revision; the manager uses it to stop an endless retry loop.
        update["revision_count"] = state.get("revision_count", 0) + 1
    else:
        # Only the first pass records completion, so `completed_agents` does
        # not accumulate duplicates across revisions.
        update["completed_agents"] = [AgentName.ROUTING]

    return update
