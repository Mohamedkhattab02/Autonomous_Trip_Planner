"""Critic Agent (plan.md agent #9).

The gate before the final itinerary. It checks that the places are real, the
schedule is physically possible, and no stated constraint or budget limit is
broken. When it finds a blocker, the graph sends the plan back to the Routing
Agent for a revision.
"""

from __future__ import annotations

from langchain.agents import create_agent

from trip_planner.llm import get_model
from trip_planner.schemas import (
    AgentName,
    AttractionsResult,
    BudgetResult,
    CriticResult,
    RoutingResult,
    TravelerProfile,
)
from trip_planner.state import TripState
from trip_planner.tools import CRITIC_TOOLS

SYSTEM_PROMPT = """You are the Critic Agent of an autonomous trip planner.

Your job is to find what is wrong with the draft plan before the traveler
ever sees it. Be specific and skeptical, but fair: report real problems, not
matters of taste.

Follow these steps:
1. Call `validate_schedule` with the whole day plan. It finds overlaps, stops
   that start before travel could finish, and times outside the day.
2. Call `validate_budget` with the total and the traveler's budget.
3. Call `verify_place` on any place whose name looks generic or invented, to
   confirm it exists and is not permanently closed.
4. Call `verify_opening_hours` for places on days where being closed would
   wreck that day, especially museums, which usually close one weekday.
5. Check every stated traveler constraint against the plan yourself.

Rules:
- Report only problems you actually confirmed with a tool or can point to in
  the plan. Do not invent issues to look thorough.
- Use `blocker` only for something that makes the plan unusable: a place that
  does not exist, a visit on a day it is closed, overlapping stops, or a
  breached budget. Use `warning` for a tight connection or a long transfer,
  and `info` for a suggestion.
- Set `approved` to true only when there is not a single `blocker` issue.
- Every issue needs a `location` naming where it occurs (e.g. "day 2,
  Gulbenkian Museum") and a `suggested_fix` the Routing Agent can act on.
- If a verification tool fails or returns nothing, that is not proof of a
  problem. Say so in `reasoning` and do not raise a blocker for it.
"""


def build_critic_agent():
    """Build the Critic Agent runnable."""
    return create_agent(
        model=get_model(),
        tools=CRITIC_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=CriticResult,
        name="critic_agent",
    )


def _plan_table(
    routing: RoutingResult | None, attractions: AttractionsResult | None
) -> str:
    """Render the draft day plan for the Critic to check.

    Args:
        routing: The day-by-day plan.
        attractions: The pool the plan drew from, holding closing days.

    Returns:
        The plan as text, one line per stop.
    """
    if routing is None or not routing.days:
        return "(the plan has no days at all)"

    closed = {
        place.place_id: ", ".join(place.closed_days) or "never"
        for place in (attractions.places if attractions else [])
    }

    lines = []
    for day in routing.days:
        lines.append(f"Day {day.day_number} - {day.date} ({day.summary})")
        for stop in day.stops:
            lines.append(
                f"  - {stop.place_id} | {stop.name} | {stop.start_time}-{stop.end_time}"
                f" | travel from previous: {stop.travel_minutes_from_previous} min"
                f" by {stop.travel_mode}"
                f" | normally closed: {closed.get(stop.place_id, 'unknown')}"
            )
    return "\n".join(lines)


def _budget_line(budget: BudgetResult | None) -> str:
    """Describe the budget position for the Critic.

    Args:
        budget: The Budget Agent's output, when it ran.

    Returns:
        A one-line budget summary.
    """
    if budget is None:
        return "The budget stage has not run, so do not raise budget issues."
    return (
        f"total {budget.total_cost} {budget.currency} against a budget of "
        f"{budget.budget_amount} {budget.currency} "
        f"(agent reported within_budget={budget.within_budget})"
    )


def _critic_brief(
    profile: TravelerProfile,
    routing: RoutingResult | None,
    attractions: AttractionsResult | None,
    budget: BudgetResult | None,
) -> str:
    """Build the review request.

    Args:
        profile: The traveler profile, holding the constraints to enforce.
        routing: The day-by-day plan under review.
        attractions: The pool of places the plan used.
        budget: The budget position.

    Returns:
        A prompt containing the whole draft plan.
    """
    constraints = (
        "\n".join(f"- {item}" for item in profile.constraints)
        if profile.constraints
        else "- none stated"
    )
    unscheduled = (
        ", ".join(routing.unscheduled_place_ids)
        if routing and routing.unscheduled_place_ids
        else "none"
    )
    return (
        f"Review this draft trip plan.\n"
        f"Destination: {profile.destination}\n"
        f"Dates: {profile.start_date} to {profile.end_date}\n"
        f"Travelers: {profile.travelers or 1}\n"
        f"Constraints the traveler stated:\n{constraints}\n\n"
        f"Budget position: {_budget_line(budget)}\n\n"
        f"The plan:\n{_plan_table(routing, attractions)}\n\n"
        f"Places left unscheduled: {unscheduled}"
    )


def critic_node(state: TripState) -> dict:
    """Graph node: run the Critic Agent.

    Args:
        state: The shared trip state; reads `intake`, `routing`,
            `attractions` and `budget`.

    Returns:
        A partial state update with the Critic's verdict. Completion is
        recorded only on the first pass, so a revision does not add a
        duplicate entry to `completed_agents`.
    """
    profile = state["intake"].profile
    agent = build_critic_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": _critic_brief(
                        profile,
                        state.get("routing"),
                        state.get("attractions"),
                        state.get("budget"),
                    ),
                }
            ]
        }
    )
    critic: CriticResult = result["structured_response"]

    update: dict = {"critic": critic}
    if AgentName.CRITIC not in state.get("completed_agents", []):
        update["completed_agents"] = [AgentName.CRITIC]

    return update
