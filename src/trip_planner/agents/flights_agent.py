"""Flights Agent (plan.md agent #4).

Searches real flights for the traveler's route and dates, ranks them, and
recommends one with its trade-offs stated.
"""

from __future__ import annotations

from trip_planner.agents.factory import build_structured_agent, run_agent
from trip_planner.schemas import AgentName, FlightsResult, TravelerProfile
from trip_planner.state import TripState
from trip_planner.tools import FLIGHT_TOOLS

SYSTEM_PROMPT = """You are the Flights Agent of an autonomous trip planner.

Your job is to find real flights for the traveler and recommend one.

Follow these steps:
1. Convert the origin and destination cities to their main airport IATA codes
   (Tel Aviv -> TLV, Lisbon -> LIS, Paris -> CDG, and so on).
2. Call `search_flights` with those codes, the trip dates, the number of
   travelers and the traveler's budget currency.
3. Call `compare_flights` with the options it returned to rank them.
4. If a shortlisted option lacks baggage information the traveler needs, call
   `get_flight_details` for it.
5. Return the ranked options with a recommendation.

Rules:
- Report only flights that `search_flights` actually returned. Never invent a
  flight number, price or schedule.
- `price` is the total for all travelers, in the currency you searched with.
- Order `options` best first, following the `compare_flights` ranking, and set
  `recommended_option_id` to its `best_option_id` unless a stated constraint
  rules that option out - if it does, explain the override in `reasoning`.
- Give every option at least one honest `pros` entry and one `cons` entry.
- If the search returns an error or no flights, return an empty `options` list
  and say so plainly in `reasoning`. Do not fabricate a fallback.
"""


def build_flights_agent():
    """Build the Flights Agent runnable."""
    return build_structured_agent(
        role="flights",
        tools=FLIGHT_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=FlightsResult,
        name="flights_agent",
    )


def _flights_brief(profile: TravelerProfile) -> str:
    """Turn the traveler profile into the flight search request.

    Args:
        profile: The profile produced by the Intake Agent.

    Returns:
        A prompt describing which flights to look for.
    """
    constraints = (
        "\n".join(f"- {item}" for item in profile.constraints)
        if profile.constraints
        else "- none stated"
    )
    return (
        f"Find flights for this trip.\n"
        f"Origin: {profile.origin or 'not stated'}\n"
        f"Destination: {profile.destination}\n"
        f"Departure date: {profile.start_date}\n"
        f"Return date: {profile.end_date}\n"
        f"Travelers: {profile.travelers or 1}\n"
        f"Budget currency: {profile.budget_currency or 'USD'}\n"
        f"Total trip budget: {profile.budget_amount or 'not stated'}\n"
        f"Constraints to respect:\n{constraints}"
    )


def flights_node(state: TripState, collector=None) -> dict:
    """Graph node: run the Flights Agent.

    Args:
        state: The shared trip state; reads `intake`.

    Returns:
        A partial state update with the flight options.
    """
    profile = state["intake"].profile
    flights: FlightsResult = run_agent(
        build_flights_agent(),
        _flights_brief(profile),
        role="flights",
        collector=collector,
    )

    return {
        "flights": flights,
        "completed_agents": [AgentName.FLIGHTS],
    }
