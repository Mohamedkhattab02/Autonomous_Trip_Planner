"""Lodging Agent (plan.md agent #5).

Finds real places to stay that fit the budget, the location and the
traveler's preferences, and recommends one.
"""

from __future__ import annotations

from langchain.agents import create_agent

from trip_planner.llm import get_model
from trip_planner.schemas import AgentName, LodgingResult, TravelerProfile
from trip_planner.state import TripState
from trip_planner.tools import LODGING_TOOLS

SYSTEM_PROMPT = """You are the Lodging Agent of an autonomous trip planner.

Your job is to find real places to stay and recommend one.

Follow these steps:
1. Call `search_hotels` for the destination, the trip dates and the number of
   travelers. If a nightly budget was given, pass it as `max_price_per_night`.
2. Call `compare_hotels` on the results to rank them by price against rating.
3. For the top few, call `check_hotel_location` against the landmarks the
   traveler will actually visit, to confirm the location is genuinely central.
4. Call `get_hotel_details` on the leading option when you need amenities or
   check-in times to justify the recommendation.
5. Return the ranked options with a recommendation.

Rules:
- Report only stays that `search_hotels` actually returned. Never invent a
  hotel, a price or a rating.
- `total_price` is the cost of the whole stay for all travelers. Copy the
  coordinates through unchanged - the Routing Agent needs them.
- Order `options` best first, following the `compare_hotels` ranking, and set
  `recommended_option_id` to its `best_option_id` unless location or a stated
  constraint rules it out - if it does, explain the override in `reasoning`.
- Give every option at least one honest `pros` entry and one `cons` entry.
- If the search returns an error or nothing, return an empty `options` list and
  say so plainly in `reasoning`. Do not fabricate a fallback.
"""


def build_lodging_agent():
    """Build the Lodging Agent runnable."""
    return create_agent(
        model=get_model(),
        tools=LODGING_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=LodgingResult,
        name="lodging_agent",
    )


def _lodging_brief(profile: TravelerProfile, nights: int) -> str:
    """Turn the traveler profile into the lodging search request.

    Args:
        profile: The profile produced by the Intake Agent.
        nights: Number of nights the stay must cover.

    Returns:
        A prompt describing what to look for.
    """
    interests = ", ".join(profile.interests) if profile.interests else "none stated"
    constraints = (
        "\n".join(f"- {item}" for item in profile.constraints)
        if profile.constraints
        else "- none stated"
    )
    return (
        f"Find a place to stay for this trip.\n"
        f"Destination: {profile.destination}\n"
        f"Check-in: {profile.start_date}\n"
        f"Check-out: {profile.end_date}\n"
        f"Nights: {nights}\n"
        f"Travelers: {profile.travelers or 2}\n"
        f"Currency: {profile.budget_currency or 'USD'}\n"
        f"Total trip budget: {profile.budget_amount or 'not stated'} "
        f"(lodging should stay well inside it, alongside flights and activities)\n"
        f"Interests: {interests}\n"
        f"Constraints to respect:\n{constraints}"
    )


def lodging_node(state: TripState) -> dict:
    """Graph node: run the Lodging Agent.

    Args:
        state: The shared trip state; reads `intake`.

    Returns:
        A partial state update with the lodging options.
    """
    profile = state["intake"].profile
    nights = (
        (profile.end_date - profile.start_date).days
        if profile.start_date and profile.end_date
        else 1
    )
    agent = build_lodging_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": _lodging_brief(profile, nights)}]}
    )
    lodging: LodgingResult = result["structured_response"]

    return {
        "lodging": lodging,
        "completed_agents": [AgentName.LODGING],
    }
