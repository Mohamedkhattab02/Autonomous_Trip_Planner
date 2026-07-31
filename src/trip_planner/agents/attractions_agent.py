"""Attractions Agent (plan.md agent #6).

Builds the pool of real places the Routing Agent will distribute across the
days: attractions, restaurants and activities that match the traveler's
interests, each with coordinates, hours and cost.
"""

from __future__ import annotations

from langchain.agents import create_agent

from trip_planner.llm import get_model
from trip_planner.schemas import AgentName, AttractionsResult, TravelerProfile
from trip_planner.state import TripState
from trip_planner.tools import ATTRACTION_TOOLS

# Roughly how many places make a full day, used to size the search.
PLACES_PER_DAY = 4

SYSTEM_PROMPT = """You are the Attractions Agent of an autonomous trip planner.

Your job is to build a pool of real places the traveler will enjoy. A later
agent splits them into days, so gather more than one day's worth.

Follow these steps:
1. Call `search_places` once per interest, e.g. "art museums", then
   "seafood restaurants", then "viewpoints". One focused search at a time
   returns far better results than one broad search.
2. Include restaurants, not only sights - the traveler has to eat.
3. Call `get_opening_hours` for any place whose hours came back unknown.
4. Call `get_place_details` for any place you found through `web_search`, so
   it gets coordinates.
5. Use `web_search` for seasonal events, festivals and ticket prices.

Rules:
- Report only places the tools actually returned. Never invent a place, an
  address or a rating.
- Every place must carry coordinates. Drop any place you cannot get them for,
  because the routing stage cannot schedule it.
- Copy `closed_days` through exactly as reported. The Routing and Critic
  agents rely on it to avoid sending the traveler to a closed museum.
- Set `estimated_cost` to the per-person entry price when you know it, and 0
  for places that are free. Do not guess a number you have no source for.
- Give each place a realistic `visit_duration_minutes`: about 60 for a
  viewpoint or church, 90-120 for a major museum, 90 for a sit-down meal.
- Fill `why_recommended` with how the place matches a stated interest.
"""


def build_attractions_agent():
    """Build the Attractions Agent runnable."""
    return create_agent(
        model=get_model(),
        tools=ATTRACTION_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=AttractionsResult,
        name="attractions_agent",
    )


def _attractions_brief(profile: TravelerProfile, days: int) -> str:
    """Turn the traveler profile into the attractions search request.

    Args:
        profile: The profile produced by the Intake Agent.
        days: Number of days the trip covers.

    Returns:
        A prompt describing what kind of places to gather.
    """
    interests = (
        ", ".join(profile.interests) if profile.interests else "general sightseeing"
    )
    constraints = (
        "\n".join(f"- {item}" for item in profile.constraints)
        if profile.constraints
        else "- none stated"
    )
    return (
        f"Find places to visit for this trip.\n"
        f"Destination: {profile.destination}\n"
        f"Dates: {profile.start_date} to {profile.end_date} ({days} days)\n"
        f"Travelers: {profile.travelers or 1}\n"
        f"Interests: {interests}\n"
        f"Currency: {profile.budget_currency or 'USD'}\n"
        f"Constraints to respect:\n{constraints}\n\n"
        f"Gather at least {days * PLACES_PER_DAY} places, covering every "
        f"interest listed and including restaurants for meals."
    )


def attractions_node(state: TripState) -> dict:
    """Graph node: run the Attractions Agent.

    Args:
        state: The shared trip state; reads `intake`.

    Returns:
        A partial state update with the pool of candidate places.
    """
    profile = state["intake"].profile
    days = (
        (profile.end_date - profile.start_date).days + 1
        if profile.start_date and profile.end_date
        else 1
    )
    agent = build_attractions_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": _attractions_brief(profile, days)}]}
    )
    attractions: AttractionsResult = result["structured_response"]

    return {
        "attractions": attractions,
        "completed_agents": [AgentName.ATTRACTIONS],
    }
