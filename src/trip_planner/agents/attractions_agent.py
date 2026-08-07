"""Attractions Agent (plan.md agent #6).

Builds the pool of real places the Routing Agent will distribute across the
days: attractions, restaurants and activities that match the traveler's
interests, each with coordinates, hours and cost.

**How big the pool is.** `PLACES_PER_DAY` sizes it against the trip, but
`max_total_places()` caps it — 15 by default, `TRIP_MAX_PLACES` to change it.
The cap matters because every place is carried whole through the rest of the
run: the Routing Agent reads the full table to cluster it, and the Critic
verifies each name. A week-long trip used to ask for 28 and a bulk search could
hand back ~50, most of which no day ever had room for. The cap is enforced
twice — the tools stop returning more, and `_capped` trims whatever the model
reports anyway — because a prompt asking for a number is a request, not a
limit.
"""

from __future__ import annotations

import logging

from trip_planner.agents.factory import build_structured_agent, run_agent
from trip_planner.schemas import AgentName, AttractionsResult, TravelerProfile
from trip_planner.state import TripState
from trip_planner.tools import ATTRACTION_TOOLS
from trip_planner.tools.attraction_tools import max_total_places

logger = logging.getLogger(__name__)

# Roughly how many places make a full day, used to size the search.
PLACES_PER_DAY = 4

SYSTEM_PROMPT = """You are the Attractions Agent of an autonomous trip planner.

Your job is to build a small, high-quality pool of real places the traveler
will enjoy. A later agent splits them into days.

The pool has a size limit, given in the request. It is a hard limit, not a
target to beat: a shorter list of places that genuinely fit the traveler's
interests is worth far more than a long one. Stop searching once you have
enough, and return only your best places, best first.

Follow these steps:
1. Call `search_places_bulk` ONCE with a list of focused queries, one per
   interest, e.g. ["art museums", "seafood restaurants", "viewpoints"]. It
   runs them in parallel, so this is far faster than searching one at a time,
   and it already returns the whole pool you are allowed. Use `search_places`
   only to follow up on a single gap afterwards.
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
    return build_structured_agent(
        role="attractions",
        tools=ATTRACTION_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=AttractionsResult,
        name="attractions_agent",
    )


def pool_size(days: int) -> int:
    """How many places to gather for a trip of this length.

    Args:
        days: Number of days the trip covers.

    Returns:
        Roughly a day's worth per day, never more than the configured cap.
    """
    return max(1, min(days * PLACES_PER_DAY, max_total_places()))


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
        f"Gather at most {pool_size(days)} places - that is a hard limit, and "
        f"anything past it is discarded. Spend them on every interest listed, "
        f"including restaurants for meals, and return your best first."
    )


def _capped(attractions: AttractionsResult, limit: int) -> AttractionsResult:
    """Trim the pool to the size limit, keeping the agent's own ordering.

    The agent is asked for `limit` places and the tools return no more than
    that, but neither is a guarantee: a model can report places it found across
    several calls. Enforcing it here is what makes the limit real.

    Args:
        attractions: What the agent returned.
        limit: The most places the pool may hold.

    Returns:
        The result, with at most `limit` places.
    """
    if len(attractions.places) <= limit:
        return attractions

    logger.info(
        "attractions returned %d places; keeping the first %d",
        len(attractions.places),
        limit,
    )
    return attractions.model_copy(update={"places": attractions.places[:limit]})


def attractions_node(state: TripState, collector=None) -> dict:
    """Graph node: run the Attractions Agent.

    Args:
        state: The shared trip state; reads `intake`.

    Returns:
        A partial state update with the pool of candidate places, never larger
        than `pool_size()`.
    """
    profile = state["intake"].profile
    days = (
        (profile.end_date - profile.start_date).days + 1
        if profile.start_date and profile.end_date
        else 1
    )
    attractions: AttractionsResult = run_agent(
        build_attractions_agent(),
        _attractions_brief(profile, days),
        role="attractions",
        collector=collector,
    )

    return {
        "attractions": _capped(attractions, pool_size(days)),
        "completed_agents": [AgentName.ATTRACTIONS],
    }
