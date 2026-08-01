"""Itinerary Agent (plan.md agent #10).

Merges every agent's output into the single document the traveler actually
reads: the day-by-day plan, the chosen flights and stay, the budget position
and the practical notes.
"""

from __future__ import annotations

from trip_planner.agents.factory import build_structured_agent
from trip_planner.schemas import AgentName, ItineraryResult, TravelerProfile
from trip_planner.state import TripState
from trip_planner.tools import ITINERARY_TOOLS, make_read_agent_results

SYSTEM_PROMPT = """You are the Itinerary Agent of an autonomous trip planner.

Your job is to merge everything the other agents produced into one clear plan
for the traveler. You are the only agent whose output a person reads directly,
so write for them, not for a machine.

Follow these steps:
1. Call `read_agent_results` to get every agent's output.
2. Call `build_daily_itinerary` with the routed days to put them in order.
3. Write the overview, the flight and lodging summaries, and the budget
   summary, in plain language.
4. Turn the research into `practical_notes`: weather to pack for, visa rules,
   currency and any safety advice.
5. Call `format_trip_plan` last and put its output in `markdown`.

Rules:
- Report only what the other agents produced. Never add a place, a price or a
  flight that is not in their results, and never quietly fix a gap by
  inventing something.
- `days` must match the routed plan exactly - same stops, same times.
- Write summaries a traveler can act on: name the airline and times, the hotel
  and what it costs, and what the trip totals against the budget.
- If the Critic left warnings, mention the relevant ones as practical notes so
  the traveler is not surprised by them.
- If a stage produced nothing, leave its summary empty rather than inventing
  content for it.
"""


def build_itinerary_agent(state: TripState):
    """Build the Itinerary Agent runnable.

    Args:
        state: The state this turn should be able to read.

    Returns:
        The agent runnable, with `read_agent_results` bound to `state`.
    """
    return build_structured_agent(
        role="itinerary",
        tools=[make_read_agent_results(state), *ITINERARY_TOOLS],
        system_prompt=SYSTEM_PROMPT,
        response_format=ItineraryResult,
        name="itinerary_agent",
    )


def _itinerary_brief(profile: TravelerProfile, days: int) -> str:
    """Build the write-up request.

    Args:
        profile: The traveler profile.
        days: Number of days the trip covers.

    Returns:
        A prompt describing the document to produce.
    """
    interests = ", ".join(profile.interests) if profile.interests else "general travel"
    return (
        f"Write the final trip plan.\n"
        f"Destination: {profile.destination}\n"
        f"Dates: {profile.start_date} to {profile.end_date} ({days} days)\n"
        f"Travelers: {profile.travelers or 1}\n"
        f"Interests: {interests}\n\n"
        f"Start by calling `read_agent_results` to collect what the other "
        f"agents produced, then build the document from it."
    )


def itinerary_node(state: TripState, collector=None) -> dict:
    """Graph node: run the Itinerary Agent.

    Args:
        state: The shared trip state; reads every agent's output.

    Returns:
        A partial state update with the finished itinerary.
    """
    profile = state["intake"].profile
    days = (
        (profile.end_date - profile.start_date).days + 1
        if profile.start_date and profile.end_date
        else 1
    )

    agent = build_itinerary_agent(state)
    if collector is not None:
        # Route this agent's token and tool usage into the run metrics.
        agent = agent.with_config(callbacks=[collector])
    result = agent.invoke(
        {"messages": [{"role": "user", "content": _itinerary_brief(profile, days)}]}
    )
    itinerary: ItineraryResult = result["structured_response"]

    return {
        "itinerary": itinerary,
        "completed_agents": [AgentName.ITINERARY],
    }
