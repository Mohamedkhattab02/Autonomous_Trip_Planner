"""Destination Research Agent (plan.md agent #3).

Researches the destination with real web search and returns a cited summary
covering weather, safety, currency, transport and entry requirements.
"""

from __future__ import annotations

from trip_planner.agents.factory import build_structured_agent
from trip_planner.schemas import AgentName, DestinationResearch, TravelerProfile
from trip_planner.state import TripState
from trip_planner.tools import RESEARCH_TOOLS

SYSTEM_PROMPT = """You are the Destination Research Agent of an autonomous trip planner.

Your job is to research a destination and report reliable, cited facts.

Use your tools to cover every one of these areas before answering:
- `get_weather` for the weather during the traveler's dates.
- `get_entry_requirements` for visa and passport rules.
- `get_currency_info` for the local currency and payment norms.
- `tavily_search` for safety, and again for local transportation.

Rules:
- Ground every field in tool results. Never answer from memory alone.
- Populate `sources` with the titles and URLs the tools returned. Every source
  you list must be one a tool actually gave you; never invent a URL.
- If a search returns nothing useful for a field, say so plainly in that field
  rather than guessing.
- Keep each field to a few sentences of practical, specific information.
"""


def build_research_agent():
    """Build the Destination Research Agent runnable."""
    return build_structured_agent(
        role="destination_research",
        tools=RESEARCH_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=DestinationResearch,
        name="destination_research_agent",
    )


def _research_brief(profile: TravelerProfile) -> str:
    """Turn the traveler profile into the research request for the agent.

    Args:
        profile: The profile produced by the Intake Agent.

    Returns:
        A prompt describing what to research.
    """
    dates = (
        f"{profile.start_date} to {profile.end_date}"
        if profile.start_date and profile.end_date
        else "dates not yet fixed"
    )
    interests = ", ".join(profile.interests) if profile.interests else "none stated"
    return (
        f"Research this destination: {profile.destination}\n"
        f"Travel dates: {dates}\n"
        f"Travelers: {profile.travelers or 'unspecified'}\n"
        f"Interests: {interests}\n\n"
        "Cover weather, safety, currency, local transportation and entry requirements."
    )


def research_node(state: TripState, collector=None) -> dict:
    """Graph node: run the Destination Research Agent.

    Args:
        state: The shared trip state; reads `intake`.

    Returns:
        A partial state update with the destination research.
    """
    profile = state["intake"].profile
    agent = build_research_agent()
    if collector is not None:
        # Route this agent's token and tool usage into the run metrics.
        agent = agent.with_config(callbacks=[collector])
    result = agent.invoke(
        {"messages": [{"role": "user", "content": _research_brief(profile)}]}
    )
    research: DestinationResearch = result["structured_response"]

    return {
        "research": research,
        "completed_agents": [AgentName.DESTINATION_RESEARCH],
    }
