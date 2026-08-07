"""Calendar Agent (plan.md agent #11).

Puts the approved trip on the traveler's calendar.

This agent has two interchangeable backends, and does not care which one it
gets:

* **Google Calendar over MCP** - the tools are discovered at runtime from an
  MCP server that owns the OAuth flow and the Google API calls. Events land in
  the traveler's real calendar.
* **Local `.ics` export** - the fallback when no MCP server is configured.

Because the tool names differ between backends, the system prompt is built from
whatever tools are actually loaded rather than hard-coding names. That is what
makes the agent backend-agnostic: it is told the job and the tools it has, and
works out the calls itself.
"""

from __future__ import annotations

import re

from trip_planner.agents.factory import (
    AgentOutputError,
    agent_config,
    build_structured_agent,
)
from langchain_core.tools import BaseTool

from trip_planner.mcp_client import load_calendar_tools, run_async
from trip_planner.schemas import (
    AgentName,
    CalendarResult,
    FlightsResult,
    ItineraryResult,
    LodgingResult,
    TravelerProfile,
)
from trip_planner.state import TripState
from trip_planner.tools.calendar_tools import reset_calendar

BASE_PROMPT = """You are the Calendar Agent of an autonomous trip planner.

Your job is to put the approved trip on the traveler's calendar: one event per
flight, one for the hotel stay, and one for every scheduled activity.

Rules that hold whatever tools you are given:
- Create events only for things that are actually in the plan. Never invent an
  event, and never guess a flight time that was not given to you.
- Times are local to the destination. Build each one by combining the day's
  date with the stop's time, in ISO 8601: "2026-09-10T16:05:00".
- Put the address in the event's location field so the traveler's phone can
  navigate to it.
- Mark each event's category in your answer: "flight", "lodging" or "activity".
- If a tool call fails, say so plainly in `reasoning` rather than claiming the
  event was created.
"""

MCP_PROMPT = """
You are connected to the traveler's real Google Calendar through an MCP server.
These are the tools it offers:

{tool_list}

How to use them:
- If you are unsure which calendar to write to, list the available calendars
  first, then use the traveler's primary calendar.
- When a tool asks for a `calendarId`, "primary" is the traveler's main calendar.
- When a tool asks for a time zone, use the IANA zone of the destination
  (Lisbon -> "Europe/Lisbon", Rome -> "Europe/Rome", Tokyo -> "Asia/Tokyo").
- Create the events one at a time and check each result before moving on.
- Leave `ics_path` null: the events live in Google Calendar, not in a file.
- In `events`, report exactly what you successfully created.
"""

LOCAL_PROMPT = """
No Google Calendar connection is configured, so you are writing a calendar file
the traveler can import. These are your tools:

{tool_list}

How to use them:
- Call `create_calendar_event` once per flight, stay and activity.
- Give every event a unique `event_id` such as "event-1", "event-2".
- Call `export_ics` once at the very end, after every event exists, and put the
  path it returns in `ics_path`.
"""


def _tool_list(tools: list[BaseTool]) -> str:
    """Render the available tools as a bulleted list for the prompt.

    Args:
        tools: The tools the agent has been given.

    Returns:
        One line per tool, with its name and description.
    """
    lines = []
    for tool in tools:
        description = (tool.description or "").strip().splitlines()
        summary = description[0] if description else ""
        lines.append(f"- `{tool.name}`: {summary}")
    return "\n".join(lines) or "- (no tools available)"


def build_calendar_agent(tools: list[BaseTool], using_mcp: bool):
    """Build the Calendar Agent around whichever backend is active.

    Args:
        tools: The calendar tools, from MCP or the local fallback.
        using_mcp: True when `tools` came from the Google Calendar MCP server.

    Returns:
        The agent runnable.
    """
    backend_prompt = MCP_PROMPT if using_mcp else LOCAL_PROMPT
    system_prompt = BASE_PROMPT + backend_prompt.format(tool_list=_tool_list(tools))

    return build_structured_agent(
        role="calendar",
        tools=tools,
        system_prompt=system_prompt,
        response_format=CalendarResult,
        name="calendar_agent",
    )


def _slugify(text: str) -> str:
    """Turn a destination into a safe filename stem.

    Args:
        text: The text to convert, e.g. "Lisbon, Portugal".

    Returns:
        A lowercase, hyphenated slug, e.g. "lisbon-portugal".
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "trip"


def _flight_events(flights: FlightsResult | None) -> str:
    """Describe the chosen flight's legs as calendar-ready lines.

    Args:
        flights: The Flights Agent's output, when it ran.

    Returns:
        One line per leg, for the prompt.
    """
    if flights is None or not flights.options:
        return "- (no flights were booked; create no flight events)"

    chosen = next(
        (
            option
            for option in flights.options
            if option.option_id == flights.recommended_option_id
        ),
        flights.options[0],
    )
    return (
        "\n".join(
            f"- {leg.airline} {leg.flight_number}: {leg.departure_airport} "
            f"{leg.departure_time} -> {leg.arrival_airport} {leg.arrival_time}"
            for leg in chosen.legs
        )
        or "- (the chosen flight has no legs listed)"
    )


def _lodging_event(lodging: LodgingResult | None, profile: TravelerProfile) -> str:
    """Describe the chosen stay as a calendar-ready line.

    Args:
        lodging: The Lodging Agent's output, when it ran.
        profile: The traveler profile, for the check-in and check-out dates.

    Returns:
        A single line for the prompt.
    """
    if lodging is None or not lodging.options:
        return "- (no lodging was booked; create no lodging event)"

    chosen = next(
        (
            option
            for option in lodging.options
            if option.option_id == lodging.recommended_option_id
        ),
        lodging.options[0],
    )
    return (
        f"- {chosen.name}, {chosen.address or 'address not recorded'}: "
        f"check in {profile.start_date} 15:00, check out {profile.end_date} 11:00"
    )


def _activity_events(itinerary: ItineraryResult | None) -> str:
    """Describe every scheduled activity as calendar-ready lines.

    Args:
        itinerary: The final itinerary.

    Returns:
        One line per stop, for the prompt.
    """
    if itinerary is None or not itinerary.days:
        return "- (the itinerary has no days)"

    lines = []
    for day in itinerary.days:
        for stop in day.stops:
            lines.append(
                f"- {day.date} {stop.start_time}-{stop.end_time}: {stop.name}"
                + (f" ({stop.notes})" if stop.notes else "")
            )
    return "\n".join(lines) or "- (no activities are scheduled)"


def _calendar_brief(
    profile: TravelerProfile,
    itinerary: ItineraryResult | None,
    flights: FlightsResult | None,
    lodging: LodgingResult | None,
    filename: str,
    using_mcp: bool,
) -> str:
    """Build the calendar export request.

    Args:
        profile: The traveler profile.
        itinerary: The final itinerary.
        flights: The Flights Agent's output.
        lodging: The Lodging Agent's output.
        filename: The .ics filename to write, when exporting locally.
        using_mcp: Whether the events go to Google Calendar.

    Returns:
        A prompt listing every event to create.
    """
    target = (
        "Add these to the traveler's Google Calendar."
        if using_mcp
        else f"Put these on the calendar and export them as '{filename}'."
    )
    return (
        f"{target}\n"
        f"Destination: {profile.destination}\n"
        f"Dates: {profile.start_date} to {profile.end_date}\n\n"
        f"Flights:\n{_flight_events(flights)}\n\n"
        f"Lodging:\n{_lodging_event(lodging, profile)}\n\n"
        f"Activities:\n{_activity_events(itinerary)}"
    )


def calendar_node(state: TripState, collector=None) -> dict:
    """Graph node: run the Calendar Agent.

    The agent is invoked asynchronously because MCP tools are async-native;
    `run_async` bridges that back to the sync graph.

    Args:
        state: The shared trip state; reads `intake`, `itinerary`, `flights`
            and `lodging`.

    Returns:
        A partial state update with the exported calendar.
    """
    profile = state["intake"].profile
    filename = f"{_slugify(profile.destination or 'trip')}.ics"

    tools, using_mcp = load_calendar_tools()
    if not using_mcp:
        # Start from an empty calendar so a re-run never merges with old events.
        reset_calendar()

    agent = build_calendar_agent(tools, using_mcp)
    brief = _calendar_brief(
        profile,
        state.get("itinerary"),
        state.get("flights"),
        state.get("lodging"),
        filename,
        using_mcp,
    )
    # Invoked here rather than through `run_agent` because MCP tools are
    # async-native; the config it would have built is used unchanged.
    result = run_async(
        agent.ainvoke(
            {"messages": [{"role": "user", "content": brief}]},
            agent_config("calendar", collector),
        )
    )

    calendar: CalendarResult | None = result.get("structured_response")
    if calendar is None:
        raise AgentOutputError(
            "the calendar agent finished without reporting the events it "
            "created, so there is nothing to record for this stage"
        )

    return {
        "calendar": calendar,
        "completed_agents": [AgentName.CALENDAR],
    }
