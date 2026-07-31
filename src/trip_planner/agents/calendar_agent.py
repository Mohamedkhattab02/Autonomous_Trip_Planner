"""Calendar Agent (plan.md agent #11).

Turns the approved itinerary into calendar events and writes them to a `.ics`
file the traveler can import into Google Calendar or any other calendar app.
"""

from __future__ import annotations

import re

from langchain.agents import create_agent

from trip_planner.llm import get_model
from trip_planner.schemas import (
    AgentName,
    CalendarResult,
    FlightsResult,
    ItineraryResult,
    LodgingResult,
    TravelerProfile,
)
from trip_planner.state import TripState
from trip_planner.tools import CALENDAR_TOOLS
from trip_planner.tools.calendar_tools import reset_calendar

SYSTEM_PROMPT = """You are the Calendar Agent of an autonomous trip planner.

Your job is to put the approved plan on the traveler's calendar.

Follow these steps:
1. Call `create_calendar_event` for each flight, using the real departure and
   arrival times, with `category` set to "flight".
2. Call `create_calendar_event` for the hotel stay: check-in on the first day
   and check-out on the last, with `category` set to "lodging".
3. Call `create_calendar_event` for every scheduled activity, using the date
   of its day and its start and end times, with `category` set to "activity".
4. Call `export_ics` once at the end, after every event exists.

Rules:
- Times must be ISO 8601, e.g. "2026-09-10T10:00:00". Combine each day's date
  with the stop's time to build them.
- Create events only for things that are actually in the plan. Never invent an
  event, and never guess a flight time that was not given to you.
- Give every event a unique `event_id` like "event-1", "event-2".
- Put the address in `location` so the traveler's phone can navigate to it.
- If `export_ics` returns an error, report it in `reasoning` rather than
  claiming the export succeeded.
"""


def build_calendar_agent():
    """Build the Calendar Agent runnable."""
    return create_agent(
        model=get_model(),
        tools=CALENDAR_TOOLS,
        system_prompt=SYSTEM_PROMPT,
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
    return "\n".join(
        f"- {leg.airline} {leg.flight_number}: {leg.departure_airport} "
        f"{leg.departure_time} -> {leg.arrival_airport} {leg.arrival_time}"
        for leg in chosen.legs
    ) or "- (the chosen flight has no legs listed)"


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
) -> str:
    """Build the calendar export request.

    Args:
        profile: The traveler profile.
        itinerary: The final itinerary.
        flights: The Flights Agent's output.
        lodging: The Lodging Agent's output.
        filename: The .ics filename to write.

    Returns:
        A prompt listing every event to create.
    """
    return (
        f"Put this approved trip on the calendar and export it as '{filename}'.\n"
        f"Destination: {profile.destination}\n"
        f"Dates: {profile.start_date} to {profile.end_date}\n\n"
        f"Flights:\n{_flight_events(flights)}\n\n"
        f"Lodging:\n{_lodging_event(lodging, profile)}\n\n"
        f"Activities:\n{_activity_events(itinerary)}"
    )


def calendar_node(state: TripState) -> dict:
    """Graph node: run the Calendar Agent.

    Args:
        state: The shared trip state; reads `intake`, `itinerary`, `flights`
            and `lodging`.

    Returns:
        A partial state update with the exported calendar.
    """
    profile = state["intake"].profile
    filename = f"{_slugify(profile.destination or 'trip')}.ics"

    # Start from an empty calendar so a re-run never merges with old events.
    reset_calendar()

    agent = build_calendar_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": _calendar_brief(
                        profile,
                        state.get("itinerary"),
                        state.get("flights"),
                        state.get("lodging"),
                        filename,
                    ),
                }
            ]
        }
    )
    calendar: CalendarResult = result["structured_response"]

    return {
        "calendar": calendar,
        "completed_agents": [AgentName.CALENDAR],
    }
