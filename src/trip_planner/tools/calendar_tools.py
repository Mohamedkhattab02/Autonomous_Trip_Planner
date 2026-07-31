"""Tools for the Calendar Agent (plan.md agent #11).

Events are collected in an in-memory calendar and written out as a real `.ics`
file, which every calendar app - Google Calendar included - can import. That
keeps the export working without an OAuth flow; a Google Calendar API client
can be dropped in behind `create_calendar_event` later without changing the
agent or the schema.

The store is module-level and reset per export, because a graph run builds one
calendar and writes it once.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from icalendar import Calendar, Event
from langchain_core.tools import tool

# Where .ics files are written, relative to the project root.
EXPORT_DIR = Path(__file__).resolve().parents[3] / "exports"

# Events staged for the current export, keyed by event id.
_EVENTS: dict[str, dict] = {}


def reset_calendar() -> None:
    """Clear the staged events.

    Called by the Calendar node before the agent runs, so events from an
    earlier run never leak into a new export.
    """
    _EVENTS.clear()


def _parse_datetime(value: str) -> datetime | None:
    """Parse an ISO 8601 datetime, tolerating a space instead of 'T'.

    Args:
        value: The datetime text, e.g. "2026-09-10T10:00:00".

    Returns:
        The parsed datetime, or None when it is not valid.
    """
    try:
        return datetime.fromisoformat(str(value).strip().replace(" ", "T"))
    except (ValueError, AttributeError):
        return None


@tool
def create_calendar_event(
    event_id: str,
    title: str,
    start: str,
    end: str,
    location: str = "",
    description: str = "",
    category: str = "activity",
) -> dict:
    """Add one event to the trip calendar.

    Call this once per flight, hotel stay and scheduled activity.

    Args:
        event_id: A short unique id, e.g. "event-1". Reusing an id replaces
            that event.
        title: Event title as it should appear in the calendar.
        start: Local start time, ISO 8601, e.g. "2026-09-10T10:00:00".
        end: Local end time, ISO 8601.
        location: Where the event takes place.
        description: Details for the event body.
        category: What it is: "flight", "lodging" or "activity".

    Returns:
        A dict confirming the event, or `error` when the times are invalid.
    """
    parsed_start = _parse_datetime(start)
    parsed_end = _parse_datetime(end)

    if parsed_start is None or parsed_end is None:
        return {
            "error": (
                f"Could not read the times for '{title}'. Use ISO 8601, "
                f"e.g. 2026-09-10T10:00:00."
            )
        }
    if parsed_end <= parsed_start:
        return {"error": f"'{title}' ends at or before it starts ({start} to {end})."}

    _EVENTS[event_id] = {
        "event_id": event_id,
        "title": title,
        "start": parsed_start.isoformat(),
        "end": parsed_end.isoformat(),
        "location": location,
        "description": description,
        "category": category,
    }
    return {"created": event_id, "title": title, "total_events": len(_EVENTS)}


@tool
def update_event(
    event_id: str,
    title: str | None = None,
    start: str | None = None,
    end: str | None = None,
    location: str | None = None,
    description: str | None = None,
) -> dict:
    """Change an event that has already been added.

    Args:
        event_id: The id of the event to change.
        title: New title, if it should change.
        start: New start time, ISO 8601, if it should change.
        end: New end time, ISO 8601, if it should change.
        location: New location, if it should change.
        description: New description, if it should change.

    Returns:
        The updated event, or `error` when the id is unknown or times invalid.
    """
    event = _EVENTS.get(event_id)
    if event is None:
        return {
            "error": f"No event with id '{event_id}'. Known ids: "
            f"{', '.join(sorted(_EVENTS)) or 'none'}."
        }

    if start is not None:
        parsed = _parse_datetime(start)
        if parsed is None:
            return {"error": f"'{start}' is not a valid ISO 8601 datetime."}
        event["start"] = parsed.isoformat()
    if end is not None:
        parsed = _parse_datetime(end)
        if parsed is None:
            return {"error": f"'{end}' is not a valid ISO 8601 datetime."}
        event["end"] = parsed.isoformat()

    if datetime.fromisoformat(event["end"]) <= datetime.fromisoformat(event["start"]):
        return {"error": f"'{event_id}' would end at or before it starts."}

    if title is not None:
        event["title"] = title
    if location is not None:
        event["location"] = location
    if description is not None:
        event["description"] = description

    return {"updated": event_id, "event": event}


@tool
def export_ics(filename: str = "trip.ics") -> dict:
    """Write every staged event to a .ics file the traveler can import.

    Call this once, after all the events have been created. The file imports
    into Google Calendar, Apple Calendar and Outlook.

    Args:
        filename: Name of the file to write, e.g. "lisbon-trip.ics".

    Returns:
        A dict with the `path` written and the `event_count`, or `error` when
        there is nothing to export.
    """
    if not _EVENTS:
        return {"error": "No events have been created, so there is nothing to export."}

    calendar = Calendar()
    calendar.add("prodid", "-//Autonomous Trip Planner//EN")
    calendar.add("version", "2.0")

    for event in sorted(_EVENTS.values(), key=lambda entry: entry["start"]):
        entry = Event()
        entry.add("uid", event["event_id"])
        entry.add("summary", event["title"])
        entry.add("dtstart", datetime.fromisoformat(event["start"]))
        entry.add("dtend", datetime.fromisoformat(event["end"]))
        if event.get("location"):
            entry.add("location", event["location"])
        if event.get("description"):
            entry.add("description", event["description"])
        entry.add("categories", [event.get("category", "activity")])
        calendar.add_component(entry)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not filename.endswith(".ics"):
        filename = f"{filename}.ics"
    path = EXPORT_DIR / filename
    path.write_bytes(calendar.to_ical())

    return {
        "path": str(path),
        "event_count": len(_EVENTS),
        "message": f"Wrote {len(_EVENTS)} events to {path}.",
    }


CALENDAR_TOOLS = [create_calendar_event, update_event, export_ics]
