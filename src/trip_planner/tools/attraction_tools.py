"""Tools for the Attractions Agent (plan.md agent #6).

`search_places`, `get_place_details` and `get_opening_hours` read live Google
Maps data through SerpApi, which is what makes coordinates and real opening
hours available - the two things the Routing and Critic agents depend on.
`web_search` falls back to Tavily for anything Maps does not cover, such as
festivals or seasonal events.
"""

from __future__ import annotations

from langchain_core.tools import tool

from trip_planner.tools.research_tools import tavily_web_search
from trip_planner.tools.serp import serp_search

# How many places one search returns.
MAX_PLACES = 10

WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _closed_days(operating_hours: dict) -> list[str]:
    """Return the weekdays a place is closed, capitalized.

    Args:
        operating_hours: The SerpApi `operating_hours` map, day -> hours text.

    Returns:
        Weekday names whose hours read "closed", e.g. ["Monday"].
    """
    return [
        day.capitalize()
        for day in WEEKDAYS
        if "closed" in str(operating_hours.get(day, "")).lower()
    ]


def _summarize_place(index: int, result: dict) -> dict:
    """Reduce one SerpApi Maps result to the fields the agent needs.

    Args:
        index: Position in the result list, used to build a stable place id.
        result: A raw entry from the SerpApi `local_results` array.

    Returns:
        A dict matching the `Place` schema.
    """
    coordinates = result.get("gps_coordinates") or {}
    operating_hours = result.get("operating_hours") or {}
    hours_text = (
        "; ".join(f"{day.capitalize()} {hours}" for day, hours in operating_hours.items())
        or result.get("hours", "")
        or "unknown"
    )
    return {
        "place_id": f"place-{index}",
        "name": result.get("title", ""),
        "category": result.get("type", "attraction"),
        "categories": result.get("types", []),
        "address": result.get("address", ""),
        "coordinates": {
            "latitude": coordinates.get("latitude"),
            "longitude": coordinates.get("longitude"),
        }
        if coordinates
        else None,
        "opening_hours": hours_text,
        "closed_days": _closed_days(operating_hours),
        "rating": result.get("rating"),
        "reviews": result.get("reviews"),
        "price_level": result.get("price", ""),
        "description": result.get("description", ""),
        "website": result.get("website", ""),
        "source_url": result.get("website") or result.get("place_id_search", ""),
        "maps_place_id": result.get("place_id", ""),
    }


@tool
def search_places(query: str, destination: str) -> dict:
    """Search real places at a destination: attractions, restaurants, activities.

    Returns coordinates and opening hours, which later agents need to build a
    workable route. Search one interest at a time, e.g. "museums" then
    "seafood restaurants", rather than everything at once.

    Args:
        query: What to look for, e.g. "art museums" or "rooftop bars".
        destination: Where to look, e.g. "Lisbon, Portugal".

    Returns:
        A dict with `places` (name, category, address, coordinates, hours,
        rating), or `error` when the lookup failed.
    """
    response = serp_search(
        engine="google_maps",
        q=f"{query} in {destination}",
        type="search",
        hl="en",
    )
    if "error" in response:
        return {"error": response["error"], "places": []}

    results = response.get("local_results", [])
    places = [
        _summarize_place(index, result)
        for index, result in enumerate(results[:MAX_PLACES], start=1)
    ]
    if not places:
        return {"error": f"No places found for '{query}' in {destination}.", "places": []}
    return {"places": places}


@tool
def get_place_details(place_name: str, destination: str) -> dict:
    """Look up one specific place in detail.

    Use this when a place came from a web search rather than `search_places`
    and is missing coordinates or hours.

    Args:
        place_name: The exact name of the place.
        destination: The city it is in, e.g. "Lisbon, Portugal".

    Returns:
        The place's address, coordinates, hours, rating and website, or
        `error` when it could not be found.
    """
    response = serp_search(
        engine="google_maps",
        q=f"{place_name} {destination}",
        type="search",
        hl="en",
    )
    if "error" in response:
        return {"error": response["error"]}

    results = response.get("local_results", [])
    if not results:
        place = response.get("place_results")
        if not place:
            return {"error": f"'{place_name}' was not found in {destination}."}
        return _summarize_place(1, place)
    return _summarize_place(1, results[0])


@tool
def get_opening_hours(place_name: str, destination: str) -> dict:
    """Look up when a place is open, day by day.

    Args:
        place_name: The exact name of the place.
        destination: The city it is in, e.g. "Lisbon, Portugal".

    Returns:
        A dict with `opening_hours` (per weekday), `closed_days` and the
        current `open_state`, or `error` when it could not be found.
    """
    response = serp_search(
        engine="google_maps",
        q=f"{place_name} {destination} opening hours",
        type="search",
        hl="en",
    )
    if "error" in response:
        return {"error": response["error"]}

    results = response.get("local_results") or []
    record = results[0] if results else response.get("place_results")
    if not record:
        return {"error": f"No hours found for '{place_name}' in {destination}."}

    operating_hours = record.get("operating_hours") or {}
    if not operating_hours:
        return {
            "name": record.get("title", place_name),
            "opening_hours": record.get("hours", "unknown"),
            "closed_days": [],
            "note": "Google Maps did not report per-day hours for this place.",
        }
    return {
        "name": record.get("title", place_name),
        "opening_hours": operating_hours,
        "closed_days": _closed_days(operating_hours),
        "open_state": record.get("open_state", ""),
    }


@tool
def web_search(query: str) -> dict:
    """Search the web for things Google Maps does not list.

    Use this for seasonal events, festivals, ticket prices and local advice.
    For a physical place, prefer `search_places`, which returns coordinates.

    Args:
        query: A specific query, e.g. "Lisbon festivals September 2026".

    Returns:
        A summarized answer plus the sources it came from.
    """
    return tavily_web_search(query)


ATTRACTION_TOOLS = [web_search, search_places, get_opening_hours, get_place_details]
