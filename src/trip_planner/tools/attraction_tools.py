"""Tools for the Attractions Agent (plan.md agent #6).

`search_places`, `get_place_details` and `get_opening_hours` read live Google
Maps data through SerpApi, which is what makes coordinates and real opening
hours available - the two things the Routing and Critic agents depend on.
`web_search` falls back to Tavily for anything Maps does not cover, such as
festivals or seasonal events.
"""

from __future__ import annotations

import os

from langchain_core.tools import tool

from trip_planner.tools.research_tools import tavily_web_search
from trip_planner.tools.serp import serp_search

# How many places one search returns.
MAX_PLACES = 10

# How many places the whole pool may hold, across every query. Each result
# carries coordinates, hours and a description, so an uncapped bulk search of
# five interests put ~50 of them through the model - paid for once by the
# Attractions Agent and again by Routing, for a pool no trip ever schedules.
DEFAULT_MAX_TOTAL_PLACES = 15


def max_total_places() -> int:
    """How many places the pool may hold in total.

    Read from the environment on every call rather than at import, so a test or
    a run can change the size without reloading the module.

    Returns:
        `TRIP_MAX_PLACES` when it is a positive whole number, else 15.
    """
    raw = os.getenv("TRIP_MAX_PLACES", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_MAX_TOTAL_PLACES

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
    limit = min(MAX_PLACES, max_total_places())
    places = [
        _summarize_place(index, result)
        for index, result in enumerate(results[:limit], start=1)
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


@tool
def search_places_bulk(queries: list[str], destination: str) -> dict:
    """Search several place categories at once, in parallel.

    Prefer this over calling `search_places` repeatedly: the lookups run
    concurrently, so four interests cost roughly one search's worth of time
    instead of four.

    The whole result is capped at `max_total_places()`, and the cap is spent a
    rank at a time across the queries rather than query by query: the best hit
    for every interest is taken before the second-best of any of them. That is
    what keeps a small pool balanced instead of filling it with one interest.

    Args:
        queries: What to look for, one entry per interest, e.g.
            ["art museums", "seafood restaurants", "viewpoints"].
        destination: Where to look, e.g. "Lisbon, Portugal".

    Returns:
        A dict with `places` (deduplicated across every query, each tagged with
        the `matched_query` that found it) and `queries_run`.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not queries:
        return {"places": [], "queries_run": 0}

    def run(query: str) -> tuple[str, dict]:
        response = serp_search(
            engine="google_maps", q=f"{query} in {destination}", type="search", hl="en"
        )
        return query, response

    with ThreadPoolExecutor(max_workers=min(len(queries), 6)) as pool:
        responses = list(pool.map(run, queries))

    ranked = [
        [(query, result) for result in (response.get("local_results") or [])[:MAX_PLACES]]
        for query, response in responses
    ]

    budget = max_total_places()
    places: list[dict] = []
    seen: set[str] = set()
    index = 1
    for rank in range(MAX_PLACES):
        if len(places) >= budget:
            break
        for hits in ranked:
            if len(places) >= budget:
                break
            if rank >= len(hits):
                continue
            query, result = hits[rank]
            key = result.get("place_id") or result.get("title", "")
            if not key or key in seen:
                continue
            seen.add(key)
            place = _summarize_place(index, result)
            place["matched_query"] = query
            places.append(place)
            index += 1

    if not places:
        return {
            "error": f"No places found for any of {queries} in {destination}.",
            "places": [],
            "queries_run": len(queries),
        }
    return {
        "places": places,
        "queries_run": len(queries),
        "note": (
            f"This is the whole pool for this trip, capped at {budget} places. "
            f"Do not search for more."
        ),
    }


ATTRACTION_TOOLS.append(search_places_bulk)
