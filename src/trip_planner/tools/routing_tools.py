"""Tools for the Routing Agent (plan.md agent #7).

Every tool here is deterministic Python. Distances, clustering and opening
hours are arithmetic and lookups, and a model asked to eyeball them produces
itineraries that criss-cross the city. The agent decides which places matter
and in what order; these tools decide what is geometrically possible.
"""

from __future__ import annotations

from datetime import date as date_type

from langchain_core.tools import tool
from sklearn.cluster import KMeans

from trip_planner.tools.geo import (
    haversine_km,
    suggest_mode,
    travel_minutes,
)

WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


@tool
def calculate_distance(
    from_latitude: float,
    from_longitude: float,
    to_latitude: float,
    to_longitude: float,
) -> dict:
    """Measure the distance between two points.

    Args:
        from_latitude: Latitude of the starting point.
        from_longitude: Longitude of the starting point.
        to_latitude: Latitude of the destination.
        to_longitude: Longitude of the destination.

    Returns:
        A dict with `distance_km` and the `suggested_mode` for covering it.
    """
    km = haversine_km(from_latitude, from_longitude, to_latitude, to_longitude)
    return {
        "distance_km": round(km, 2),
        "suggested_mode": suggest_mode(km),
    }


@tool
def calculate_travel_time(distance_km: float, mode: str = "walking") -> dict:
    """Estimate how long it takes to cover a distance in a city.

    Args:
        distance_km: The distance to cover, from `calculate_distance`.
        mode: How to travel: "walking", "transit", "taxi" or "driving".

    Returns:
        A dict with `travel_minutes` and the `mode` used.
    """
    return {
        "travel_minutes": travel_minutes(distance_km, mode),
        "mode": mode,
        "distance_km": distance_km,
    }


@tool
def cluster_locations(places: list[dict], days: int) -> dict:
    """Group places into geographic clusters, one per day of the trip.

    Uses k-means on the coordinates so each day stays in one part of the city
    instead of criss-crossing it. Call this before assigning times.

    Args:
        places: Places to group. Each needs `place_id`, `name`, `latitude`
            and `longitude`.
        days: How many days to split them across.

    Returns:
        A dict with `clusters` (one per day, each holding its places, center
        and span in km) and `unassigned` for places lacking coordinates.
    """
    located = [
        place
        for place in places
        if place.get("latitude") is not None and place.get("longitude") is not None
    ]
    unassigned = [
        place.get("place_id", "")
        for place in places
        if place.get("latitude") is None or place.get("longitude") is None
    ]
    if not located:
        return {"clusters": [], "unassigned": unassigned, "note": "No located places."}

    # k-means needs at least as many points as clusters.
    cluster_count = max(1, min(days, len(located)))
    coordinates = [[place["latitude"], place["longitude"]] for place in located]
    labels = (
        KMeans(n_clusters=cluster_count, n_init=10, random_state=0)
        .fit(coordinates)
        .labels_
    )

    clusters = []
    for index in range(cluster_count):
        members = [
            place for place, label in zip(located, labels) if int(label) == index
        ]
        if not members:
            continue
        center_lat = sum(place["latitude"] for place in members) / len(members)
        center_lon = sum(place["longitude"] for place in members) / len(members)
        span = max(
            (
                haversine_km(
                    center_lat, center_lon, place["latitude"], place["longitude"]
                )
                for place in members
            ),
            default=0.0,
        )
        clusters.append(
            {
                "cluster_index": index,
                "places": [
                    {"place_id": place.get("place_id", ""), "name": place.get("name", "")}
                    for place in members
                ],
                "center": {
                    "latitude": round(center_lat, 6),
                    "longitude": round(center_lon, 6),
                },
                "radius_km": round(span, 2),
            }
        )

    # Biggest clusters first, so the agent fills the busiest days first.
    clusters.sort(key=lambda entry: len(entry["places"]), reverse=True)
    return {"clusters": clusters, "unassigned": unassigned}


@tool
def check_opening_hours(place_name: str, closed_days: list[str], visit_date: str) -> dict:
    """Check whether a place is open on the day it is scheduled for.

    Args:
        place_name: Name of the place, for the message.
        closed_days: Weekdays the place is closed, e.g. ["Monday"].
        visit_date: The planned date in YYYY-MM-DD format.

    Returns:
        A dict with `is_open`, the `weekday` of that date, and a `message`.
    """
    try:
        parsed = date_type.fromisoformat(visit_date)
    except ValueError:
        return {
            "is_open": None,
            "message": f"'{visit_date}' is not a valid YYYY-MM-DD date.",
        }

    weekday = WEEKDAYS[parsed.weekday()].capitalize()
    normalized = {day.strip().lower() for day in closed_days}
    is_open = weekday.lower() not in normalized

    return {
        "is_open": is_open,
        "weekday": weekday,
        "message": (
            f"{place_name} is open on {weekday} {visit_date}."
            if is_open
            else f"{place_name} is CLOSED on {weekday} {visit_date}. Move it to another day."
        ),
    }


ROUTING_TOOLS = [
    calculate_distance,
    calculate_travel_time,
    cluster_locations,
    check_opening_hours,
]


def _parse_window(text: str) -> tuple[int, int] | None:
    """Read an opening-hours string into minutes-since-midnight bounds.

    Args:
        text: Hours text as Google reports it, e.g. "10 AM-6 PM" or "10:00-18:00".

    Returns:
        An `(open, close)` pair in minutes, or None when it cannot be read.
    """
    import re

    if not text or "closed" in text.lower():
        return None
    if "24 hours" in text.lower() or "open 24" in text.lower():
        return (0, 24 * 60)

    matches = re.findall(r"(\d{1,2})(?::(\d{2}))?\s*([AaPp][Mm])?", text)
    bounds = []
    for hour, minute, meridiem in matches:
        value = int(hour) % 12 * 60 + int(minute or 0)
        if (meridiem or "").lower() == "pm":
            value += 12 * 60
        elif not meridiem and int(hour) >= 13:
            value = int(hour) * 60 + int(minute or 0)
        bounds.append(value)
        if len(bounds) == 2:
            break

    if len(bounds) != 2 or bounds[1] <= bounds[0]:
        return None
    return (bounds[0], bounds[1])


@tool
def schedule_day(
    places: list[dict],
    visit_date: str,
    day_start: str = "09:00",
    day_end: str = "21:00",
) -> dict:
    """Order and time one day's places so the schedule is actually feasible.

    Greedy insertion under time windows: repeatedly take the place whose
    opening hours are tightest, place it at the earliest feasible slot, and
    account for the travel time to reach it. This is deterministic arithmetic —
    the agent chooses *which* places matter, this decides what is possible.

    Args:
        places: Places to schedule. Each needs `place_id`, `name`, `latitude`,
            `longitude` and `visit_duration_minutes`; `opening_hours` and
            `closed_days` are respected when present.
        visit_date: The day in YYYY-MM-DD format, used for closing days.
        day_start: Earliest the traveler will start, "HH:MM".
        day_end: Latest the traveler will finish, "HH:MM".

    Returns:
        A dict with `stops` (ordered, with start/end times and travel gaps) and
        `unscheduled` — places that could not be fitted, with the reason.
    """

    def to_minutes(text: str) -> int:
        hour, _, minute = text.partition(":")
        return int(hour) * 60 + int(minute or 0)

    def to_text(value: int) -> str:
        return f"{value // 60:02d}:{value % 60:02d}"

    try:
        weekday = WEEKDAYS[date_type.fromisoformat(visit_date).weekday()].capitalize()
    except ValueError:
        return {"stops": [], "unscheduled": [], "error": f"'{visit_date}' is not a date."}

    open_from, close_at = to_minutes(day_start), to_minutes(day_end)

    candidates, unscheduled = [], []
    for place in places:
        if weekday in [d.capitalize() for d in place.get("closed_days", [])]:
            unscheduled.append(
                {"place_id": place.get("place_id"), "reason": f"closed on {weekday}"}
            )
            continue
        window = _parse_window(str(place.get("opening_hours", ""))) or (open_from, close_at)
        candidates.append({**place, "window": window})

    # Tightest window first: the hardest places to place go first.
    candidates.sort(key=lambda p: p["window"][1] - p["window"][0])

    stops: list[dict] = []
    cursor = open_from
    previous: dict | None = None

    while candidates:
        best, best_start, best_travel, best_mode = None, None, 0, "walking"
        for place in candidates:
            travel, mode = 0, "walking"
            if previous is not None:
                km = haversine_km(
                    previous["latitude"],
                    previous["longitude"],
                    place["latitude"],
                    place["longitude"],
                )
                mode = suggest_mode(km)
                travel = travel_minutes(km, mode)

            start = max(cursor + travel, place["window"][0])
            duration = int(place.get("visit_duration_minutes", 90))
            if start + duration > min(place["window"][1], close_at):
                continue
            if best_start is None or start < best_start:
                best, best_start, best_travel, best_mode = place, start, travel, mode

        if best is None:
            break

        duration = int(best.get("visit_duration_minutes", 90))
        stops.append(
            {
                "place_id": best.get("place_id"),
                "name": best.get("name", ""),
                "start_time": to_text(best_start),
                "end_time": to_text(best_start + duration),
                "travel_minutes_from_previous": best_travel,
                "travel_mode": best_mode,
            }
        )
        cursor = best_start + duration
        previous = best
        candidates.remove(best)

    unscheduled += [
        {"place_id": p.get("place_id"), "reason": "no feasible slot left in the day"}
        for p in candidates
    ]
    return {"stops": stops, "unscheduled": unscheduled, "weekday": weekday}


ROUTING_TOOLS.append(schedule_day)
