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
