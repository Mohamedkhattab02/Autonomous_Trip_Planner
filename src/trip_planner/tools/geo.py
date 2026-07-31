"""Shared geography helpers for the lodging and routing tools.

Distances come from `geopy`'s geodesic calculation rather than an API, so
they cost nothing and always work offline. Travel times are estimates from
average city speeds - good enough to catch an itinerary that sends the
traveler across town twice, which is what the Routing Agent needs.
"""

from __future__ import annotations

from geopy.distance import geodesic

# Average door-to-door speeds in km/h, including waiting and walking to stops.
SPEED_KMH: dict[str, float] = {
    "walking": 4.5,
    "transit": 15.0,
    "taxi": 22.0,
    "driving": 25.0,
}

# Beyond this, walking stops being reasonable and transit is assumed instead.
MAX_WALKING_KM = 2.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the straight-line distance between two points in kilometers.

    Args:
        lat1: Latitude of the first point.
        lon1: Longitude of the first point.
        lat2: Latitude of the second point.
        lon2: Longitude of the second point.

    Returns:
        The geodesic distance in kilometers.
    """
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers


def walking_minutes(distance_km: float) -> int:
    """Return how long `distance_km` takes on foot, in whole minutes.

    Args:
        distance_km: The distance to walk.

    Returns:
        Estimated walking time in minutes.
    """
    return travel_minutes(distance_km, "walking")


def travel_minutes(distance_km: float, mode: str = "walking") -> int:
    """Estimate travel time for a distance and mode, in whole minutes.

    Street routes are longer than straight lines, so the distance is scaled
    by 1.3 - the usual detour factor for city grids.

    Args:
        distance_km: Straight-line distance between the two points.
        mode: One of "walking", "transit", "taxi" or "driving".

    Returns:
        Estimated travel time in minutes, at least 1 for any real distance.
    """
    speed = SPEED_KMH.get(mode, SPEED_KMH["walking"])
    minutes = (distance_km * 1.3) / speed * 60
    return max(1, round(minutes)) if distance_km > 0 else 0


def suggest_mode(distance_km: float) -> str:
    """Pick the sensible way to cover a distance in a city.

    Args:
        distance_km: Straight-line distance between the two points.

    Returns:
        "walking" for short hops, "transit" for anything longer.
    """
    return "walking" if distance_km <= MAX_WALKING_KM else "transit"
