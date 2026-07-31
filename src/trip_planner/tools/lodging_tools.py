"""Tools for the Lodging Agent (plan.md agent #5).

`search_hotels` and `get_hotel_details` read live Google Hotels data through
SerpApi. `check_hotel_location` and `compare_hotels` are plain Python: how far
a hotel sits from the places the traveler will visit, and how options rank on
price and rating, are both arithmetic the model should not be guessing at.
"""

from __future__ import annotations

from langchain_core.tools import tool

from trip_planner.tools.geo import haversine_km, walking_minutes
from trip_planner.tools.serp import serp_search

# How many stays to hand the agent.
MAX_HOTEL_OPTIONS = 8

# Weights for `compare_hotels`: price against the cheapest, offset by rating.
PRICE_WEIGHT = 1.0
RATING_WEIGHT = 0.6


def _summarize_property(index: int, prop: dict) -> dict:
    """Reduce one SerpApi hotel property to the fields the agent needs.

    Args:
        index: Position in the result list, used to build a stable option id.
        prop: A raw entry from the SerpApi `properties` array.

    Returns:
        A dict matching the `LodgingOption` schema.
    """
    coordinates = prop.get("gps_coordinates") or {}
    rate = prop.get("rate_per_night") or {}
    total = prop.get("total_rate") or {}
    nearby = [
        f"{place.get('name', '')}"
        + (
            f" ({place['transportations'][0].get('duration', '')} "
            f"{place['transportations'][0].get('type', '').lower()})"
            if place.get("transportations")
            else ""
        )
        for place in (prop.get("nearby_places") or [])[:4]
    ]
    return {
        "option_id": f"stay-{index}",
        "name": prop.get("name", ""),
        "kind": prop.get("type", "hotel"),
        "description": prop.get("description", ""),
        "coordinates": {
            "latitude": coordinates.get("latitude"),
            "longitude": coordinates.get("longitude"),
        }
        if coordinates
        else None,
        "price_per_night": rate.get("extracted_lowest"),
        "total_price": total.get("extracted_lowest"),
        "rating": prop.get("overall_rating"),
        "reviews": prop.get("reviews"),
        "stars": prop.get("extracted_hotel_class"),
        "amenities": (prop.get("amenities") or [])[:8],
        "nearby_places": nearby,
        "link": prop.get("link", ""),
        "property_token": prop.get("property_token", ""),
    }


@tool
def search_hotels(
    destination: str,
    check_in_date: str,
    check_out_date: str,
    travelers: int = 2,
    currency: str = "USD",
    max_price_per_night: float | None = None,
    min_rating: float | None = None,
) -> dict:
    """Search real hotels and apartments at a destination for the given dates.

    Args:
        destination: Where to stay, e.g. "Lisbon, Portugal". A neighborhood
            works too, e.g. "Alfama, Lisbon".
        check_in_date: Arrival date in YYYY-MM-DD format.
        check_out_date: Departure date in YYYY-MM-DD format.
        travelers: Number of adult guests.
        currency: ISO 4217 currency code for the prices.
        max_price_per_night: Optional nightly price ceiling.
        min_rating: Optional minimum guest rating, out of 5.

    Returns:
        A dict with `options` (name, prices, rating, coordinates, amenities)
        and `currency`, or `error` when the lookup failed.
    """
    params = {
        "engine": "google_hotels",
        "q": destination,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": travelers,
        "currency": currency,
        "hl": "en",
    }
    if max_price_per_night:
        params["max_price"] = int(max_price_per_night)
    if min_rating:
        # SerpApi encodes rating filters as 7 = 3.5+, 8 = 4.0+, 9 = 4.5+.
        params["rating"] = {3.5: "7", 4.0: "8", 4.5: "9"}.get(round(min_rating * 2) / 2)

    response = serp_search(**{k: v for k, v in params.items() if v is not None})
    if "error" in response:
        return {"error": response["error"], "options": []}

    properties = response.get("properties", [])
    options = [
        _summarize_property(index, prop)
        for index, prop in enumerate(properties[:MAX_HOTEL_OPTIONS], start=1)
    ]
    if not options:
        return {
            "error": f"No stays found in {destination} for those dates.",
            "options": [],
        }
    return {"options": options, "currency": currency}


@tool
def get_hotel_details(property_token: str, check_in_date: str, check_out_date: str) -> dict:
    """Fetch the full record for one hotel, including rates and amenities.

    Use this on a shortlisted stay when `search_hotels` did not report enough
    to justify the recommendation.

    Args:
        property_token: The `property_token` from `search_hotels`.
        check_in_date: Arrival date in YYYY-MM-DD format.
        check_out_date: Departure date in YYYY-MM-DD format.

    Returns:
        A dict with the hotel's name, prices, amenities and nearby places, or
        `error` when the lookup failed.
    """
    response = serp_search(
        engine="google_hotels",
        property_token=property_token,
        check_in_date=check_in_date,
        check_out_date=check_out_date,
        hl="en",
    )
    if "error" in response:
        return {"error": response["error"]}

    return {
        "name": response.get("name", ""),
        "description": response.get("description", ""),
        "address": response.get("address", ""),
        "rating": response.get("overall_rating"),
        "reviews": response.get("reviews"),
        "prices": (response.get("prices") or [])[:5],
        "amenities": (response.get("amenities") or [])[:12],
        "nearby_places": (response.get("nearby_places") or [])[:5],
        "check_in_time": response.get("check_in_time", ""),
        "check_out_time": response.get("check_out_time", ""),
    }


@tool
def check_hotel_location(
    hotel_latitude: float,
    hotel_longitude: float,
    reference_points: list[dict],
) -> dict:
    """Measure how far a hotel is from the places the traveler cares about.

    Call this on shortlisted stays to check a hotel is actually well placed,
    rather than trusting its marketing description.

    Args:
        hotel_latitude: The hotel's latitude.
        hotel_longitude: The hotel's longitude.
        reference_points: Places to measure against. Each needs `name`,
            `latitude` and `longitude`.

    Returns:
        A dict with `distances` (each point with its km and walking minutes)
        and `average_km`, so options can be compared on centrality.
    """
    if not reference_points:
        return {"distances": [], "average_km": None, "note": "No reference points given."}

    distances = []
    for point in reference_points:
        km = haversine_km(
            hotel_latitude,
            hotel_longitude,
            point["latitude"],
            point["longitude"],
        )
        distances.append(
            {
                "name": point.get("name", ""),
                "distance_km": round(km, 2),
                "walking_minutes": walking_minutes(km),
            }
        )

    distances.sort(key=lambda entry: entry["distance_km"])
    average = sum(entry["distance_km"] for entry in distances) / len(distances)
    return {"distances": distances, "average_km": round(average, 2)}


@tool
def compare_hotels(options: list[dict]) -> dict:
    """Rank lodging options by total price against guest rating.

    Args:
        options: Options from `search_hotels`. Each needs `option_id` and
            `total_price`; `rating` is used when present.

    Returns:
        A dict with `ranked` (option ids best first, with their scores) and
        `best_option_id`.
    """
    usable = [option for option in options if option.get("total_price")]
    if not usable:
        return {"ranked": [], "best_option_id": None, "note": "No priced options given."}

    cheapest = min(option["total_price"] for option in usable)
    best_rating = max((option.get("rating") or 0) for option in usable) or 1

    scored = []
    for option in usable:
        rating = option.get("rating") or 0
        # Lower is better: price penalty over the cheapest, minus a rating credit.
        score = PRICE_WEIGHT * (option["total_price"] / cheapest - 1) - RATING_WEIGHT * (
            rating / best_rating
        )
        scored.append(
            {
                "option_id": option.get("option_id", ""),
                "name": option.get("name", ""),
                "score": round(score, 4),
                "total_price": option["total_price"],
                "rating": rating or None,
            }
        )

    scored.sort(key=lambda entry: entry["score"])
    return {"ranked": scored, "best_option_id": scored[0]["option_id"]}


LODGING_TOOLS = [search_hotels, get_hotel_details, check_hotel_location, compare_hotels]
