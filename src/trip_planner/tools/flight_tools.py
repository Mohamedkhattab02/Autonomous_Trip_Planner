"""Tools for the Flights Agent (plan.md agent #4).

`search_flights` and `get_flight_details` read live Google Flights data
through SerpApi. `compare_flights` is deliberately plain Python: ranking by
price, duration and stops is arithmetic, and doing it in code rather than in
the model keeps the ordering reproducible and auditable.
"""

from __future__ import annotations

from langchain_core.tools import tool

from trip_planner.tools.serp import serp_search

# How many itineraries to hand the agent. Enough to compare meaningfully
# without flooding the model's context.
MAX_FLIGHT_OPTIONS = 6

# Weights for `compare_flights`. Price dominates, then total travel time,
# then the hassle of stops.
PRICE_WEIGHT = 1.0
DURATION_WEIGHT = 0.5
STOPS_WEIGHT = 0.3


def _summarize_leg(leg: dict) -> dict:
    """Reduce one SerpApi flight leg to the fields the agent needs.

    Args:
        leg: A raw leg from the SerpApi `flights` array.

    Returns:
        A flat dict matching the `FlightLeg` schema.
    """
    departure = leg.get("departure_airport", {})
    arrival = leg.get("arrival_airport", {})
    return {
        "airline": leg.get("airline", ""),
        "flight_number": leg.get("flight_number", ""),
        "departure_airport": departure.get("id", ""),
        "arrival_airport": arrival.get("id", ""),
        "departure_time": departure.get("time", ""),
        "arrival_time": arrival.get("time", ""),
        "duration_minutes": leg.get("duration", 0),
        "airplane": leg.get("airplane", ""),
        "travel_class": leg.get("travel_class", ""),
        "legroom": leg.get("legroom", ""),
    }


def _summarize_itinerary(index: int, itinerary: dict) -> dict:
    """Reduce one SerpApi itinerary to the fields the agent needs.

    Args:
        index: Position in the result list, used to build a stable option id.
        itinerary: A raw entry from `best_flights` or `other_flights`.

    Returns:
        A dict matching the `FlightOption` schema, plus baggage extensions.
    """
    legs = [_summarize_leg(leg) for leg in itinerary.get("flights", [])]
    layovers = [
        f"{layover.get('name', '')} ({layover.get('duration', 0)} min)"
        for layover in itinerary.get("layovers", [])
    ]
    # Baggage and fare rules arrive as free-text extensions on the legs.
    extensions = [
        text
        for leg in itinerary.get("flights", [])
        for text in leg.get("extensions", [])
        if "bag" in text.lower() or "carry-on" in text.lower()
    ]
    return {
        "option_id": f"flight-{index}",
        "legs": legs,
        "stops": max(len(legs) - 1, 0),
        "total_duration_minutes": itinerary.get("total_duration", 0),
        "price": itinerary.get("price", 0),
        "layovers": layovers,
        "baggage_notes": extensions or ["not stated"],
        "type": itinerary.get("type", ""),
    }


@tool
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None = None,
    travelers: int = 1,
    currency: str = "USD",
) -> dict:
    """Search real flights between two airports for the given dates.

    Args:
        origin: Departure airport IATA code, e.g. "TLV". Cities are not
            accepted; convert the city to its main airport code first.
        destination: Arrival airport IATA code, e.g. "LIS".
        departure_date: Outbound date in YYYY-MM-DD format.
        return_date: Return date in YYYY-MM-DD format. Omit for one-way.
        travelers: Number of adult passengers.
        currency: ISO 4217 currency code for the prices.

    Returns:
        A dict with `options` (list of itineraries with price, legs, stops and
        duration) and `currency`, or `error` when the lookup failed.
    """
    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": departure_date,
        "adults": travelers,
        "currency": currency,
        "hl": "en",
        # SerpApi type 1 = round trip, 2 = one way.
        "type": "1" if return_date else "2",
    }
    if return_date:
        params["return_date"] = return_date

    response = serp_search(**params)
    if "error" in response:
        return {"error": response["error"], "options": []}

    itineraries = response.get("best_flights", []) + response.get("other_flights", [])
    options = [
        _summarize_itinerary(index, itinerary)
        for index, itinerary in enumerate(itineraries[:MAX_FLIGHT_OPTIONS], start=1)
    ]
    if not options:
        return {
            "error": f"No flights found from {origin} to {destination} on {departure_date}.",
            "options": [],
        }
    return {"options": options, "currency": currency}


@tool
def compare_flights(options: list[dict]) -> dict:
    """Rank flight options by price, total duration and number of stops.

    Scores each option against the best value seen for each factor, so the
    ranking is reproducible rather than a judgement call. Price matters most,
    then travel time, then stops.

    Args:
        options: Options from `search_flights`. Each needs `option_id`,
            `price`, `total_duration_minutes` and `stops`.

    Returns:
        A dict with `ranked` (option ids best first, each with its score and
        the numbers behind it) and `best_option_id`.
    """
    usable = [option for option in options if option.get("price")]
    if not usable:
        return {"ranked": [], "best_option_id": None, "note": "No priced options given."}

    cheapest = min(option["price"] for option in usable)
    fastest = min(
        option.get("total_duration_minutes") or 1 for option in usable
    )
    fewest_stops = min(option.get("stops", 0) for option in usable)

    scored = []
    for option in usable:
        duration = option.get("total_duration_minutes") or fastest
        stops = option.get("stops", 0)
        # 0 is a perfect score: each term is the penalty over the best seen.
        score = (
            PRICE_WEIGHT * (option["price"] / cheapest - 1)
            + DURATION_WEIGHT * (duration / fastest - 1)
            + STOPS_WEIGHT * (stops - fewest_stops)
        )
        scored.append(
            {
                "option_id": option.get("option_id", ""),
                "score": round(score, 4),
                "price": option["price"],
                "total_duration_minutes": duration,
                "stops": stops,
            }
        )

    scored.sort(key=lambda entry: entry["score"])
    return {"ranked": scored, "best_option_id": scored[0]["option_id"]}


@tool
def get_flight_details(
    airline: str, flight_number: str, route: str = ""
) -> dict:
    """Look up details for a specific flight, such as baggage and aircraft.

    Use this when a shortlisted option is missing baggage information the
    traveler needs before choosing.

    Args:
        airline: The operating airline, e.g. "TAP Air Portugal".
        flight_number: The flight number, e.g. "TP 1234".
        route: Optional route context, e.g. "Tel Aviv to Lisbon".

    Returns:
        A dict with `summary` and `sources`, or `error` when nothing was found.
    """
    query = f"{airline} flight {flight_number} {route} baggage allowance and aircraft"
    response = serp_search(engine="google", q=query, hl="en", num=5)
    if "error" in response:
        return {"error": response["error"]}

    organic = response.get("organic_results", [])
    if not organic:
        return {"error": f"No details found for {airline} {flight_number}."}

    return {
        "summary": response.get("answer_box", {}).get("snippet", ""),
        "sources": [
            {
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "content": result.get("snippet", ""),
            }
            for result in organic[:4]
        ],
    }


FLIGHT_TOOLS = [search_flights, compare_flights, get_flight_details]
