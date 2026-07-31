"""Tools for the Critic Agent (plan.md agent #9).

`validate_schedule` and `validate_budget` are deterministic: overlapping stops
and a total that exceeds the budget are facts, and the whole point of the
Critic is that they are checked by code rather than re-judged by a model.
`verify_place` and `verify_opening_hours` go back to the live source to
confirm a place is real and open when the plan says it is.
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime, time

from langchain_core.tools import tool

from trip_planner.tools.serp import serp_search

WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

# The day the Routing Agent schedules inside. Anything outside it is an error.
DAY_START = time(9, 0)
DAY_END = time(21, 0)

# A stop longer than this is almost certainly a scheduling mistake.
MAX_STOP_MINUTES = 300


def _parse_time(value: str) -> time | None:
    """Parse an "HH:MM" or "HH:MM:SS" string into a time.

    Args:
        value: The time text to parse.

    Returns:
        The parsed time, or None when it is not valid.
    """
    for pattern in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(str(value).strip(), pattern).time()
        except (ValueError, AttributeError):
            continue
    return None


def _minutes(value: time) -> int:
    """Return a time as minutes since midnight.

    Args:
        value: The time to convert.

    Returns:
        Minutes since midnight.
    """
    return value.hour * 60 + value.minute


@tool
def verify_place(place_name: str, destination: str) -> dict:
    """Check that a place really exists at the destination.

    Use this on places whose name looks generic or invented, to confirm the
    plan is not sending the traveler somewhere that does not exist.

    Args:
        place_name: The name as it appears in the plan.
        destination: The city it is supposed to be in.

    Returns:
        A dict with `exists`, plus the address, rating and coordinates found.
    """
    response = serp_search(
        engine="google_maps",
        q=f"{place_name} {destination}",
        type="search",
        hl="en",
    )
    if "error" in response:
        return {
            "exists": None,
            "error": response["error"],
            "note": "Could not verify; do not report this as a fabricated place.",
        }

    results = response.get("local_results") or []
    record = results[0] if results else response.get("place_results")
    if not record:
        return {
            "exists": False,
            "message": f"No place called '{place_name}' was found in {destination}.",
        }

    return {
        "exists": True,
        "name": record.get("title", ""),
        "address": record.get("address", ""),
        "rating": record.get("rating"),
        "coordinates": record.get("gps_coordinates", {}),
        "permanently_closed": "permanently closed"
        in str(record.get("open_state", "")).lower(),
    }


@tool
def verify_opening_hours(place_name: str, destination: str, visit_date: str) -> dict:
    """Check a place is actually open on the date the plan visits it.

    Args:
        place_name: The name as it appears in the plan.
        destination: The city it is in.
        visit_date: The scheduled date in YYYY-MM-DD format.

    Returns:
        A dict with `is_open`, the `weekday`, and the hours reported for it.
    """
    try:
        parsed = date_type.fromisoformat(visit_date)
    except ValueError:
        return {"is_open": None, "error": f"'{visit_date}' is not a YYYY-MM-DD date."}

    weekday = WEEKDAYS[parsed.weekday()]
    response = serp_search(
        engine="google_maps",
        q=f"{place_name} {destination}",
        type="search",
        hl="en",
    )
    if "error" in response:
        return {"is_open": None, "error": response["error"]}

    results = response.get("local_results") or []
    record = results[0] if results else response.get("place_results")
    if not record:
        return {"is_open": None, "message": f"'{place_name}' was not found."}

    operating_hours = record.get("operating_hours") or {}
    if not operating_hours:
        return {
            "is_open": None,
            "weekday": weekday.capitalize(),
            "message": "No per-day hours are published for this place.",
        }

    hours_text = str(operating_hours.get(weekday, ""))
    return {
        "is_open": bool(hours_text) and "closed" not in hours_text.lower(),
        "weekday": weekday.capitalize(),
        "hours_that_day": hours_text or "not reported",
        "all_hours": operating_hours,
    }


@tool
def validate_schedule(days: list[dict]) -> dict:
    """Check a day-by-day plan for overlaps, gaps and impossible timing.

    Catches stops that overlap, stops that start before travel from the
    previous one could finish, and anything outside the traveler's day.

    Args:
        days: The plan. Each day needs `day_number`, `date` and `stops`, and
            each stop needs `name`, `start_time`, `end_time` and
            `travel_minutes_from_previous`.

    Returns:
        A dict with `valid` and a list of `problems`, each naming the day, the
        stop and what is wrong with it.
    """
    problems: list[dict] = []

    for day in days:
        day_number = day.get("day_number", "?")
        stops = day.get("stops", [])
        previous_end: time | None = None
        previous_name = ""

        if not stops:
            problems.append(
                {
                    "day": day_number,
                    "stop": "-",
                    "problem": "The day has no stops at all.",
                }
            )
            continue

        for stop in stops:
            name = stop.get("name", "unnamed stop")
            start = _parse_time(stop.get("start_time", ""))
            end = _parse_time(stop.get("end_time", ""))

            if start is None or end is None:
                problems.append(
                    {
                        "day": day_number,
                        "stop": name,
                        "problem": "Start or end time is missing or unparseable.",
                    }
                )
                continue

            if _minutes(end) <= _minutes(start):
                problems.append(
                    {
                        "day": day_number,
                        "stop": name,
                        "problem": f"Ends at {end} but starts at {start}.",
                    }
                )

            duration = _minutes(end) - _minutes(start)
            if duration > MAX_STOP_MINUTES:
                problems.append(
                    {
                        "day": day_number,
                        "stop": name,
                        "problem": (
                            f"Lasts {duration} minutes, longer than a realistic visit."
                        ),
                    }
                )

            if _minutes(start) < _minutes(DAY_START) or _minutes(end) > _minutes(DAY_END):
                problems.append(
                    {
                        "day": day_number,
                        "stop": name,
                        "problem": (
                            f"Scheduled {start}-{end}, outside the "
                            f"{DAY_START}-{DAY_END} day."
                        ),
                    }
                )

            if previous_end is not None:
                travel = stop.get("travel_minutes_from_previous", 0) or 0
                earliest = _minutes(previous_end) + travel
                if _minutes(start) < _minutes(previous_end):
                    problems.append(
                        {
                            "day": day_number,
                            "stop": name,
                            "problem": (
                                f"Starts at {start}, before '{previous_name}' "
                                f"ends at {previous_end}."
                            ),
                        }
                    )
                elif _minutes(start) < earliest:
                    problems.append(
                        {
                            "day": day_number,
                            "stop": name,
                            "problem": (
                                f"Starts at {start}, but leaving '{previous_name}' "
                                f"at {previous_end} plus {travel} min of travel means "
                                f"arriving no earlier than "
                                f"{earliest // 60:02d}:{earliest % 60:02d}."
                            ),
                        }
                    )

            previous_end = end
            previous_name = name

    return {"valid": not problems, "problems": problems, "days_checked": len(days)}


@tool
def validate_budget(
    total_cost: float, budget_amount: float | None = None, currency: str = "USD"
) -> dict:
    """Check the trip's total against the traveler's budget.

    Args:
        total_cost: The total from the Budget Agent.
        budget_amount: The traveler's stated budget, or None if none was given.
        currency: The currency both amounts are in.

    Returns:
        A dict with `within_budget`, the `overage` and a readable `message`.
    """
    if budget_amount is None:
        return {
            "within_budget": True,
            "overage": 0.0,
            "message": "No budget was stated, so there is nothing to breach.",
        }

    overage = round(max(0.0, total_cost - budget_amount), 2)
    return {
        "within_budget": overage == 0,
        "overage": overage,
        "percent_of_budget": round(total_cost / budget_amount * 100, 1)
        if budget_amount
        else None,
        "message": (
            f"The trip costs {total_cost} {currency} against a budget of "
            f"{budget_amount} {currency}"
            + (f", over by {overage} {currency}." if overage else ", within budget.")
        ),
    }


CRITIC_TOOLS = [verify_place, verify_opening_hours, validate_schedule, validate_budget]
