"""Tools for the Budget Agent (plan.md agent #8).

`calculate_total_cost` and `estimate_food_cost` are plain arithmetic, done in
Python so the totals are always right - a model doing sums in its head is the
one thing that would make the whole budget untrustworthy.
`convert_currency` reads a live rate, and `suggest_cheaper_alternatives`
turns the numbers into concrete cuts.
"""

from __future__ import annotations

from langchain_core.tools import tool

from trip_planner.tools.serp import serp_search

# Rough per-person daily food cost by style, in USD. Used only when the
# itinerary has no priced restaurants to add up.
DAILY_FOOD_USD: dict[str, float] = {
    "budget": 30.0,
    "moderate": 60.0,
    "comfortable": 100.0,
    "luxury": 180.0,
}

# Per-person daily allowance for local transport, in USD.
DAILY_LOCAL_TRANSPORT_USD = 12.0


@tool
def calculate_total_cost(
    flights: float = 0.0,
    lodging: float = 0.0,
    activities: float = 0.0,
    food: float = 0.0,
    local_transport: float = 0.0,
    budget_amount: float | None = None,
) -> dict:
    """Add up the trip's costs and compare them against the budget.

    Every amount must already be in the same currency; use `convert_currency`
    first for anything that is not.

    Args:
        flights: Total flight cost for all travelers.
        lodging: Total lodging cost for the whole stay.
        activities: Total cost of entry tickets and activities.
        food: Total food cost for the trip.
        local_transport: Total local transport cost.
        budget_amount: The traveler's stated budget, when known.

    Returns:
        A dict with the `lines` breakdown, `total_cost`, `within_budget`,
        `overage` and `remaining`.
    """
    lines = [
        {"category": "flights", "amount": round(flights, 2)},
        {"category": "lodging", "amount": round(lodging, 2)},
        {"category": "activities", "amount": round(activities, 2)},
        {"category": "food", "amount": round(food, 2)},
        {"category": "local_transport", "amount": round(local_transport, 2)},
    ]
    total = round(sum(line["amount"] for line in lines), 2)

    if budget_amount is None:
        return {
            "lines": lines,
            "total_cost": total,
            "within_budget": True,
            "overage": 0.0,
            "note": "No budget was stated, so nothing to compare against.",
        }

    overage = round(max(0.0, total - budget_amount), 2)
    return {
        "lines": lines,
        "total_cost": total,
        "budget_amount": budget_amount,
        "within_budget": overage == 0,
        "overage": overage,
        "remaining": round(max(0.0, budget_amount - total), 2),
        "percent_of_budget": round(total / budget_amount * 100, 1) if budget_amount else None,
    }


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies at the current rate.

    Args:
        amount: The amount to convert.
        from_currency: ISO 4217 code to convert from, e.g. "EUR".
        to_currency: ISO 4217 code to convert to, e.g. "USD".

    Returns:
        A dict with `converted_amount`, the `rate` used and `as_of`, or
        `error` when no rate could be read.
    """
    if from_currency.upper() == to_currency.upper():
        return {
            "converted_amount": round(amount, 2),
            "rate": 1.0,
            "from_currency": from_currency.upper(),
            "to_currency": to_currency.upper(),
            "as_of": "same currency",
        }

    response = serp_search(
        engine="google",
        q=f"{amount} {from_currency} to {to_currency}",
        hl="en",
    )
    if "error" in response:
        return {"error": response["error"]}

    answer_box = response.get("answer_box") or {}
    converted = answer_box.get("price")
    if converted is None:
        return {
            "error": (
                f"No exchange rate found for {from_currency} to {to_currency}. "
                "Report the amounts in their original currency instead."
            )
        }

    return {
        "converted_amount": round(float(converted), 2),
        "rate": round(float(converted) / amount, 6) if amount else None,
        "from_currency": from_currency.upper(),
        "to_currency": to_currency.upper(),
        "as_of": answer_box.get("date", ""),
    }


@tool
def estimate_food_cost(
    travelers: int,
    days: int,
    style: str = "moderate",
    known_restaurant_costs: float = 0.0,
) -> dict:
    """Estimate what the travelers will spend on food.

    Use the itinerary's actual restaurant prices where you have them, and pass
    them as `known_restaurant_costs`; this tool covers the remaining meals with
    a per-day allowance.

    Args:
        travelers: Number of people eating.
        days: Number of days the trip covers.
        style: Spending level: "budget", "moderate", "comfortable" or "luxury".
        known_restaurant_costs: Meals already priced from the itinerary, in USD.

    Returns:
        A dict with `estimated_total_usd`, the `daily_rate_per_person` used and
        how the figure was built.
    """
    rate = DAILY_FOOD_USD.get(style.lower(), DAILY_FOOD_USD["moderate"])
    allowance = rate * travelers * days
    total = round(max(allowance, known_restaurant_costs), 2)

    return {
        "estimated_total_usd": total,
        "daily_rate_per_person": rate,
        "style": style.lower(),
        "travelers": travelers,
        "days": days,
        "known_restaurant_costs": known_restaurant_costs,
        "basis": (
            "Used the itemized restaurant costs, which exceed the allowance."
            if known_restaurant_costs > allowance
            else f"{rate} USD per person per day at '{style}' level."
        ),
    }


@tool
def estimate_local_transport_cost(travelers: int, days: int) -> dict:
    """Estimate what the travelers will spend getting around locally.

    Args:
        travelers: Number of people travelling.
        days: Number of days the trip covers.

    Returns:
        A dict with `estimated_total_usd` and the daily rate used.
    """
    total = round(DAILY_LOCAL_TRANSPORT_USD * travelers * days, 2)
    return {
        "estimated_total_usd": total,
        "daily_rate_per_person": DAILY_LOCAL_TRANSPORT_USD,
        "basis": "Metro passes and occasional taxis at city rates.",
    }


@tool
def suggest_cheaper_alternatives(
    overage: float,
    currency: str,
    flights_cost: float = 0.0,
    lodging_cost: float = 0.0,
    activities_cost: float = 0.0,
    food_cost: float = 0.0,
) -> dict:
    """Work out where to cut when the trip costs more than the budget.

    Targets the largest categories first and states how much each cut has to
    save, so the suggestions are specific rather than generic advice.

    Args:
        overage: How much the trip exceeds the budget by.
        currency: Currency of every amount here.
        flights_cost: Current flight cost.
        lodging_cost: Current lodging cost.
        activities_cost: Current activities cost.
        food_cost: Current food cost.

    Returns:
        A dict with `suggestions`, each naming a category, a target saving and
        a concrete action.
    """
    if overage <= 0:
        return {"suggestions": [], "note": "The trip is within budget; no cuts needed."}

    categories = sorted(
        [
            ("lodging", lodging_cost),
            ("flights", flights_cost),
            ("activities", activities_cost),
            ("food", food_cost),
        ],
        key=lambda entry: entry[1],
        reverse=True,
    )

    actions = {
        "lodging": (
            "Move to a cheaper stay or one slightly outside the center, or "
            "drop a hotel class"
        ),
        "flights": (
            "Accept one stop instead of a direct flight, or shift departure "
            "by a day to a cheaper fare"
        ),
        "activities": (
            "Replace two paid attractions with free ones such as viewpoints, "
            "parks and markets"
        ),
        "food": (
            "Swap one restaurant dinner a day for a market or bakery lunch"
        ),
    }

    suggestions = []
    remaining = overage
    for name, cost in categories:
        if remaining <= 0 or cost <= 0:
            continue
        # Ask no more than 30% out of any one category, so the plan stays intact.
        target = round(min(remaining, cost * 0.3), 2)
        if target <= 0:
            continue
        suggestions.append(
            {
                "category": name,
                "target_saving": target,
                "currency": currency,
                "action": f"{actions[name]} to save about {target} {currency}.",
            }
        )
        remaining = round(remaining - target, 2)

    return {
        "suggestions": suggestions,
        "overage": overage,
        "currency": currency,
        "fully_covered": remaining <= 0,
        "still_uncovered": max(0.0, remaining),
    }


BUDGET_TOOLS = [
    calculate_total_cost,
    convert_currency,
    estimate_food_cost,
    estimate_local_transport_cost,
    suggest_cheaper_alternatives,
]
