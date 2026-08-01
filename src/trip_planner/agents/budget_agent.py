"""Budget Agent (plan.md agent #8).

Adds up what the trip actually costs - flights, lodging, activities, food and
local transport - and reports whether it fits the traveler's budget, with
concrete cuts when it does not.
"""

from __future__ import annotations

from trip_planner.agents.factory import build_structured_agent
from trip_planner.schemas import (
    AgentName,
    AttractionsResult,
    BudgetResult,
    FlightsResult,
    LodgingResult,
    RoutingResult,
    TravelerProfile,
)
from trip_planner.state import TripState
from trip_planner.tools import BUDGET_TOOLS

SYSTEM_PROMPT = """You are the Budget Agent of an autonomous trip planner.

Your job is to total the trip's real cost and check it against the budget.

Follow these steps:
1. Take the chosen flight and lodging prices from the brief. If either is in a
   different currency from the budget, call `convert_currency` first.
2. Add up the entry costs of the scheduled activities from the itinerary.
3. Call `estimate_food_cost` and `estimate_local_transport_cost` for the meals
   and getting around, then convert those USD figures to the budget currency
   if the budget is in something else.
4. Call `calculate_total_cost` with all five categories and the budget. Use
   its numbers exactly - never do the arithmetic yourself.
5. If it comes back over budget, call `suggest_cheaper_alternatives` and put
   its suggestions in `savings_suggestions`.

Rules:
- Every amount in your answer must be in the budget currency, and `currency`
  must say which that is.
- `total_cost`, `within_budget` and `overage` must be exactly what
  `calculate_total_cost` returned.
- Include one `lines` entry per category, with `detail` saying where the
  number came from, e.g. "flight-2 at 617 USD for 2 travelers".
- Set `savings_suggestions` only when the trip is over budget.
- Never invent a price. If a cost is unknown, say so in that line's `detail`
  and exclude it from the total.
"""


def build_budget_agent():
    """Build the Budget Agent runnable."""
    return build_structured_agent(
        role="budget",
        tools=BUDGET_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=BudgetResult,
        name="budget_agent",
    )


def _chosen_flight(flights: FlightsResult | None) -> str:
    """Describe the recommended flight's cost for the prompt.

    Args:
        flights: The Flights Agent's output, when it ran.

    Returns:
        A one-line cost description.
    """
    if flights is None or not flights.options:
        return "No flights were found, so exclude flights from the total."

    chosen = next(
        (
            option
            for option in flights.options
            if option.option_id == flights.recommended_option_id
        ),
        flights.options[0],
    )
    return f"{chosen.option_id}: {chosen.price} {chosen.currency} total for all travelers"


def _chosen_lodging(lodging: LodgingResult | None) -> str:
    """Describe the recommended stay's cost for the prompt.

    Args:
        lodging: The Lodging Agent's output, when it ran.

    Returns:
        A one-line cost description.
    """
    if lodging is None or not lodging.options:
        return "No lodging was found, so exclude lodging from the total."

    chosen = next(
        (
            option
            for option in lodging.options
            if option.option_id == lodging.recommended_option_id
        ),
        lodging.options[0],
    )
    return (
        f"{chosen.option_id} ({chosen.name}): {chosen.total_price} "
        f"{chosen.currency} for the whole stay"
    )


def _activity_costs(
    routing: RoutingResult | None, attractions: AttractionsResult | None
) -> str:
    """List the per-person entry cost of every scheduled place.

    Args:
        routing: The day-by-day plan.
        attractions: The pool the plan drew from, holding the costs.

    Returns:
        One line per scheduled place with a cost, for the prompt.
    """
    if routing is None or attractions is None:
        return "- (no scheduled activities)"

    costs = {place.place_id: place for place in attractions.places}
    lines = []
    for day in routing.days:
        for stop in day.stops:
            place = costs.get(stop.place_id)
            if place is None:
                continue
            lines.append(
                f"- day {day.day_number} {place.name}: "
                f"{place.estimated_cost} {place.currency} per person"
            )
    return "\n".join(lines) or "- (no scheduled activities carry a cost)"


def _budget_brief(
    profile: TravelerProfile,
    flights: FlightsResult | None,
    lodging: LodgingResult | None,
    routing: RoutingResult | None,
    attractions: AttractionsResult | None,
    days: int,
) -> str:
    """Build the budget request from everything chosen so far.

    Args:
        profile: The traveler profile, holding the budget.
        flights: The Flights Agent's output.
        lodging: The Lodging Agent's output.
        routing: The day-by-day plan.
        attractions: The pool of places, holding activity costs.
        days: Number of days the trip covers.

    Returns:
        A prompt listing every cost to total.
    """
    return (
        f"Total the cost of this trip and check it against the budget.\n"
        f"Destination: {profile.destination}\n"
        f"Travelers: {profile.travelers or 1}\n"
        f"Days: {days}\n"
        f"Budget: {profile.budget_amount or 'not stated'} "
        f"{profile.budget_currency or 'USD'}\n\n"
        f"Chosen flight: {_chosen_flight(flights)}\n"
        f"Chosen lodging: {_chosen_lodging(lodging)}\n\n"
        f"Scheduled activity costs:\n"
        f"{_activity_costs(routing, attractions)}"
    )


def budget_node(state: TripState, collector=None) -> dict:
    """Graph node: run the Budget Agent.

    Args:
        state: The shared trip state; reads `intake`, `flights`, `lodging`,
            `routing` and `attractions`.

    Returns:
        A partial state update with the budget breakdown.
    """
    profile = state["intake"].profile
    days = (
        (profile.end_date - profile.start_date).days + 1
        if profile.start_date and profile.end_date
        else 1
    )

    agent = build_budget_agent()
    if collector is not None:
        # Route this agent's token and tool usage into the run metrics.
        agent = agent.with_config(callbacks=[collector])
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": _budget_brief(
                        profile,
                        state.get("flights"),
                        state.get("lodging"),
                        state.get("routing"),
                        state.get("attractions"),
                        days,
                    ),
                }
            ]
        }
    )
    budget: BudgetResult = result["structured_response"]

    update: dict = {
        "budget": budget,
        # Record which plan these costs belong to, so a later revision forces
        # a recount instead of shipping figures for a plan that no longer exists.
        "budget_version": state.get("plan_version", 0),
    }
    if AgentName.BUDGET not in state.get("completed_agents", []):
        update["completed_agents"] = [AgentName.BUDGET]

    return update
