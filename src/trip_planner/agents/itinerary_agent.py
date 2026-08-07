"""Itinerary Agent (plan.md agent #10).

Merges every agent's output into the single document the traveler actually
reads: the day-by-day plan, the chosen flights and stay, the budget position
and the practical notes.

**Why this node bounds the agent and can finish without it.** Every tool here
is deterministic — `read_agent_results` re-reads state that cannot change
mid-node, and the two formatters are pure functions. A model that treats an
unchanged answer as "not finished yet" calls the same tool again, gets the same
answer, and never stops; the sub-agent inherited the graph's `recursion_limit`
(80), so that loop ran ~40 model calls deep and the run never reached the
Calendar stage. Three things now make that impossible:

* **The read tool answers once.** A repeat call returns an instruction to write
  the plan instead of the identical dict — see `itinerary_tools`.
* **The agent gets its own step budget.** `STEP_LIMIT` bounds this sub-agent
  rather than letting it spend the whole graph's.
* **The document is not the model's to lose.** Every field of the itinerary
  can be derived in Python from what the other agents already produced. If the
  agent loops out, errors, or comes back with gaps, `_repair` fills them from
  state. So the node always hands the Calendar Agent a real itinerary, and the
  worst a misbehaving model can cost is the prose, not the trip.
"""

from __future__ import annotations

import logging

from trip_planner.agents.factory import build_structured_agent, run_agent, step_limit
from trip_planner.schemas import (
    AgentName,
    BudgetResult,
    CriticResult,
    DayPlan,
    DestinationResearch,
    FlightsResult,
    ItineraryResult,
    LodgingResult,
    Severity,
    TravelerProfile,
)
from trip_planner.state import TripState
from trip_planner.tools import ITINERARY_TOOLS, make_read_agent_results
from trip_planner.tools.itinerary_tools import format_trip_plan

logger = logging.getLogger(__name__)

# Ceiling on the agent's own loop, from the one table that bounds every agent.
# The intended path is three tool calls — read, build the days, format — so it
# leaves room for a bad call or two and then stops.
STEP_LIMIT = step_limit("itinerary")

SYSTEM_PROMPT = """You are the Itinerary Agent of an autonomous trip planner.

Your job is to merge everything the other agents produced into one clear plan
for the traveler. You are the only agent whose output a person reads directly,
so write for them, not for a machine.

Follow these steps, each exactly once and in this order:
1. Call `read_agent_results` to get every agent's output. Call it ONCE - the
   results cannot change while you work, so a second call returns nothing new.
2. Call `build_daily_itinerary` with the routed days to put them in order.
3. Write the overview, the flight and lodging summaries, and the budget
   summary, in plain language.
4. Turn the research into `practical_notes`: weather to pack for, visa rules,
   currency and any safety advice.
5. Call `format_trip_plan` last and put its output in `markdown`. Then return
   your answer - you are done.

Rules:
- Never call the same tool twice with the same arguments. If a result looks
  incomplete, that is what the other agents produced: report it as it is and
  finish. Repeating a call cannot change it.
- Report only what the other agents produced. Never add a place, a price or a
  flight that is not in their results, and never quietly fix a gap by
  inventing something.
- `days` must match the routed plan exactly - same stops, same times.
- Write summaries a traveler can act on: name the airline and times, the hotel
  and what it costs, and what the trip totals against the budget.
- If the Critic left warnings, mention the relevant ones as practical notes so
  the traveler is not surprised by them.
- If a stage produced nothing, leave its summary empty rather than inventing
  content for it.
"""


def build_itinerary_agent(state: TripState):
    """Build the Itinerary Agent runnable.

    Args:
        state: The state this turn should be able to read.

    Returns:
        The agent runnable, with `read_agent_results` bound to `state`.
    """
    return build_structured_agent(
        role="itinerary",
        tools=[make_read_agent_results(state), *ITINERARY_TOOLS],
        system_prompt=SYSTEM_PROMPT,
        response_format=ItineraryResult,
        name="itinerary_agent",
    )


def _itinerary_brief(profile: TravelerProfile, days: int) -> str:
    """Build the write-up request.

    Args:
        profile: The traveler profile.
        days: Number of days the trip covers.

    Returns:
        A prompt describing the document to produce.
    """
    interests = ", ".join(profile.interests) if profile.interests else "general travel"
    return (
        f"Write the final trip plan.\n"
        f"Destination: {profile.destination}\n"
        f"Dates: {profile.start_date} to {profile.end_date} ({days} days)\n"
        f"Travelers: {profile.travelers or 1}\n"
        f"Interests: {interests}\n\n"
        f"Start by calling `read_agent_results` once to collect what the other "
        f"agents produced, then build the document from it and finish."
    )


# ---------------------------------------------------------------------------
# Deterministic composition
#
# Everything below derives a field of the itinerary from state alone. It backs
# the agent up rather than replacing it: the model writes the prose, and these
# fill whatever it left empty — including everything, when it never answered.
# ---------------------------------------------------------------------------


def _chosen(options: list, recommended_id: str | None):
    """Pick the recommended option, falling back to the first one.

    Args:
        options: The options an agent returned, best first.
        recommended_id: The `option_id` that agent recommended.

    Returns:
        The chosen option, or None when there are none.
    """
    if not options:
        return None
    return next(
        (option for option in options if option.option_id == recommended_id),
        options[0],
    )


def _title(profile: TravelerProfile, days: int) -> str:
    """Build the trip's title.

    Args:
        profile: The traveler profile.
        days: Number of days the trip covers.

    Returns:
        A title such as "5 Days in Lisbon, Portugal".
    """
    destination = profile.destination or "Your Trip"
    return f"{days} Day{'s' if days != 1 else ''} in {destination}"


def _overview(profile: TravelerProfile, days: int) -> str:
    """Build a plain introduction to the trip.

    Args:
        profile: The traveler profile.
        days: Number of days the trip covers.

    Returns:
        A short paragraph introducing the trip.
    """
    parts = [
        f"A {days}-day trip to {profile.destination or 'your destination'}",
    ]
    if profile.start_date and profile.end_date:
        parts.append(f"from {profile.start_date} to {profile.end_date}")
    if profile.travelers:
        parts.append(
            f"for {profile.travelers} traveler{'s' if profile.travelers != 1 else ''}"
        )
    sentence = ", ".join(parts) + "."
    if profile.interests:
        sentence += f" Planned around {', '.join(profile.interests)}."
    return sentence


def _flight_summary(flights: FlightsResult | None) -> str:
    """Describe the chosen flights in words.

    Args:
        flights: The Flights Agent's output, when it ran.

    Returns:
        A sentence naming the airline, times and price, or "" when no flights
        were found.
    """
    option = _chosen(flights.options, flights.recommended_option_id) if flights else None
    if option is None:
        return ""

    legs = "; ".join(
        f"{leg.airline} {leg.flight_number} {leg.departure_airport} "
        f"{leg.departure_time} to {leg.arrival_airport} {leg.arrival_time}"
        for leg in option.legs
    )
    price = f"{option.price:,.0f} {option.currency} for all travelers"
    return f"{legs}. Total {price}." if legs else f"Flights total {price}."


def _lodging_summary(lodging: LodgingResult | None) -> str:
    """Describe the chosen stay in words.

    Args:
        lodging: The Lodging Agent's output, when it ran.

    Returns:
        A sentence naming the stay and what it costs, or "" when none was
        found.
    """
    option = _chosen(lodging.options, lodging.recommended_option_id) if lodging else None
    if option is None:
        return ""

    where = f"{option.name}"
    if option.address:
        where += f", {option.address}"
    cost = f"{option.total_price:,.0f} {option.currency} for the whole stay"
    if option.price_per_night:
        cost += f" ({option.price_per_night:,.0f} {option.currency} per night)"
    return f"{where}. {cost.capitalize()}."


def _budget_summary(budget: BudgetResult | None) -> str:
    """Describe the cost position in words.

    Args:
        budget: The Budget Agent's output, when it ran.

    Returns:
        A sentence stating the total against the stated budget, or "" when the
        budget stage produced nothing.
    """
    if budget is None:
        return ""

    total = f"The trip totals {budget.total_cost:,.0f} {budget.currency}"
    if budget.budget_amount is None:
        return total + "."
    against = f" against a budget of {budget.budget_amount:,.0f} {budget.currency}"
    if budget.within_budget:
        return f"{total}{against}, which it stays within."
    return f"{total}{against}, over by {budget.overage:,.0f} {budget.currency}."


def _practical_notes(
    research: DestinationResearch | None, critic: CriticResult | None
) -> list[str]:
    """Turn the research and the Critic's warnings into traveler notes.

    Args:
        research: The Destination Research Agent's output, when it ran.
        critic: The Critic's verdict, when it ran.

    Returns:
        One note per topic that has content. Blockers are left out: they went
        back to Routing rather than to the traveler.
    """
    notes: list[str] = []
    if research is not None:
        for label, value in (
            ("Weather", research.weather),
            ("Entry requirements", research.entry_requirements),
            ("Currency", research.currency),
            ("Getting around", research.transportation),
            ("Safety", research.safety),
        ):
            if value and value.strip():
                notes.append(f"{label}: {value.strip()}")

    if critic is not None:
        notes += [
            f"Heads-up: {issue.description}"
            for issue in critic.issues
            if issue.severity != Severity.BLOCKER and issue.description.strip()
        ]
    return notes


def _day_dicts(days: list[DayPlan]) -> list[dict]:
    """Render the days as the plain dicts `format_trip_plan` expects.

    Times are written as HH:MM rather than the ISO HH:MM:SS a JSON dump gives,
    because this text goes straight to the traveler.

    Args:
        days: The final day-by-day plan.

    Returns:
        One dict per day, stops included.
    """
    return [
        {
            "day_number": day.day_number,
            "date": str(day.date),
            "summary": day.summary,
            "stops": [
                {
                    "name": stop.name,
                    "start_time": stop.start_time.strftime("%H:%M"),
                    "end_time": stop.end_time.strftime("%H:%M"),
                    "travel_minutes_from_previous": stop.travel_minutes_from_previous,
                    "travel_mode": stop.travel_mode,
                    "notes": stop.notes,
                }
                for stop in day.stops
            ],
        }
        for day in days
    ]


def _render(itinerary: ItineraryResult) -> str:
    """Render a finished itinerary as Markdown.

    Uses the same tool the agent is asked to call, so a document built here is
    laid out identically to one the agent formatted itself.

    Args:
        itinerary: The itinerary to render.

    Returns:
        The plan as Markdown.
    """
    return format_trip_plan.invoke(
        {
            "title": itinerary.title,
            "overview": itinerary.overview,
            "days": _day_dicts(itinerary.days),
            "flight_summary": itinerary.flight_summary,
            "lodging_summary": itinerary.lodging_summary,
            "budget_summary": itinerary.budget_summary,
            "practical_notes": itinerary.practical_notes,
        }
    )


def _repair(
    itinerary: ItineraryResult,
    state: TripState,
    profile: TravelerProfile,
    days: int,
) -> ItineraryResult:
    """Fill every field the agent left empty from the state, then render.

    Called on the agent's own output as well as on the empty result used when
    the agent never answered, so both paths produce the same shape of document.
    Anything the agent did write is kept exactly as written.

    Args:
        itinerary: The agent's result, or an empty one.
        state: The shared trip state, holding every other agent's output.
        profile: The traveler profile.
        days: Number of days the trip covers.

    Returns:
        A complete itinerary, with `markdown` rendered.
    """
    update: dict = {}

    if not itinerary.days:
        routing = state.get("routing")
        if routing is not None and routing.days:
            update["days"] = list(routing.days)
    if not itinerary.title.strip():
        update["title"] = _title(profile, days)
    if not itinerary.overview.strip():
        update["overview"] = _overview(profile, days)
    if not itinerary.flight_summary.strip():
        update["flight_summary"] = _flight_summary(state.get("flights"))
    if not itinerary.lodging_summary.strip():
        update["lodging_summary"] = _lodging_summary(state.get("lodging"))
    if not itinerary.budget_summary.strip():
        update["budget_summary"] = _budget_summary(state.get("budget"))
    if not itinerary.practical_notes:
        update["practical_notes"] = _practical_notes(
            state.get("research"), state.get("critic")
        )

    if update:
        itinerary = itinerary.model_copy(update=update)
    if not itinerary.markdown.strip():
        itinerary = itinerary.model_copy(update={"markdown": _render(itinerary)})
    return itinerary


def _write(
    state: TripState, profile: TravelerProfile, days: int, collector=None
) -> ItineraryResult | None:
    """Run the agent over one plan and return what it wrote.

    A failure is contained rather than raised: the node can finish the document
    from state, and doing so is far better than losing ten completed stages to
    the eleventh. That includes the step limit being hit, which is exactly the
    runaway-loop case this bound exists for.

    Because nothing escapes to `resilient_node`, its retry never fires for this
    stage — so the retry is applied here instead. Otherwise a single 503 would
    quietly cost the traveler the model's prose, when waiting two seconds would
    have got it.

    Args:
        state: The shared trip state.
        profile: The traveler profile.
        days: Number of days the trip covers.
        collector: Optional metrics callback.

    Returns:
        The agent's `ItineraryResult`, or None when it could not produce one.
    """
    from trip_planner.resilience import with_retry

    def write() -> ItineraryResult:
        return run_agent(
            build_itinerary_agent(state),
            _itinerary_brief(profile, days),
            role="itinerary",
            collector=collector,
        )

    try:
        return with_retry(write, "itinerary")()
    except Exception as exc:  # noqa: BLE001 - contained deliberately
        from langgraph.errors import GraphBubbleUp, GraphRecursionError

        if isinstance(exc, GraphBubbleUp):
            # LangGraph's own control flow, not a failure. It has to travel up.
            raise
        if isinstance(exc, GraphRecursionError):
            logger.warning(
                "itinerary agent hit its %d-step limit; writing the plan from "
                "the other agents' results instead",
                STEP_LIMIT,
            )
        else:
            logger.error("itinerary agent failed: %s", exc)
        return None


def itinerary_node(state: TripState, collector=None) -> dict:
    """Graph node: run the Itinerary Agent.

    Always returns an itinerary, so the Calendar Agent always has a plan to put
    on the calendar. See this module's docstring.

    Args:
        state: The shared trip state; reads every agent's output.
        collector: Optional metrics callback.

    Returns:
        A partial state update with the finished itinerary.
    """
    profile = state["intake"].profile
    days = (
        (profile.end_date - profile.start_date).days + 1
        if profile.start_date and profile.end_date
        else 1
    )

    written = _write(state, profile, days, collector)
    itinerary = _repair(
        written if written is not None else ItineraryResult(title="", overview=""),
        state,
        profile,
        days,
    )

    return {
        "itinerary": itinerary,
        "completed_agents": [AgentName.ITINERARY],
    }
