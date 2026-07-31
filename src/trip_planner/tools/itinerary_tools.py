"""Tools for the Itinerary Agent (plan.md agent #10).

`read_agent_results` is bound to the live state, like the manager's tool, so
the Itinerary Agent can pull every other agent's output without it all being
pasted into the prompt. `build_daily_itinerary` and `format_trip_plan` are
plain Python: the day structure and the Markdown layout are presentation
rules of the system, not something the model should improvise each run.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.tools import tool

from trip_planner.state import TripState


def make_read_agent_results(state: TripState) -> Callable:
    """Build a `read_agent_results` tool bound to one graph invocation's state.

    Args:
        state: The state as seen by the itinerary node on this turn.

    Returns:
        A LangChain tool that returns every agent's structured output.
    """

    @tool
    def read_agent_results() -> dict:
        """Read every agent's result: research, flights, lodging, plan, budget.

        Returns:
            A dict with one entry per agent that has run, holding its
            structured output.
        """
        results: dict = {}
        for key in ("intake", "research", "flights", "lodging", "budget", "routing"):
            value = state.get(key)
            if value is not None:
                results[key] = value.model_dump(mode="json")

        critic = state.get("critic")
        if critic is not None:
            # Only the surviving warnings matter to the traveler; blockers
            # were already sent back to routing for a fix.
            results["critic"] = {
                "approved": critic.approved,
                "remaining_issues": [
                    issue.model_dump(mode="json")
                    for issue in critic.issues
                    if issue.severity != "blocker"
                ],
            }
        return results

    return read_agent_results


@tool
def build_daily_itinerary(days: list[dict]) -> dict:
    """Turn the routed days into clean, ordered day entries.

    Sorts each day's stops by start time and reports how full each day is, so
    the write-up describes the plan as it actually stands.

    Args:
        days: The days from the routing stage. Each needs `day_number`,
            `date` and `stops`.

    Returns:
        A dict with the ordered `days` and a `totals` summary.
    """
    ordered = []
    for day in sorted(days, key=lambda entry: entry.get("day_number", 0)):
        stops = sorted(
            day.get("stops", []), key=lambda stop: str(stop.get("start_time", ""))
        )
        travel = sum(stop.get("travel_minutes_from_previous", 0) or 0 for stop in stops)
        ordered.append(
            {
                "day_number": day.get("day_number"),
                "date": day.get("date"),
                "summary": day.get("summary", ""),
                "stops": stops,
                "stop_count": len(stops),
                "total_travel_minutes": travel,
                "first_start": stops[0].get("start_time") if stops else None,
                "last_end": stops[-1].get("end_time") if stops else None,
            }
        )

    return {
        "days": ordered,
        "totals": {
            "days": len(ordered),
            "stops": sum(day["stop_count"] for day in ordered),
            "travel_minutes": sum(day["total_travel_minutes"] for day in ordered),
        },
    }


@tool
def format_trip_plan(
    title: str,
    overview: str,
    days: list[dict],
    flight_summary: str = "",
    lodging_summary: str = "",
    budget_summary: str = "",
    practical_notes: list[str] | None = None,
) -> str:
    """Render the whole trip as Markdown for the traveler.

    Call this last, once every section is written, and put the result in the
    `markdown` field of your answer.

    Args:
        title: Title of the trip, e.g. "5 Days in Lisbon".
        overview: Short paragraph introducing the trip.
        days: The days from `build_daily_itinerary`.
        flight_summary: The chosen flights, in words.
        lodging_summary: The chosen stay, in words.
        budget_summary: The cost position, in words.
        practical_notes: Weather, visa, currency and safety notes.

    Returns:
        The complete plan as a Markdown document.
    """
    parts = [f"# {title}", "", overview, ""]

    if flight_summary:
        parts += ["## Flights", "", flight_summary, ""]
    if lodging_summary:
        parts += ["## Where You're Staying", "", lodging_summary, ""]

    parts += ["## Day by Day", ""]
    for day in sorted(days, key=lambda entry: entry.get("day_number", 0)):
        heading = f"### Day {day.get('day_number')} - {day.get('date')}"
        if day.get("summary"):
            heading += f": {day['summary']}"
        parts.append(heading)
        parts.append("")

        stops = day.get("stops", [])
        if not stops:
            parts += ["_No activities scheduled._", ""]
            continue

        for stop in stops:
            line = (
                f"- **{stop.get('start_time')}-{stop.get('end_time')}** "
                f"{stop.get('name', '')}"
            )
            travel = stop.get("travel_minutes_from_previous", 0) or 0
            if travel:
                line += (
                    f" _({travel} min {stop.get('travel_mode', 'travel')} from the "
                    f"previous stop)_"
                )
            parts.append(line)
            if stop.get("notes"):
                parts.append(f"  - {stop['notes']}")
        parts.append("")

    if budget_summary:
        parts += ["## Budget", "", budget_summary, ""]

    if practical_notes:
        parts += ["## Practical Notes", ""]
        parts += [f"- {note}" for note in practical_notes]
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


ITINERARY_TOOLS = [build_daily_itinerary, format_trip_plan]
