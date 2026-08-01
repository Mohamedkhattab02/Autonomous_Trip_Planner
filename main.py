"""Run the trip planner on a single request and print what each agent produced.

Usage:
    uv run main.py
    uv run main.py "I want 4 days in Rome in May 2027 for 2 people, 2500 EUR"
"""

from __future__ import annotations

import sys
import time

from trip_planner.export import to_pdf
from trip_planner.graph import plan_trip
from trip_planner.llm import describe_models
from trip_planner.metrics import summarize
from trip_planner.state import TripState

DEFAULT_REQUEST = (
    "I want to travel to Lisbon, Portugal from 2026-09-10 to 2026-09-15. "
    "We are 2 travelers flying from Tel Aviv, our total budget is 3000 USD, "
    "and we love food, history and walking tours."
)


def _heading(title: str) -> None:
    """Print a section heading.

    Args:
        title: The section name.
    """
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _print_intake(state: TripState) -> None:
    """Print the traveler profile and any question the planner needs answered."""
    intake = state.get("intake")
    if intake is None:
        return

    _heading("2. INTAKE")
    print(intake.profile.model_dump_json(indent=2))
    if intake.clarifying_question:
        print(f"\nNeeds from the traveler: {intake.clarifying_question}")


def _print_research(state: TripState) -> None:
    """Print the destination research with its sources."""
    research = state.get("research")
    if research is None:
        return

    _heading("3. DESTINATION RESEARCH")
    print(f"Summary:   {research.summary}")
    print(f"Weather:   {research.weather}")
    print(f"Safety:    {research.safety}")
    print(f"Currency:  {research.currency}")
    print(f"Transport: {research.transportation}")
    print(f"Entry:     {research.entry_requirements}")
    if research.sources:
        print("\nSources:")
        for source in research.sources:
            print(f"  - {source.title}: {source.url}")


def _print_flights(state: TripState) -> None:
    """Print the ranked flight options and the recommendation."""
    flights = state.get("flights")
    if flights is None:
        return

    _heading("4. FLIGHTS")
    for option in flights.options:
        marker = " <- recommended" if option.option_id == flights.recommended_option_id else ""
        print(
            f"\n{option.option_id}: {option.price} {option.currency} | "
            f"{option.stops} stop(s) | {option.total_duration_minutes} min{marker}"
        )
        for leg in option.legs:
            print(
                f"    {leg.airline} {leg.flight_number}: "
                f"{leg.departure_airport} {leg.departure_time} -> "
                f"{leg.arrival_airport} {leg.arrival_time}"
            )
    print(f"\n{flights.reasoning}")


def _print_lodging(state: TripState) -> None:
    """Print the ranked lodging options and the recommendation."""
    lodging = state.get("lodging")
    if lodging is None:
        return

    _heading("5. LODGING")
    for option in lodging.options:
        marker = " <- recommended" if option.option_id == lodging.recommended_option_id else ""
        rating = f" | {option.rating}/5" if option.rating else ""
        print(
            f"{option.option_id}: {option.name} | {option.total_price} "
            f"{option.currency} total{rating}{marker}"
        )
    print(f"\n{lodging.reasoning}")


def _print_attractions(state: TripState) -> None:
    """Print the pool of candidate places."""
    attractions = state.get("attractions")
    if attractions is None:
        return

    _heading("6. ATTRACTIONS")
    print(f"{len(attractions.places)} places found\n")
    for place in attractions.places:
        cost = f"{place.estimated_cost} {place.currency}" if place.estimated_cost else "free"
        print(f"  - {place.name} ({place.category}) | {cost} | {place.opening_hours}")


def _print_routing(state: TripState) -> None:
    """Print the day-by-day plan."""
    routing = state.get("routing")
    if routing is None:
        return

    _heading("7. ROUTING")
    for day in routing.days:
        print(f"\nDay {day.day_number} - {day.date}: {day.summary}")
        for stop in day.stops:
            travel = (
                f" (+{stop.travel_minutes_from_previous} min {stop.travel_mode})"
                if stop.travel_minutes_from_previous
                else ""
            )
            print(f"  {stop.start_time}-{stop.end_time}  {stop.name}{travel}")
    if routing.unscheduled_place_ids:
        print(f"\nDid not fit: {', '.join(routing.unscheduled_place_ids)}")


def _print_budget(state: TripState) -> None:
    """Print the cost breakdown against the budget."""
    budget = state.get("budget")
    if budget is None:
        return

    _heading("8. BUDGET")
    for line in budget.lines:
        print(f"  {line.category:<16} {line.amount:>10.2f} {budget.currency}  {line.detail}")
    print(f"  {'TOTAL':<16} {budget.total_cost:>10.2f} {budget.currency}")
    if budget.budget_amount:
        verdict = "within budget" if budget.within_budget else f"OVER by {budget.overage}"
        print(f"  Budget: {budget.budget_amount} {budget.currency} -> {verdict}")
    if budget.savings_suggestions:
        print("\nWays to save:")
        for suggestion in budget.savings_suggestions:
            print(f"  - {suggestion}")


def _print_critic(state: TripState) -> None:
    """Print the Critic's issues and verdict."""
    critic = state.get("critic")
    if critic is None:
        return

    _heading("9. CRITIC")
    print(f"Approved: {critic.approved}")
    for issue in critic.issues:
        print(f"  [{issue.severity}] {issue.location}: {issue.description}")
        if issue.suggested_fix:
            print(f"      fix: {issue.suggested_fix}")
    print(f"\n{critic.reasoning}")


def _print_itinerary(state: TripState) -> None:
    """Print the final traveler-facing plan."""
    itinerary = state.get("itinerary")
    if itinerary is None:
        return

    _heading("10. ITINERARY")
    print(itinerary.markdown or itinerary.overview)


def _print_calendar(state: TripState) -> None:
    """Print the exported calendar events."""
    calendar = state.get("calendar")
    if calendar is None:
        return

    _heading("11. CALENDAR")
    for event in calendar.events:
        print(f"  [{event.category}] {event.start} -> {event.end}  {event.title}")
    if calendar.ics_path:
        print(f"\nCalendar file: {calendar.ics_path}")


def _print_efficiency(state: TripState, wall: float) -> None:
    """Print what the run cost in time, calls and money.

    Args:
        state: The final trip state.
        wall: Measured wall-clock seconds for the whole run.
    """
    records = state.get("metrics", [])
    if not records:
        return

    summary = summarize(records, wall_seconds=wall)
    _heading("EFFICIENCY")
    print(f"{'agent':<24}{'secs':>8}{'llm':>6}{'tools':>7}{'tokens':>10}{'cost':>10}")
    print("-" * 65)
    for record in summary.agents:
        cost = f"${record.cost_usd:.4f}" if record.cost_usd is not None else "-"
        flag = "  FAILED" if record.failed else ""
        print(
            f"{record.agent:<24}{record.seconds:>8.1f}{record.llm_calls:>6}"
            f"{record.tool_calls:>7}{record.total_tokens:>10,}{cost:>10}{flag}"
        )
    print("-" * 65)
    total = f"${summary.cost_usd:.4f}" if summary.cost_usd is not None else "-"
    print(
        f"{'TOTAL':<24}{summary.seconds:>8.1f}{summary.llm_calls:>6}"
        f"{summary.tool_calls:>7}"
        f"{summary.input_tokens + summary.output_tokens:>10,}{total:>10}"
    )
    print(
        f"\nWall clock: {summary.wall_seconds:.1f}s "
        f"(parallelism saved {summary.parallel_saving:.1f}s)"
    )

    failed = state.get("failed_agents", [])
    if failed:
        print(f"\nDEGRADED - {len(failed)} stage(s) failed:")
        for entry in failed:
            print(f"  - {entry['agent']}: {entry['error'][:150]}")


def main() -> None:
    """Plan a trip and print the result of each agent."""
    request = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REQUEST

    print(f"Model:   {describe_models()}")
    print(f"Request: {request}")

    started = time.perf_counter()
    final = plan_trip(request)
    wall = time.perf_counter() - started

    completed = ", ".join(str(name) for name in final.get("completed_agents", []))
    print(f"\nAgents that ran: {completed or 'none'}")
    if final.get("revision_count"):
        print(f"Plan revisions: {final['revision_count']}")

    _print_intake(final)
    _print_research(final)
    _print_flights(final)
    _print_lodging(final)
    _print_attractions(final)
    _print_routing(final)
    _print_budget(final)
    _print_critic(final)
    _print_itinerary(final)
    _print_calendar(final)

    _print_efficiency(final, wall)

    itinerary = final.get("itinerary")
    if itinerary and itinerary.days:
        print(f"\nPDF: {to_pdf(itinerary)}")

    decision = final.get("manager_decision")
    if decision:
        _heading("1. MANAGER (final decision)")
        print(decision.reasoning)


if __name__ == "__main__":
    main()
