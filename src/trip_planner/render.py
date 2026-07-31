"""Turn agent results into the Markdown the frontend displays.

Kept out of the Gradio app so the presentation logic can be tested without a
browser, and so the CLI and the UI describe a trip the same way.
"""

from __future__ import annotations

from trip_planner.schemas import (
    AgentName,
    AttractionsResult,
    BudgetResult,
    CalendarResult,
    CriticResult,
    DestinationResearch,
    FlightsResult,
    IntakeResult,
    ItineraryResult,
    LodgingResult,
    RoutingResult,
    Severity,
)

# The agents shown in the progress tracker, in running order.
PIPELINE: tuple[tuple[AgentName, str, str], ...] = (
    (AgentName.INTAKE, "Intake", "📋"),
    (AgentName.DESTINATION_RESEARCH, "Research", "🌍"),
    (AgentName.FLIGHTS, "Flights", "✈️"),
    (AgentName.LODGING, "Lodging", "🏨"),
    (AgentName.ATTRACTIONS, "Attractions", "📍"),
    (AgentName.ROUTING, "Routing", "🗺"),
    (AgentName.BUDGET, "Budget", "💰"),
    (AgentName.CRITIC, "Critic", "🔍"),
    (AgentName.ITINERARY, "Itinerary", "📖"),
    (AgentName.CALENDAR, "Calendar", "📅"),
)

_SEVERITY_ICON = {
    Severity.BLOCKER: "🛑",
    Severity.WARNING: "⚠️",
    Severity.INFO: "💡",
}

_EMPTY = "_Nothing here yet — this stage has not run._"


def progress_html(completed: list[AgentName], active: str | None = None) -> str:
    """Render the eleven-stage progress tracker.

    Args:
        completed: Agents that have finished.
        active: The node currently running, if any.

    Returns:
        An HTML fragment of status chips.
    """
    done = set(completed)
    chips = []
    for name, label, icon in PIPELINE:
        if name in done:
            state = "done"
        elif active == str(name):
            state = "active"
        else:
            state = "idle"
        chips.append(
            f'<div class="chip chip-{state}">'
            f'<span class="chip-icon">{icon}</span>'
            f'<span class="chip-label">{label}</span>'
            f"</div>"
        )
    return f'<div class="pipeline">{"".join(chips)}</div>'


def render_profile(intake: IntakeResult | None) -> str:
    """Render the traveler profile.

    Args:
        intake: The Intake Agent's output.

    Returns:
        Markdown describing the trip requirements.
    """
    if intake is None:
        return _EMPTY

    profile = intake.profile
    rows = [
        ("Destination", profile.destination),
        ("From", profile.origin),
        ("Dates", f"{profile.start_date} → {profile.end_date}"),
        ("Travelers", profile.travelers),
        (
            "Budget",
            f"{profile.budget_amount:,.0f} {profile.budget_currency or ''}".strip()
            if profile.budget_amount
            else None,
        ),
        ("Interests", ", ".join(profile.interests) or None),
        ("Constraints", ", ".join(profile.constraints) or None),
    ]
    lines = ["| | |", "|---|---|"]
    lines += [
        f"| **{label}** | {value} |" for label, value in rows if value not in (None, "")
    ]

    if intake.clarifying_question:
        lines.append("")
        lines.append(f"> ❓ **I need to know:** {intake.clarifying_question}")
    return "\n".join(lines)


def render_research(research: DestinationResearch | None) -> str:
    """Render the destination briefing with its sources.

    Args:
        research: The Destination Research Agent's output.

    Returns:
        Markdown describing the destination.
    """
    if research is None:
        return _EMPTY

    sections = [
        ("🌤 Weather", research.weather),
        ("🛡 Safety", research.safety),
        ("💱 Currency", research.currency),
        ("🚇 Getting around", research.transportation),
        ("🛂 Entry requirements", research.entry_requirements),
    ]
    parts = [research.summary, ""]
    parts += [f"**{title}**  \n{body}\n" for title, body in sections if body]

    if research.sources:
        parts.append("**Sources**")
        parts += [f"- [{s.title}]({s.url})" for s in research.sources]
    return "\n".join(parts)


def render_flights(flights: FlightsResult | None) -> str:
    """Render the ranked flight options.

    Args:
        flights: The Flights Agent's output.

    Returns:
        Markdown describing each option.
    """
    if flights is None:
        return _EMPTY
    if not flights.options:
        return f"_No flights found._\n\n{flights.reasoning}"

    parts = []
    for option in flights.options:
        chosen = option.option_id == flights.recommended_option_id
        badge = " ⭐ **Recommended**" if chosen else ""
        hours, minutes = divmod(option.total_duration_minutes, 60)
        stops = "Direct" if option.stops == 0 else f"{option.stops} stop(s)"
        parts.append(
            f"#### {option.price:,.0f} {option.currency} · {stops} · "
            f"{hours}h {minutes:02d}m{badge}"
        )
        for leg in option.legs:
            parts.append(
                f"- `{leg.departure_airport} → {leg.arrival_airport}` "
                f"{leg.airline} {leg.flight_number} · "
                f"{leg.departure_time} → {leg.arrival_time}"
            )
        if option.baggage and option.baggage != "not stated":
            parts.append(f"- 🧳 {option.baggage}")
        for pro in option.pros:
            parts.append(f"- ✅ {pro}")
        for con in option.cons:
            parts.append(f"- ⚠️ {con}")
        parts.append("")

    parts.append(f"> {flights.reasoning}")
    return "\n".join(parts)


def render_lodging(lodging: LodgingResult | None) -> str:
    """Render the ranked lodging options.

    Args:
        lodging: The Lodging Agent's output.

    Returns:
        Markdown describing each stay.
    """
    if lodging is None:
        return _EMPTY
    if not lodging.options:
        return f"_No stays found._\n\n{lodging.reasoning}"

    parts = ["| | Stay | Total | Rating | |", "|---|---|---|---|---|"]
    for option in lodging.options:
        chosen = "⭐" if option.option_id == lodging.recommended_option_id else ""
        rating = f"{option.rating}/5" if option.rating else "—"
        stars = "★" * option.stars if option.stars else ""
        parts.append(
            f"| {chosen} | **{option.name}** | "
            f"{option.total_price:,.0f} {option.currency} | {rating} | {stars} |"
        )

    chosen_option = next(
        (o for o in lodging.options if o.option_id == lodging.recommended_option_id),
        None,
    )
    if chosen_option:
        parts.append("")
        parts.append(f"#### ⭐ {chosen_option.name}")
        if chosen_option.address:
            parts.append(f"📍 {chosen_option.address}")
        for pro in chosen_option.pros:
            parts.append(f"- ✅ {pro}")
        for con in chosen_option.cons:
            parts.append(f"- ⚠️ {con}")

    parts.append("")
    parts.append(f"> {lodging.reasoning}")
    return "\n".join(parts)


def render_places(attractions: AttractionsResult | None) -> str:
    """Render the pool of candidate places.

    Args:
        attractions: The Attractions Agent's output.

    Returns:
        Markdown listing every place found.
    """
    if attractions is None:
        return _EMPTY
    if not attractions.places:
        return "_No places found._"

    parts = [
        f"**{len(attractions.places)} places found**",
        "",
        "| Place | Type | Cost | Visit | Closed |",
        "|---|---|---|---|---|",
    ]
    for place in attractions.places:
        cost = (
            f"{place.estimated_cost:,.0f} {place.currency}"
            if place.estimated_cost
            else "Free"
        )
        closed = ", ".join(place.closed_days) if place.closed_days else "—"
        parts.append(
            f"| **{place.name}** | {place.category} | {cost} | "
            f"{place.visit_duration_minutes} min | {closed} |"
        )
    parts.append("")
    parts.append(f"> {attractions.reasoning}")
    return "\n".join(parts)


def render_itinerary(
    itinerary: ItineraryResult | None, routing: RoutingResult | None
) -> str:
    """Render the traveler-facing plan, falling back to the routed days.

    Args:
        itinerary: The Itinerary Agent's finished write-up.
        routing: The routed day plan, shown while the write-up is pending.

    Returns:
        Markdown of the full trip plan.
    """
    if itinerary is not None and itinerary.markdown:
        return itinerary.markdown

    if routing is not None and routing.days:
        parts = ["## Day by day", ""]
        for day in routing.days:
            heading = f"### Day {day.day_number} — {day.date}"
            if day.summary:
                heading += f": {day.summary}"
            parts.append(heading)
            for stop in day.stops:
                travel = (
                    f" _({stop.travel_minutes_from_previous} min {stop.travel_mode})_"
                    if stop.travel_minutes_from_previous
                    else ""
                )
                parts.append(
                    f"- **{stop.start_time:%H:%M}–{stop.end_time:%H:%M}** "
                    f"{stop.name}{travel}"
                )
            parts.append("")
        return "\n".join(parts)

    return _EMPTY


def render_budget(budget: BudgetResult | None) -> str:
    """Render the cost breakdown against the budget.

    Args:
        budget: The Budget Agent's output.

    Returns:
        Markdown describing the budget position.
    """
    if budget is None:
        return _EMPTY

    parts = ["| Category | Amount | Detail |", "|---|---:|---|"]
    for line in budget.lines:
        parts.append(
            f"| {line.category.replace('_', ' ').title()} | "
            f"{line.amount:,.2f} {budget.currency} | {line.detail} |"
        )
    parts.append(f"| **Total** | **{budget.total_cost:,.2f} {budget.currency}** | |")
    parts.append("")

    if budget.budget_amount:
        if budget.within_budget:
            remaining = budget.budget_amount - budget.total_cost
            parts.append(
                f"> ✅ **Within budget** — {remaining:,.2f} {budget.currency} "
                f"left of {budget.budget_amount:,.0f}."
            )
        else:
            parts.append(
                f"> 🛑 **Over budget** by {budget.overage:,.2f} {budget.currency} "
                f"(budget was {budget.budget_amount:,.0f})."
            )

    if budget.savings_suggestions:
        parts.append("")
        parts.append("**Ways to save**")
        parts += [f"- 💡 {s}" for s in budget.savings_suggestions]
    return "\n".join(parts)


def render_critic(critic: CriticResult | None) -> str:
    """Render the Critic's verdict and issues.

    Args:
        critic: The Critic Agent's output.

    Returns:
        Markdown describing what the review found.
    """
    if critic is None:
        return _EMPTY

    verdict = (
        "> ✅ **Approved** — the plan is real, possible and within limits."
        if critic.approved
        else f"> 🛑 **Sent back for revision** — {len(critic.blockers)} blocker(s) found."
    )
    parts = [verdict, ""]

    if critic.issues:
        for issue in critic.issues:
            icon = _SEVERITY_ICON.get(issue.severity, "•")
            where = f" _({issue.location})_" if issue.location else ""
            parts.append(f"- {icon} **{issue.category}**{where}: {issue.description}")
            if issue.suggested_fix:
                parts.append(f"  - 🔧 {issue.suggested_fix}")
    else:
        parts.append("_No issues found._")

    parts.append("")
    parts.append(critic.reasoning)
    return "\n".join(parts)


def render_calendar(calendar: CalendarResult | None, backend: str) -> str:
    """Render the exported calendar events.

    Args:
        calendar: The Calendar Agent's output.
        backend: Description of the active calendar backend.

    Returns:
        Markdown listing every event created.
    """
    if calendar is None:
        return f"_Backend: {backend}_\n\n{_EMPTY}"

    parts = [f"_Backend: **{backend}**_", ""]
    if calendar.events:
        parts += ["| | Event | Starts | Ends |", "|---|---|---|---|"]
        icons = {"flight": "✈️", "lodging": "🏨", "activity": "📍"}
        for event in calendar.events:
            icon = icons.get(event.category, "📌")
            parts.append(
                f"| {icon} | **{event.title}** | {event.start} | {event.end} |"
            )
    else:
        parts.append("_No events were created._")

    if calendar.ics_path:
        parts.append("")
        parts.append(f"📎 Calendar file: `{calendar.ics_path}`")

    parts.append("")
    parts.append(f"> {calendar.reasoning}")
    return "\n".join(parts)
