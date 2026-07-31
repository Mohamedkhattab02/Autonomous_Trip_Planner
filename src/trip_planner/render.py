"""Turn agent results into the HTML panels the frontend displays.

Kept out of the Gradio app so the presentation logic can be tested without a
browser, and so every agent's output is described the same way everywhere.

Each agent gets a panel built from the same small set of components — stat
tiles, cards, badges and meters — so the traveler learns one visual language
and can then read any stage at a glance. Colours come from the validated
data-viz palette (see the `.tp-*` custom properties in `app.py`): status hues
for state, categorical slots for the budget breakdown, and never colour alone —
every status carries an icon and a label too.

All agent-produced text is escaped before it reaches the markup.
"""

from __future__ import annotations

from html import escape

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

# Severity → (icon, label, status class). The icon and label are what carry the
# meaning; the colour only reinforces it.
_SEVERITY = {
    Severity.BLOCKER: ("🛑", "Blocker", "critical"),
    Severity.WARNING: ("⚠️", "Warning", "warning"),
    Severity.INFO: ("💡", "Note", "info"),
}

# Budget categories in a fixed order, each pinned to a categorical slot so a
# category never changes colour between runs.
_BUDGET_SLOTS = {
    "flights": (1, "✈️", "Flights"),
    "lodging": (2, "🏨", "Lodging"),
    "activities": (3, "🎟", "Activities"),
    "food": (4, "🍽", "Food"),
    "local_transport": (5, "🚇", "Local transport"),
}


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------


def _esc(value: object) -> str:
    """Escape a value for safe inclusion in HTML.

    Args:
        value: Anything renderable; agent text is never trusted as markup.

    Returns:
        The value as escaped text.
    """
    return escape(str(value if value is not None else ""), quote=True)


def _empty(message: str = "This stage has not run yet.") -> str:
    """Render the placeholder shown before an agent has produced anything.

    Args:
        message: What to tell the traveler.

    Returns:
        An HTML fragment.
    """
    return f'<div class="tp-empty"><span>⏳</span> {_esc(message)}</div>'


def _stats(items: list[tuple[str, str, str]]) -> str:
    """Render a row of stat tiles.

    Args:
        items: `(label, value, sub)` triples. `sub` may be empty.

    Returns:
        An HTML fragment.
    """
    tiles = "".join(
        f'<div class="tp-stat"><div class="tp-stat-label">{_esc(label)}</div>'
        f'<div class="tp-stat-value">{_esc(value)}</div>'
        + (f'<div class="tp-stat-sub">{_esc(sub)}</div>' if sub else "")
        + "</div>"
        for label, value, sub in items
    )
    return f'<div class="tp-stats">{tiles}</div>'


def _badge(text: str, kind: str = "neutral") -> str:
    """Render a small pill.

    Args:
        text: The label, already meaningful without its colour.
        kind: One of "neutral", "good", "warning", "critical", "accent".

    Returns:
        An HTML fragment.
    """
    return f'<span class="tp-badge tp-badge-{kind}">{_esc(text)}</span>'


def _note(text: str) -> str:
    """Render an agent's reasoning line beneath its results.

    Args:
        text: The agent's explanation.

    Returns:
        An HTML fragment, or an empty string when there is nothing to say.
    """
    if not text:
        return ""
    return f'<div class="tp-note"><span class="tp-note-icon">💬</span>{_esc(text)}</div>'


def _section(title: str, body: str) -> str:
    """Wrap a body in a titled section.

    Args:
        title: The section heading.
        body: Already-rendered HTML.

    Returns:
        An HTML fragment.
    """
    return f'<div class="tp-section"><h3 class="tp-h3">{_esc(title)}</h3>{body}</div>'


def _minutes(total: int) -> str:
    """Format a duration in minutes as "5h 20m".

    Args:
        total: Duration in minutes.

    Returns:
        A compact human-readable duration.
    """
    hours, minutes = divmod(int(total), 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


def _money(amount: float, currency: str) -> str:
    """Format an amount with its currency.

    Args:
        amount: The value.
        currency: ISO 4217 code.

    Returns:
        A formatted amount, e.g. "3,000 USD".
    """
    return f"{amount:,.0f} {currency}".strip()


def _stars(rating: float | None) -> str:
    """Render a 0-5 rating as filled and empty stars.

    Args:
        rating: The rating, or None.

    Returns:
        An HTML fragment, or an empty string when there is no rating.
    """
    if not rating:
        return ""
    filled = int(round(rating))
    return (
        f'<span class="tp-stars">{"★" * filled}{"☆" * (5 - filled)}'
        f'<span class="tp-stars-num">{rating:g}</span></span>'
    )


# --------------------------------------------------------------------------
# Progress tracker
# --------------------------------------------------------------------------


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
            f'<span class="chip-label">{_esc(label)}</span>'
            f"</div>"
        )
    return f'<div class="pipeline">{"".join(chips)}</div>'


# --------------------------------------------------------------------------
# 📋 Your request
# --------------------------------------------------------------------------


def render_profile(intake: IntakeResult | None) -> str:
    """Render the traveler profile the Intake Agent extracted.

    Args:
        intake: The Intake Agent's output.

    Returns:
        An HTML panel.
    """
    if intake is None:
        return _empty("Waiting for the Intake Agent to read your request.")

    profile = intake.profile
    nights = (
        (profile.end_date - profile.start_date).days
        if profile.start_date and profile.end_date
        else None
    )
    parts = [
        _stats(
            [
                (
                    "Destination",
                    profile.destination or "—",
                    f"from {profile.origin}" if profile.origin else "",
                ),
                (
                    "Dates",
                    f"{profile.start_date or '—'}" if not nights else f"{nights + 1} days",
                    f"{profile.start_date} → {profile.end_date}" if nights else "",
                ),
                ("Travelers", str(profile.travelers or "—"), ""),
                (
                    "Budget",
                    _money(profile.budget_amount, profile.budget_currency or "")
                    if profile.budget_amount
                    else "—",
                    "total, all travelers" if profile.budget_amount else "",
                ),
            ]
        )
    ]

    if profile.interests:
        chips = "".join(_badge(i, "accent") for i in profile.interests)
        parts.append(_section("Interests", f'<div class="tp-chips">{chips}</div>'))

    if profile.constraints:
        chips = "".join(_badge(c, "warning") for c in profile.constraints)
        parts.append(_section("Must respect", f'<div class="tp-chips">{chips}</div>'))

    if intake.missing_fields:
        missing = ", ".join(intake.missing_fields)
        parts.append(
            f'<div class="tp-callout tp-callout-warning">'
            f'<div class="tp-callout-title">❓ I need more before I can plan this</div>'
            f"<p>{_esc(intake.clarifying_question or '')}</p>"
            f'<p class="tp-dim">Missing: {_esc(missing)}</p></div>'
        )
    else:
        parts.append(
            '<div class="tp-callout tp-callout-good">'
            '<div class="tp-callout-title">✅ Everything needed is here</div>'
            "<p>The planner has enough to book flights, a stay and a route.</p></div>"
        )

    return "".join(parts)


# --------------------------------------------------------------------------
# 🌍 Destination
# --------------------------------------------------------------------------


def render_research(research: DestinationResearch | None) -> str:
    """Render the destination briefing with its sources.

    Args:
        research: The Destination Research Agent's output.

    Returns:
        An HTML panel.
    """
    if research is None:
        return _empty("Waiting for the Destination Research Agent.")

    cards = "".join(
        f'<div class="tp-card tp-brief"><div class="tp-brief-icon">{icon}</div>'
        f'<div><div class="tp-brief-title">{_esc(title)}</div>'
        f'<div class="tp-brief-body">{_esc(body)}</div></div></div>'
        for icon, title, body in (
            ("🌤", "Weather", research.weather),
            ("🛡", "Safety", research.safety),
            ("💱", "Currency", research.currency),
            ("🚇", "Getting around", research.transportation),
            ("🛂", "Entry requirements", research.entry_requirements),
        )
        if body
    )

    parts = [
        f'<div class="tp-lede">{_esc(research.summary)}</div>',
        f'<div class="tp-grid">{cards}</div>',
    ]

    if research.sources:
        links = "".join(
            f'<a class="tp-source" href="{_esc(s.url)}" target="_blank" '
            f'rel="noopener noreferrer">🔗 {_esc(s.title or s.url)}</a>'
            for s in research.sources
        )
        parts.append(
            _section(
                f"Sources ({len(research.sources)})", f'<div class="tp-sources">{links}</div>'
            )
        )
    return "".join(parts)


# --------------------------------------------------------------------------
# ✈️ Flights
# --------------------------------------------------------------------------


def render_flights(flights: FlightsResult | None) -> str:
    """Render the ranked flight options.

    Args:
        flights: The Flights Agent's output.

    Returns:
        An HTML panel.
    """
    if flights is None:
        return _empty("Waiting for the Flights Agent.")
    if not flights.options:
        return (
            '<div class="tp-callout tp-callout-warning">'
            '<div class="tp-callout-title">✈️ No flights found</div>'
            f"<p>{_esc(flights.reasoning)}</p></div>"
        )

    best = min(option.price for option in flights.options)
    fastest = min(option.total_duration_minutes or 0 for option in flights.options)
    currency = flights.options[0].currency

    parts = [
        _stats(
            [
                ("Options found", str(len(flights.options)), "ranked best first"),
                ("Cheapest", _money(best, currency), "total, all travelers"),
                ("Fastest", _minutes(fastest), "door to door"),
            ]
        )
    ]

    for option in flights.options:
        chosen = option.option_id == flights.recommended_option_id
        badges = [
            _badge("Direct" if option.stops == 0 else f"{option.stops} stop(s)",
                   "good" if option.stops == 0 else "neutral"),
            _badge(_minutes(option.total_duration_minutes)),
        ]
        if option.baggage and option.baggage != "not stated":
            badges.append(_badge(f"🧳 {option.baggage}"))
        if chosen:
            badges.insert(0, _badge("⭐ Recommended", "accent"))

        legs = "".join(
            f'<div class="tp-leg">'
            f'<div class="tp-leg-route">'
            f'<span class="tp-iata">{_esc(leg.departure_airport)}</span>'
            f'<span class="tp-leg-line"><span class="tp-leg-plane">✈</span></span>'
            f'<span class="tp-iata">{_esc(leg.arrival_airport)}</span></div>'
            f'<div class="tp-leg-meta">{_esc(leg.airline)} {_esc(leg.flight_number)}'
            f' · {_esc(leg.departure_time)} → {_esc(leg.arrival_time)}'
            f' · {_esc(_minutes(leg.duration_minutes))}</div></div>'
            for leg in option.legs
        )

        pros = "".join(f"<li>✅ {_esc(p)}</li>" for p in option.pros)
        cons = "".join(f"<li>⚠️ {_esc(c)}</li>" for c in option.cons)
        tradeoffs = (
            f'<ul class="tp-tradeoffs">{pros}{cons}</ul>' if (pros or cons) else ""
        )

        parts.append(
            f'<div class="tp-card{" tp-card-chosen" if chosen else ""}">'
            f'<div class="tp-card-head">'
            f'<div class="tp-price">{_esc(_money(option.price, option.currency))}</div>'
            f'<div class="tp-badges">{"".join(badges)}</div></div>'
            f'<div class="tp-legs">{legs}</div>{tradeoffs}</div>'
        )

    parts.append(_note(flights.reasoning))
    return "".join(parts)


# --------------------------------------------------------------------------
# 🏨 Stay
# --------------------------------------------------------------------------


def render_lodging(lodging: LodgingResult | None) -> str:
    """Render the ranked lodging options.

    Args:
        lodging: The Lodging Agent's output.

    Returns:
        An HTML panel.
    """
    if lodging is None:
        return _empty("Waiting for the Lodging Agent.")
    if not lodging.options:
        return (
            '<div class="tp-callout tp-callout-warning">'
            '<div class="tp-callout-title">🏨 No stays found</div>'
            f"<p>{_esc(lodging.reasoning)}</p></div>"
        )

    priced = [o for o in lodging.options if o.total_price]
    cheapest = min((o.total_price for o in priced), default=0)
    currency = lodging.options[0].currency
    rated = [o.rating for o in lodging.options if o.rating]

    parts = [
        _stats(
            [
                ("Stays found", str(len(lodging.options)), "ranked best first"),
                ("From", _money(cheapest, currency), "whole stay"),
                (
                    "Best rated",
                    f"{max(rated):g}/5" if rated else "—",
                    "guest rating" if rated else "",
                ),
            ]
        )
    ]

    for option in lodging.options:
        chosen = option.option_id == lodging.recommended_option_id
        badges = []
        if chosen:
            badges.append(_badge("⭐ Recommended", "accent"))
        if option.kind:
            badges.append(_badge(option.kind))
        if option.stars:
            badges.append(_badge("★" * option.stars))

        meta = []
        if option.price_per_night:
            meta.append(f"{_money(option.price_per_night, option.currency)} / night")
        if option.address:
            meta.append(f"📍 {option.address}")

        pros = "".join(f"<li>✅ {_esc(p)}</li>" for p in option.pros)
        cons = "".join(f"<li>⚠️ {_esc(c)}</li>" for c in option.cons)
        tradeoffs = (
            f'<ul class="tp-tradeoffs">{pros}{cons}</ul>' if (pros or cons) else ""
        )

        parts.append(
            f'<div class="tp-card{" tp-card-chosen" if chosen else ""}">'
            f'<div class="tp-card-head">'
            f'<div><div class="tp-card-title">{_esc(option.name)}</div>'
            f'<div class="tp-card-meta">{_esc(" · ".join(meta))}</div></div>'
            f'<div class="tp-price-sm">{_esc(_money(option.total_price, option.currency))}'
            f'<span class="tp-dim"> total</span></div></div>'
            f'<div class="tp-badges">{"".join(badges)}{_stars(option.rating)}</div>'
            f"{tradeoffs}</div>"
        )

    parts.append(_note(lodging.reasoning))
    return "".join(parts)


# --------------------------------------------------------------------------
# 📍 Places
# --------------------------------------------------------------------------


def render_places(attractions: AttractionsResult | None) -> str:
    """Render the pool of candidate places.

    Args:
        attractions: The Attractions Agent's output.

    Returns:
        An HTML panel.
    """
    if attractions is None:
        return _empty("Waiting for the Attractions Agent.")
    if not attractions.places:
        return _empty("No places were found.")

    places = attractions.places
    free = sum(1 for p in places if not p.estimated_cost)
    total_hours = sum(p.visit_duration_minutes for p in places) / 60

    parts = [
        _stats(
            [
                ("Places found", str(len(places)), "for the routing stage"),
                ("Free to visit", f"{free} of {len(places)}", "no entry cost"),
                ("Time to see all", f"{total_hours:.0f}h", "visits only, no travel"),
            ]
        )
    ]

    cards = []
    for place in places:
        cost = (
            _badge(_money(place.estimated_cost, place.currency))
            if place.estimated_cost
            else _badge("Free", "good")
        )
        closed = (
            _badge(f"Closed {', '.join(place.closed_days)}", "warning")
            if place.closed_days
            else ""
        )
        cards.append(
            f'<div class="tp-card tp-place">'
            f'<div class="tp-card-title">{_esc(place.name)}</div>'
            f'<div class="tp-card-meta">{_esc(place.category)}'
            + (f" · 📍 {_esc(place.address)}" if place.address else "")
            + "</div>"
            f'<div class="tp-badges">{cost}'
            f'{_badge(f"⏱ {_minutes(place.visit_duration_minutes)}")}'
            f"{closed}{_stars(place.rating)}</div>"
            + (
                f'<div class="tp-card-why">{_esc(place.why_recommended)}</div>'
                if place.why_recommended
                else ""
            )
            + (
                f'<div class="tp-dim tp-hours">🕐 {_esc(place.opening_hours)}</div>'
                if place.opening_hours and place.opening_hours != "unknown"
                else ""
            )
            + "</div>"
        )

    parts.append(f'<div class="tp-grid">{"".join(cards)}</div>')
    parts.append(_note(attractions.reasoning))
    return "".join(parts)


# --------------------------------------------------------------------------
# 📖 Itinerary
# --------------------------------------------------------------------------


def _day_timeline(day) -> str:
    """Render one day as a vertical timeline of stops.

    Args:
        day: A `DayPlan`.

    Returns:
        An HTML fragment.
    """
    if not day.stops:
        return '<div class="tp-dim">No activities scheduled.</div>'

    rows = []
    for stop in day.stops:
        travel = (
            f'<div class="tp-travel">🚶 {stop.travel_minutes_from_previous} min '
            f"{_esc(stop.travel_mode)}</div>"
            if stop.travel_minutes_from_previous
            else ""
        )
        rows.append(
            f"{travel}"
            f'<div class="tp-stop">'
            f'<div class="tp-stop-time">{stop.start_time:%H:%M}'
            f'<span class="tp-dim">{stop.end_time:%H:%M}</span></div>'
            f'<div class="tp-stop-dot"></div>'
            f'<div class="tp-stop-body"><div class="tp-stop-name">{_esc(stop.name)}</div>'
            + (
                f'<div class="tp-card-meta">{_esc(stop.notes)}</div>'
                if stop.notes
                else ""
            )
            + "</div></div>"
        )
    return f'<div class="tp-timeline">{"".join(rows)}</div>'


def _days_panel(days) -> str:
    """Render every day as a card with its timeline.

    Args:
        days: A list of `DayPlan`.

    Returns:
        An HTML fragment.
    """
    cards = []
    for day in days:
        cards.append(
            f'<div class="tp-card tp-day">'
            f'<div class="tp-day-head">'
            f'<span class="tp-day-num">Day {day.day_number}</span>'
            f'<span class="tp-day-date">{_esc(day.date)}</span>'
            + (f'<span class="tp-day-theme">{_esc(day.summary)}</span>' if day.summary else "")
            + "</div>"
            f"{_day_timeline(day)}</div>"
        )
    return "".join(cards)


def render_itinerary(
    itinerary: ItineraryResult | None, routing: RoutingResult | None
) -> str:
    """Render the traveler-facing plan, falling back to the routed days.

    Args:
        itinerary: The Itinerary Agent's finished write-up.
        routing: The routed day plan, shown while the write-up is pending.

    Returns:
        An HTML panel.
    """
    days = (
        itinerary.days
        if itinerary is not None and itinerary.days
        else routing.days
        if routing is not None
        else []
    )
    if not days:
        return _empty("Waiting for the Routing and Itinerary agents.")

    stops = sum(len(day.stops) for day in days)
    travel = sum(
        stop.travel_minutes_from_previous for day in days for stop in day.stops
    )

    parts = []
    if itinerary is not None and itinerary.title:
        parts.append(f'<div class="tp-hero">{_esc(itinerary.title)}</div>')
    if itinerary is not None and itinerary.overview:
        parts.append(f'<div class="tp-lede">{_esc(itinerary.overview)}</div>')

    parts.append(
        _stats(
            [
                ("Days", str(len(days)), "planned end to end"),
                ("Stops", str(stops), f"about {stops / max(len(days), 1):.0f} a day"),
                ("Time in transit", _minutes(travel), "between stops"),
            ]
        )
    )

    if itinerary is not None:
        summaries = "".join(
            f'<div class="tp-card tp-brief"><div class="tp-brief-icon">{icon}</div>'
            f'<div><div class="tp-brief-title">{_esc(title)}</div>'
            f'<div class="tp-brief-body">{_esc(body)}</div></div></div>'
            for icon, title, body in (
                ("✈️", "Flights", itinerary.flight_summary),
                ("🏨", "Where you're staying", itinerary.lodging_summary),
                ("💰", "Budget", itinerary.budget_summary),
            )
            if body
        )
        if summaries:
            parts.append(f'<div class="tp-grid">{summaries}</div>')

    parts.append(_section("Day by day", _days_panel(days)))

    if itinerary is not None and itinerary.practical_notes:
        notes = "".join(f"<li>{_esc(n)}</li>" for n in itinerary.practical_notes)
        parts.append(
            _section("Before you go", f'<ul class="tp-notes">{notes}</ul>')
        )

    return "".join(parts)


# --------------------------------------------------------------------------
# 💰 Budget
# --------------------------------------------------------------------------


def render_budget(budget: BudgetResult | None) -> str:
    """Render the cost breakdown against the budget.

    Args:
        budget: The Budget Agent's output.

    Returns:
        An HTML panel.
    """
    if budget is None:
        return _empty("Waiting for the Budget Agent.")

    lines = [line for line in budget.lines if line.amount > 0]
    total = budget.total_cost or sum(line.amount for line in lines) or 1

    # Stacked meter: one segment per category, in the fixed slot order so a
    # category keeps its colour between runs. Each is direct-labelled below.
    segments = []
    legend = []
    for line in sorted(
        lines, key=lambda entry: _BUDGET_SLOTS.get(entry.category, (9, "", ""))[0]
    ):
        slot, icon, label = _BUDGET_SLOTS.get(line.category, (8, "•", line.category))
        share = line.amount / total * 100
        segments.append(
            f'<div class="tp-seg tp-series-{slot}" style="width:{share:.4f}%" '
            f'title="{_esc(label)}: {_esc(_money(line.amount, budget.currency))}"></div>'
        )
        legend.append(
            f'<div class="tp-legend-row">'
            f'<span class="tp-swatch tp-series-{slot}"></span>'
            f'<span class="tp-legend-label">{icon} {_esc(label)}</span>'
            f'<span class="tp-legend-value">{_esc(_money(line.amount, budget.currency))}'
            f'<span class="tp-dim"> · {share:.0f}%</span></span>'
            + (
                f'<div class="tp-legend-detail">{_esc(line.detail)}</div>'
                if line.detail
                else ""
            )
            + "</div>"
        )

    used = (
        budget.total_cost / budget.budget_amount * 100 if budget.budget_amount else None
    )

    parts = [
        f'<div class="tp-hero">{_esc(_money(budget.total_cost, budget.currency))}</div>',
        f'<div class="tp-lede">Total cost of the trip'
        + (
            f", against a budget of {_esc(_money(budget.budget_amount, budget.currency))}"
            if budget.budget_amount
            else ""
        )
        + ".</div>",
    ]

    if budget.budget_amount:
        state = "good" if budget.within_budget else "critical"
        parts.append(
            f'<div class="tp-meter"><div class="tp-meter-track">'
            f'<div class="tp-meter-fill tp-meter-{state}" '
            f'style="width:{min(used, 100):.4f}%"></div></div>'
            f'<div class="tp-meter-scale"><span>{used:.0f}% of budget used</span>'
            f"<span>{_esc(_money(budget.budget_amount, budget.currency))}</span></div></div>"
        )
        if budget.within_budget:
            left = budget.budget_amount - budget.total_cost
            parts.append(
                '<div class="tp-callout tp-callout-good">'
                '<div class="tp-callout-title">✅ Within budget</div>'
                f"<p>{_esc(_money(left, budget.currency))} still unspent.</p></div>"
            )
        else:
            parts.append(
                '<div class="tp-callout tp-callout-critical">'
                '<div class="tp-callout-title">🛑 Over budget</div>'
                f"<p>This trip costs {_esc(_money(budget.overage, budget.currency))} "
                f"more than planned.</p></div>"
            )

    if segments:
        parts.append(
            _section(
                "Where the money goes",
                f'<div class="tp-stack">{"".join(segments)}</div>'
                f'<div class="tp-legend">{"".join(legend)}</div>',
            )
        )

    if budget.savings_suggestions:
        items = "".join(f"<li>💡 {_esc(s)}</li>" for s in budget.savings_suggestions)
        parts.append(_section("Ways to save", f'<ul class="tp-notes">{items}</ul>'))

    parts.append(_note(budget.reasoning))
    return "".join(parts)


# --------------------------------------------------------------------------
# 🔍 Review
# --------------------------------------------------------------------------


def render_critic(critic: CriticResult | None) -> str:
    """Render the Critic's verdict and issues.

    Args:
        critic: The Critic Agent's output.

    Returns:
        An HTML panel.
    """
    if critic is None:
        return _empty("Waiting for the Critic Agent to check the plan.")

    blockers = len(critic.blockers)
    warnings = sum(1 for i in critic.issues if i.severity == Severity.WARNING)
    notes = sum(1 for i in critic.issues if i.severity == Severity.INFO)

    if critic.approved:
        verdict = (
            '<div class="tp-callout tp-callout-good">'
            '<div class="tp-callout-title">✅ Approved</div>'
            "<p>Every place was verified as real, the schedule is possible, and no "
            "stated limit is broken.</p></div>"
        )
    else:
        verdict = (
            '<div class="tp-callout tp-callout-critical">'
            '<div class="tp-callout-title">🛑 Sent back for revision</div>'
            f"<p>{blockers} blocker(s) must be fixed before this plan is usable. "
            "The Routing Agent rebuilds the days to address them.</p></div>"
        )

    parts = [
        verdict,
        _stats(
            [
                ("Blockers", str(blockers), "must be fixed"),
                ("Warnings", str(warnings), "worth knowing"),
                ("Notes", str(notes), "suggestions"),
            ]
        ),
    ]

    if critic.issues:
        cards = []
        for issue in critic.issues:
            icon, label, kind = _SEVERITY.get(issue.severity, ("•", "Issue", "neutral"))
            cards.append(
                f'<div class="tp-card tp-issue tp-issue-{kind}">'
                f'<div class="tp-badges">{_badge(f"{icon} {label}", kind)}'
                f"{_badge(issue.category)}"
                + (_badge(f"📍 {issue.location}") if issue.location else "")
                + "</div>"
                f'<div class="tp-issue-body">{_esc(issue.description)}</div>'
                + (
                    f'<div class="tp-issue-fix">🔧 {_esc(issue.suggested_fix)}</div>'
                    if issue.suggested_fix
                    else ""
                )
                + "</div>"
            )
        parts.append(_section("What the review found", "".join(cards)))
    else:
        parts.append(_section("What the review found", _empty("No issues at all.")))

    parts.append(_note(critic.reasoning))
    return "".join(parts)


# --------------------------------------------------------------------------
# 📅 Calendar
# --------------------------------------------------------------------------


def render_calendar(calendar: CalendarResult | None, backend: str) -> str:
    """Render the exported calendar events.

    Args:
        calendar: The Calendar Agent's output.
        backend: Description of the active calendar backend.

    Returns:
        An HTML panel.
    """
    is_mcp = "MCP" in backend
    banner = (
        f'<div class="tp-backend">'
        f'<span class="tp-backend-icon">{"🔌" if is_mcp else "📎"}</span>'
        f'<div><div class="tp-backend-title">'
        f'{"Google Calendar — live over MCP" if is_mcp else "Local .ics export"}</div>'
        f'<div class="tp-dim">{_esc(backend)}</div></div></div>'
    )

    if calendar is None:
        return banner + _empty("Waiting for the Calendar Agent.")

    if not calendar.events:
        return banner + _empty("No events were created.")

    icons = {"flight": "✈️", "lodging": "🏨", "activity": "📍"}
    counts: dict[str, int] = {}
    for event in calendar.events:
        counts[event.category] = counts.get(event.category, 0) + 1

    rows = "".join(
        f'<div class="tp-event">'
        f'<span class="tp-event-icon">{icons.get(event.category, "📌")}</span>'
        f'<div class="tp-event-body"><div class="tp-event-title">{_esc(event.title)}</div>'
        f'<div class="tp-card-meta">{_esc(event.start)} → {_esc(event.end)}'
        + (f" · 📍 {_esc(event.location)}" if event.location else "")
        + "</div></div></div>"
        for event in calendar.events
    )

    parts = [
        banner,
        _stats(
            [
                ("Events created", str(len(calendar.events)), "on your calendar"),
                ("Flights", str(counts.get("flight", 0)), ""),
                ("Activities", str(counts.get("activity", 0)), ""),
            ]
        ),
        _section("Everything added", f'<div class="tp-events">{rows}</div>'),
    ]

    if calendar.ics_path:
        parts.append(
            f'<div class="tp-callout tp-callout-good">'
            f'<div class="tp-callout-title">📎 Calendar file ready</div>'
            f"<p>{_esc(calendar.ics_path)}</p></div>"
        )

    parts.append(_note(calendar.reasoning))
    return "".join(parts)
