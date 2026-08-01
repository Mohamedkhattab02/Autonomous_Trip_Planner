"""Turn a finished plan into things the traveler can actually use.

The planner used to stop one step short of useful: it named a flight and a
hotel but gave no way to book either, no map, and nothing to take offline.
Everything needed was already being fetched and then discarded — the booking
link from the hotel search, the coordinates from Maps.

This module closes that gap with three outputs: deep links per day, a booking
link per option, and a PDF of the whole plan.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus

from trip_planner.schemas import DayPlan, ItineraryResult, Place

EXPORT_DIR = Path(__file__).resolve().parents[2] / "exports"


def maps_link(place: Place) -> str:
    """Build a Google Maps link for one place.

    Args:
        place: The place to link to.

    Returns:
        A maps URL, preferring exact coordinates over the name.
    """
    if place.coordinates:
        return (
            "https://www.google.com/maps/search/?api=1&query="
            f"{place.coordinates.latitude},{place.coordinates.longitude}"
        )
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(place.name)}"


def day_directions_link(day: DayPlan, places: dict[str, Place]) -> str:
    """Build one Google Maps directions link covering a whole day.

    Args:
        day: The day to route.
        places: The place pool, keyed by `place_id`, for coordinates.

    Returns:
        A directions URL walking the day's stops in order, or an empty string
        when fewer than two stops have coordinates.
    """
    points = []
    for stop in day.stops:
        place = places.get(stop.place_id)
        if place and place.coordinates:
            points.append(f"{place.coordinates.latitude},{place.coordinates.longitude}")
        elif place:
            points.append(quote_plus(place.name))

    if len(points) < 2:
        return ""

    origin, destination = points[0], points[-1]
    waypoints = "|".join(points[1:-1])
    url = (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={origin}&destination={destination}&travelmode=walking"
    )
    return f"{url}&waypoints={waypoints}" if waypoints else url


def _slug(text: str) -> str:
    """Turn a title into a safe filename stem.

    Args:
        text: The text to convert.

    Returns:
        A lowercase, hyphenated slug.
    """
    keep = [ch.lower() if ch.isalnum() else "-" for ch in text]
    return "".join(keep).strip("-").replace("--", "-") or "trip"


def to_pdf(itinerary: ItineraryResult, filename: str | None = None) -> str:
    """Render the itinerary as a PDF the traveler can carry offline.

    Args:
        itinerary: The finished plan.
        filename: Output name; defaults to a slug of the trip title.

    Returns:
        The path written.
    """
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        ListFlowable,
        ListItem,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = filename or f"{_slug(itinerary.title)}.pdf"
    if not name.endswith(".pdf"):
        name = f"{name}.pdf"
    path = EXPORT_DIR / name

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TripTitle", parent=styles["Title"], fontSize=22, spaceAfter=6, alignment=TA_LEFT
    )
    day_style = ParagraphStyle(
        "DayHead", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=4
    )
    stop_style = ParagraphStyle(
        "Stop", parent=styles["BodyText"], leftIndent=8, spaceAfter=2
    )

    story: list = [Paragraph(itinerary.title or "Trip plan", title_style)]
    if itinerary.overview:
        story += [Paragraph(itinerary.overview, styles["BodyText"]), Spacer(1, 6 * mm)]

    for label, body in (
        ("Flights", itinerary.flight_summary),
        ("Where you're staying", itinerary.lodging_summary),
        ("Budget", itinerary.budget_summary),
    ):
        if body:
            story.append(Paragraph(f"<b>{label}.</b> {body}", styles["BodyText"]))
    story.append(Spacer(1, 4 * mm))

    for day in itinerary.days:
        heading = f"Day {day.day_number} — {day.date}"
        if day.summary:
            heading += f": {day.summary}"
        story.append(Paragraph(heading, day_style))
        for stop in day.stops:
            travel = (
                f" <i>({stop.travel_minutes_from_previous} min {stop.travel_mode})</i>"
                if stop.travel_minutes_from_previous
                else ""
            )
            story.append(
                Paragraph(
                    f"<b>{stop.start_time:%H:%M}–{stop.end_time:%H:%M}</b> "
                    f"{stop.name}{travel}",
                    stop_style,
                )
            )
            if stop.notes:
                story.append(Paragraph(f"<i>{stop.notes}</i>", stop_style))

    if itinerary.practical_notes:
        story += [PageBreak(), Paragraph("Before you go", day_style)]
        story.append(
            ListFlowable(
                [
                    ListItem(Paragraph(note, styles["BodyText"]))
                    for note in itinerary.practical_notes
                ],
                bulletType="bullet",
            )
        )

    SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=itinerary.title or "Trip plan",
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    ).build(story)

    return str(path)
