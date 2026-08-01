"""Pydantic structured messages exchanged between agents.

Every agent produces exactly one of these models. They are the contract
between nodes: an agent never reads another agent's raw text, only its
validated structured output.
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import time
from enum import StrEnum

from pydantic import BaseModel, Field


class AgentName(StrEnum):
    """Names of the agents in the system, used for routing."""

    INTAKE = "intake"
    DESTINATION_RESEARCH = "destination_research"
    FLIGHTS = "flights"
    LODGING = "lodging"
    ATTRACTIONS = "attractions"
    ROUTING = "routing"
    BUDGET = "budget"
    CRITIC = "critic"
    ITINERARY = "itinerary"
    CALENDAR = "calendar"


# --------------------------------------------------------------------------
# 1. Travel Manager Agent
# --------------------------------------------------------------------------


class ManagerDecision(BaseModel):
    """The Travel Manager's routing decision for the next step."""

    next_agent: AgentName | None = Field(
        default=None,
        description=(
            "Which agent should run next. Null when every required stage is "
            "complete and the trip plan can be finalized."
        ),
    )
    next_agents: list[AgentName] = Field(
        default_factory=list,
        description=(
            "Every agent to dispatch now. Holds more than one when the agents "
            "are independent and can run concurrently."
        ),
    )
    reasoning: str = Field(
        description="Short explanation of why this agent was chosen, or why the work is done."
    )


# --------------------------------------------------------------------------
# 2. Intake Agent
# --------------------------------------------------------------------------


class TravelerProfile(BaseModel):
    """Organized trip requirements collected from the user's request."""

    destination: str | None = Field(
        default=None,
        description="Destination city and/or country, e.g. 'Paris, France'.",
    )
    origin: str | None = Field(
        default=None, description="City the travelers depart from."
    )
    start_date: date_type | None = Field(default=None, description="First day of the trip.")
    end_date: date_type | None = Field(default=None, description="Last day of the trip.")
    travelers: int | None = Field(
        default=None, ge=1, description="Number of people travelling."
    )
    budget_amount: float | None = Field(
        default=None, ge=0, description="Total trip budget, for all travelers combined."
    )
    budget_currency: str | None = Field(
        default=None, description="ISO 4217 currency code of the budget, e.g. 'USD'."
    )
    interests: list[str] = Field(
        default_factory=list,
        description="Interests driving the itinerary, e.g. 'museums', 'hiking', 'food'.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Hard limits, e.g. 'vegetarian only', 'no flights before 09:00'.",
    )


class IntakeResult(BaseModel):
    """Output of the Intake Agent."""

    profile: TravelerProfile
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Required fields still unknown after reading the user's request.",
    )
    clarifying_question: str | None = Field(
        default=None,
        description="A single question to ask the user, set only when fields are missing.",
    )

    @property
    def is_complete(self) -> bool:
        """True when the profile has everything later agents need."""
        return not self.missing_fields


# --------------------------------------------------------------------------
# 3. Destination Research Agent
# --------------------------------------------------------------------------


class Source(BaseModel):
    """A citation backing a research claim."""

    title: str
    url: str


class DestinationResearch(BaseModel):
    """Output of the Destination Research Agent."""

    destination: str = Field(description="The destination this research covers.")
    summary: str = Field(
        description="Concise overview of the destination for these dates."
    )
    weather: str = Field(description="Expected weather during the travel dates.")
    safety: str = Field(description="Safety notes and areas to avoid.")
    currency: str = Field(description="Local currency and rough exchange context.")
    transportation: str = Field(description="How to get around locally.")
    entry_requirements: str = Field(description="Visa, passport and entry rules.")
    sources: list[Source] = Field(
        default_factory=list, description="Sources backing the claims above."
    )


# --------------------------------------------------------------------------
# 4. Flights Agent
# --------------------------------------------------------------------------


class FlightLeg(BaseModel):
    """One leg (single takeoff-to-landing hop) of a flight option."""

    airline: str = Field(description="Operating airline, e.g. 'TAP Air Portugal'.")
    flight_number: str = Field(description="Flight number, e.g. 'TP 1234'.")
    departure_airport: str = Field(description="IATA code of the departure airport.")
    arrival_airport: str = Field(description="IATA code of the arrival airport.")
    departure_time: str = Field(description="Local departure time, 'YYYY-MM-DD HH:MM'.")
    arrival_time: str = Field(description="Local arrival time, 'YYYY-MM-DD HH:MM'.")
    duration_minutes: int = Field(ge=0, description="Duration of this leg in minutes.")


class FlightOption(BaseModel):
    """One bookable flight itinerary the traveler could choose."""

    option_id: str = Field(description="Short stable id, e.g. 'flight-1'.")
    legs: list[FlightLeg] = Field(description="Legs in travel order.")
    stops: int = Field(ge=0, description="Number of stops (legs - 1) per direction.")
    total_duration_minutes: int = Field(
        ge=0, description="Total travel time across all legs."
    )
    price: float = Field(ge=0, description="Total price for all travelers.")
    currency: str = Field(description="ISO 4217 currency code of `price`.")
    baggage: str = Field(
        default="not stated",
        description="Baggage allowance as reported by the source, or 'not stated'.",
    )
    pros: list[str] = Field(default_factory=list, description="Why this option is good.")
    cons: list[str] = Field(default_factory=list, description="Drawbacks to weigh.")


class FlightsResult(BaseModel):
    """Output of the Flights Agent: ranked options plus one recommendation."""

    options: list[FlightOption] = Field(
        default_factory=list, description="Flight options, best first."
    )
    recommended_option_id: str | None = Field(
        default=None, description="The `option_id` of the recommended flight."
    )
    reasoning: str = Field(
        description="Why the recommended option beats the alternatives."
    )


# --------------------------------------------------------------------------
# 5. Lodging Agent
# --------------------------------------------------------------------------


class Coordinates(BaseModel):
    """A point on the map. Shared by lodging, attractions and routing."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class LodgingOption(BaseModel):
    """One place to stay for the whole trip."""

    option_id: str = Field(description="Short stable id, e.g. 'stay-1'.")
    name: str = Field(description="Hotel or apartment name.")
    kind: str = Field(
        default="hotel", description="What it is, e.g. 'hotel', 'apartment', 'hostel'."
    )
    address: str = Field(default="", description="Street address, when known.")
    coordinates: Coordinates | None = Field(
        default=None, description="Location, used by the Routing Agent."
    )
    price_per_night: float | None = Field(
        default=None, ge=0, description="Nightly rate in `currency`."
    )
    total_price: float = Field(
        ge=0, description="Total price for the whole stay, all travelers."
    )
    currency: str = Field(description="ISO 4217 currency code of the prices.")
    rating: float | None = Field(
        default=None, ge=0, le=5, description="Guest rating out of 5."
    )
    stars: int | None = Field(default=None, ge=0, le=5, description="Hotel class.")
    booking_url: str = Field(
        default="",
        description="Link to book or view this stay, exactly as the search returned it.",
    )
    pros: list[str] = Field(default_factory=list, description="Why this option is good.")
    cons: list[str] = Field(default_factory=list, description="Drawbacks to weigh.")


class LodgingResult(BaseModel):
    """Output of the Lodging Agent: ranked stays plus one recommendation."""

    options: list[LodgingOption] = Field(
        default_factory=list, description="Lodging options, best first."
    )
    recommended_option_id: str | None = Field(
        default=None, description="The `option_id` of the recommended stay."
    )
    reasoning: str = Field(
        description="Why the recommended stay fits the budget and preferences."
    )


# --------------------------------------------------------------------------
# 6. Attractions Agent
# --------------------------------------------------------------------------


class Place(BaseModel):
    """A candidate place to visit: attraction, restaurant or activity."""

    place_id: str = Field(description="Short stable id, e.g. 'place-1'.")
    name: str = Field(description="Name of the place.")
    category: str = Field(
        description="What it is, e.g. 'museum', 'restaurant', 'viewpoint'."
    )
    address: str = Field(default="", description="Street address, when known.")
    coordinates: Coordinates | None = Field(
        default=None, description="Location, required for routing."
    )
    opening_hours: str = Field(
        default="unknown",
        description="Opening hours as reported, e.g. 'Tue-Sun 10:00-18:00'.",
    )
    closed_days: list[str] = Field(
        default_factory=list,
        description="Weekday names the place is closed, e.g. ['Monday'].",
    )
    estimated_cost: float = Field(
        default=0.0, ge=0, description="Entry cost per person, 0 when free."
    )
    currency: str = Field(default="USD", description="Currency of `estimated_cost`.")
    visit_duration_minutes: int = Field(
        default=90, ge=0, description="How long a visit typically takes."
    )
    rating: float | None = Field(default=None, ge=0, le=5, description="Rating out of 5.")
    why_recommended: str = Field(
        default="", description="How it matches the traveler's interests."
    )
    source_url: str = Field(default="", description="Where this information came from.")
    website: str = Field(default="", description="The place's own website, when it has one.")
    maps_url: str = Field(
        default="",
        description="Google Maps link for this place, built from its coordinates.",
    )


class AttractionsResult(BaseModel):
    """Output of the Attractions Agent: the pool the Routing Agent draws from."""

    places: list[Place] = Field(
        default_factory=list, description="Candidate places, best first."
    )
    reasoning: str = Field(description="How these places match the stated interests.")


# --------------------------------------------------------------------------
# 7. Routing Agent
# --------------------------------------------------------------------------


class ScheduledStop(BaseModel):
    """One place slotted into a specific day at a specific time."""

    place_id: str = Field(description="The `place_id` from the attractions pool.")
    name: str = Field(description="Name of the place, copied for readability.")
    start_time: time = Field(description="Local start time, e.g. '10:00'.")
    end_time: time = Field(description="Local end time, e.g. '11:30'.")
    travel_minutes_from_previous: int = Field(
        default=0, ge=0, description="Travel time from the previous stop of that day."
    )
    travel_mode: str = Field(
        default="walking", description="How to get there: 'walking', 'transit', 'taxi'."
    )
    notes: str = Field(default="", description="Anything the traveler should know.")


class DayPlan(BaseModel):
    """One day of the trip."""

    day_number: int = Field(ge=1, description="1 for the first day of the trip.")
    date: date_type = Field(description="Calendar date of this day.")
    stops: list[ScheduledStop] = Field(
        default_factory=list, description="Stops in visiting order."
    )
    summary: str = Field(default="", description="One line describing the day's theme.")


class RoutingResult(BaseModel):
    """Output of the Routing Agent: places distributed into workable days."""

    days: list[DayPlan] = Field(default_factory=list, description="Days in order.")
    unscheduled_place_ids: list[str] = Field(
        default_factory=list,
        description="Places that did not fit, so nothing is silently dropped.",
    )
    reasoning: str = Field(description="How the days were clustered and ordered.")


# --------------------------------------------------------------------------
# 8. Budget Agent
# --------------------------------------------------------------------------


class BudgetLine(BaseModel):
    """One line of the trip's cost breakdown."""

    category: str = Field(
        description="Cost category: 'flights', 'lodging', 'activities', 'food', 'local_transport'."
    )
    amount: float = Field(ge=0, description="Cost in the budget currency.")
    detail: str = Field(default="", description="How this number was arrived at.")


class BudgetResult(BaseModel):
    """Output of the Budget Agent: the total cost against the stated budget."""

    lines: list[BudgetLine] = Field(
        default_factory=list, description="Cost breakdown by category."
    )
    total_cost: float = Field(ge=0, description="Sum of all lines.")
    budget_amount: float | None = Field(
        default=None, ge=0, description="The traveler's stated budget."
    )
    currency: str = Field(description="ISO 4217 currency code of every amount above.")
    within_budget: bool = Field(description="True when `total_cost` <= `budget_amount`.")
    overage: float = Field(
        default=0.0, ge=0, description="How much the trip exceeds the budget by, else 0."
    )
    savings_suggestions: list[str] = Field(
        default_factory=list,
        description="Concrete ways to cut cost, set only when over budget.",
    )
    reasoning: str = Field(description="Summary of the budget position.")


# --------------------------------------------------------------------------
# 9. Critic Agent
# --------------------------------------------------------------------------


class Severity(StrEnum):
    """How badly an issue harms the plan."""

    BLOCKER = "blocker"
    WARNING = "warning"
    INFO = "info"


class Issue(BaseModel):
    """One problem the Critic found in the draft plan."""

    severity: Severity = Field(description="How serious the problem is.")
    category: str = Field(
        description="What it concerns: 'facts', 'schedule', 'budget', 'constraints'."
    )
    description: str = Field(description="What exactly is wrong.")
    location: str = Field(
        default="", description="Where it occurs, e.g. 'day 2, Jeronimos Monastery'."
    )
    suggested_fix: str = Field(default="", description="How to correct it.")


class CriticResult(BaseModel):
    """Output of the Critic Agent: the gate before the final itinerary."""

    issues: list[Issue] = Field(
        default_factory=list, description="Every problem found, worst first."
    )
    approved: bool = Field(
        description="True only when no blocker-severity issue remains."
    )
    reasoning: str = Field(description="Overall verdict on the plan's soundness.")

    @property
    def blockers(self) -> list[Issue]:
        """The issues that must be fixed before the plan can be finalized."""
        return [issue for issue in self.issues if issue.severity == Severity.BLOCKER]


# --------------------------------------------------------------------------
# 10. Itinerary Agent
# --------------------------------------------------------------------------


class ItineraryResult(BaseModel):
    """Output of the Itinerary Agent: the traveler-facing plan."""

    title: str = Field(description="Title of the trip, e.g. '5 Days in Lisbon'.")
    overview: str = Field(description="Short paragraph introducing the trip.")
    days: list[DayPlan] = Field(
        default_factory=list, description="The final day-by-day plan."
    )
    flight_summary: str = Field(default="", description="The chosen flights in words.")
    lodging_summary: str = Field(default="", description="The chosen stay in words.")
    budget_summary: str = Field(default="", description="The cost position in words.")
    practical_notes: list[str] = Field(
        default_factory=list, description="Weather, visas, currency and safety notes."
    )
    markdown: str = Field(
        default="", description="The whole plan rendered as Markdown for the traveler."
    )


# --------------------------------------------------------------------------
# 11. Calendar Agent
# --------------------------------------------------------------------------


class CalendarEvent(BaseModel):
    """One event to place on the traveler's calendar."""

    event_id: str = Field(description="Short stable id, e.g. 'event-1'.")
    title: str = Field(description="Event title as it appears in the calendar.")
    start: str = Field(description="Local start, ISO 8601 'YYYY-MM-DDTHH:MM:SS'.")
    end: str = Field(description="Local end, ISO 8601 'YYYY-MM-DDTHH:MM:SS'.")
    location: str = Field(default="", description="Where the event takes place.")
    description: str = Field(default="", description="Details shown in the event body.")
    category: str = Field(
        default="activity",
        description="What it is: 'flight', 'lodging' or 'activity'.",
    )


class CalendarResult(BaseModel):
    """Output of the Calendar Agent: the exported calendar."""

    events: list[CalendarEvent] = Field(
        default_factory=list, description="Every event created, in time order."
    )
    ics_path: str | None = Field(
        default=None, description="Path of the written .ics file, when one was created."
    )
    reasoning: str = Field(description="What was exported and anything skipped.")


# --------------------------------------------------------------------------
# Traveler memory (persists between trips)
# --------------------------------------------------------------------------


class TravelerPreferences(BaseModel):
    """Stable facts about a traveler, reused across trips.

    These are defaults only. Anything the current request states explicitly
    always wins — the profile is merged with the request on top.
    """

    home_airport: str | None = Field(
        default=None, description="IATA code the traveler usually departs from."
    )
    home_city: str | None = Field(default=None, description="Where the traveler lives.")
    nationality: str | None = Field(
        default=None, description="Used for visa and entry requirements."
    )
    default_currency: str | None = Field(
        default=None, description="ISO 4217 code the traveler thinks in."
    )
    interests: list[str] = Field(
        default_factory=list, description="Recurring interests across trips."
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Standing limits, e.g. 'vegetarian', 'no early flights'.",
    )
    lodging_style: str | None = Field(
        default=None, description="e.g. 'apartment', 'boutique hotel', 'hostel'."
    )
    pace: str | None = Field(
        default=None,
        description="How full a day should be: 'relaxed', 'moderate' or 'packed'.",
    )
    visited_destinations: list[str] = Field(
        default_factory=list, description="Places already planned for this traveler."
    )
