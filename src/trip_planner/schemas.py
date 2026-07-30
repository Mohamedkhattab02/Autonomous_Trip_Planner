"""Pydantic structured messages exchanged between agents.

Every agent produces exactly one of these models. They are the contract
between nodes: an agent never reads another agent's raw text, only its
validated structured output.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class AgentName(StrEnum):
    """Names of the agents in the system, used for routing."""

    INTAKE = "intake"
    DESTINATION_RESEARCH = "destination_research"


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
    start_date: date | None = Field(default=None, description="First day of the trip.")
    end_date: date | None = Field(default=None, description="Last day of the trip.")
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
