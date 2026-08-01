"""The LangGraph state shared by every agent node.

Each agent node reads the fields it needs and returns a partial update. Fields
that several nodes append to concurrently carry a reducer; everything else is
last-write-wins.

Agents communicate **only** through the typed slots below — never through a
shared conversation. An earlier version declared a `messages` field with an
`add_messages` reducer, but no node ever wrote to it: each agent runs its own
sub-agent with a private message list. The field was removed rather than left
as a misleading contract.

Concurrency note: `completed_agents`, `metrics` and `failed_agents` are all
appended to by the four agents that run in parallel, so each uses `operator.add`
to merge. Their *order* therefore reflects completion, not the pipeline order.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from trip_planner.metrics import AgentMetrics
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
    ManagerDecision,
    RoutingResult,
)


class TripState(TypedDict, total=False):
    """Shared state for the trip planning graph."""

    # The original request, kept verbatim so agents never depend on ordering.
    user_request: str

    # Anything the traveler supplied later, in answer to a clarifying question.
    user_answer: str

    # Agent outputs, keyed by the agent that produced them.
    manager_decision: ManagerDecision
    intake: IntakeResult
    research: DestinationResearch
    flights: FlightsResult
    lodging: LodgingResult
    attractions: AttractionsResult
    routing: RoutingResult
    budget: BudgetResult
    critic: CriticResult
    itinerary: ItineraryResult
    calendar: CalendarResult

    # Which agents have finished. Appended concurrently, so unordered.
    completed_agents: Annotated[list[AgentName], operator.add]

    # Which agents failed, and why. A failed agent still counts as complete so
    # the graph continues; the plan is then labelled degraded.
    failed_agents: Annotated[list[dict], operator.add]

    # Per-agent efficiency records, appended as each finishes.
    metrics: Annotated[list[AgentMetrics], operator.add]

    # Plan versioning drives the revision loop. `routing` bumps `plan_version`
    # every time it rebuilds the days; `budget` and `critic` record which
    # version they examined. Any stage whose recorded version is behind must
    # run again — which is what makes the Critic re-check a revised plan
    # instead of the plan it already rejected.
    plan_version: int
    budget_version: int
    critic_version: int

    # How many times the Critic sent the plan back. Bounds the revision loop.
    revision_count: int
