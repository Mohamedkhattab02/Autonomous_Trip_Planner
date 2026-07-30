"""The LangGraph state shared by every agent node.

Each agent node reads the fields it needs and returns a partial update.
`completed_agents` is the only field agents append to concurrently, so it
carries a reducer; everything else is last-write-wins.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from trip_planner.schemas import (
    AgentName,
    DestinationResearch,
    IntakeResult,
    ManagerDecision,
)


class TripState(TypedDict, total=False):
    """Shared state for the trip planning graph."""

    # Conversation with the user. `add_messages` appends rather than replaces.
    messages: Annotated[list[AnyMessage], add_messages]

    # The original request, kept verbatim so agents never depend on message order.
    user_request: str

    # Agent outputs, keyed by the agent that produced them.
    manager_decision: ManagerDecision
    intake: IntakeResult
    research: DestinationResearch

    # Which agents have finished, in order. Drives the manager's routing.
    completed_agents: Annotated[list[AgentName], operator.add]
