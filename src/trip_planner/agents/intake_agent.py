"""Intake Agent (plan.md agent #2).

Reads the user's free-text request and turns it into a validated
`TravelerProfile`, flagging anything still missing.
"""

from __future__ import annotations

from langchain.agents import create_agent

from trip_planner.llm import get_model
from trip_planner.schemas import AgentName, IntakeResult
from trip_planner.state import TripState
from trip_planner.tools import INTAKE_TOOLS

SYSTEM_PROMPT = """You are the Intake Agent of an autonomous trip planner.

Your job is to turn the traveler's free-text request into a structured profile.

Follow these steps:
1. Read the request and extract every trip detail it states: destination,
   origin, dates, number of travelers, budget, interests and constraints.
2. Call `validate_required_fields` with what you extracted to learn which
   required fields are still missing.
3. If any field is missing, call `ask_clarifying_question` to produce a single
   question covering all of them.
4. Return the structured result.

Rules:
- Never invent a detail the traveler did not state. Leave unknown fields null.
- Convert relative dates ("next August") into absolute YYYY-MM-DD dates only
  when the request makes the exact days unambiguous; otherwise leave them null
  and treat them as missing.
- `missing_fields` must be exactly what `validate_required_fields` reported.
- Set `clarifying_question` only when fields are missing; otherwise leave it null.
"""


def build_intake_agent():
    """Build the Intake Agent runnable."""
    return create_agent(
        model=get_model(),
        tools=INTAKE_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=IntakeResult,
        name="intake_agent",
    )


def intake_node(state: TripState) -> dict:
    """Graph node: run the Intake Agent and store its profile in the state.

    Args:
        state: The shared trip state; reads `user_request`.

    Returns:
        A partial state update with the intake result.
    """
    agent = build_intake_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": state["user_request"]}]}
    )
    intake: IntakeResult = result["structured_response"]

    return {
        "intake": intake,
        "completed_agents": [AgentName.INTAKE],
    }
