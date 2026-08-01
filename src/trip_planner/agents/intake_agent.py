"""Intake Agent (plan.md agent #2).

Reads the user's free-text request and turns it into a validated
`TravelerProfile`, flagging anything still missing.
"""

from __future__ import annotations

import os

from trip_planner.agents.factory import build_structured_agent
from trip_planner import memory
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
    return build_structured_agent(
        role="intake",
        tools=INTAKE_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=IntakeResult,
        name="intake_agent",
    )


def _can_ask() -> bool:
    """Report whether the graph may pause to ask the traveler a question.

    Interrupting needs a checkpointer to suspend into, and is pointless in a
    one-shot script, so it is opt-in.

    Returns:
        True when `TRIP_ASK_USER` is switched on.
    """
    return os.getenv("TRIP_ASK_USER", "").strip().lower() in ("1", "true", "yes")


def intake_node(state: TripState, collector=None) -> dict:
    """Graph node: run the Intake Agent and store its profile in the state.

    Two things happen around the agent itself:

    * **Stored preferences fill the gaps.** Anything the traveler did not state
      this time is taken from their saved profile — but never overrides what
      they did state.
    * **A missing detail can be asked about.** When required fields are still
      absent and checkpointing is on, the node calls `interrupt()`. The graph
      suspends with its state intact, the UI shows the question, and resuming
      with the answer continues from here rather than replanning from scratch.

    Args:
        state: The shared trip state; reads `user_request` and `user_answer`.
        collector: Optional metrics callback.

    Returns:
        A partial state update with the intake result.
    """
    request = state["user_request"]
    answer = state.get("user_answer", "")
    if answer:
        request = f"{request}\n\nAdditional details from the traveler: {answer}"

    agent = build_intake_agent()
    if collector is not None:
        # Route this agent's token and tool usage into the run metrics.
        agent = agent.with_config(callbacks=[collector])
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    intake: IntakeResult = result["structured_response"]

    # Fill anything unstated from the traveler's stored preferences.
    preferences = memory.load()
    intake = intake.model_copy(
        update={"profile": memory.apply(intake.profile, preferences)}
    )

    # Re-check completeness: a preference may have supplied a missing field.
    still_missing = [
        field
        for field in intake.missing_fields
        if getattr(intake.profile, field, None) in (None, "")
    ]
    intake = intake.model_copy(update={"missing_fields": still_missing})

    if still_missing and not answer and _can_ask():
        from langgraph.types import interrupt

        # Suspends the graph; resuming supplies the traveler's reply.
        reply = interrupt(
            {
                "question": intake.clarifying_question
                or "Could you give me a few more details?",
                "missing_fields": still_missing,
            }
        )
        return {"user_answer": str(reply)}

    return {
        "intake": intake,
        "completed_agents": [AgentName.INTAKE],
    }
