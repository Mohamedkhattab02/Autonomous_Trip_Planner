"""Intake Agent (plan.md agent #2).

Reads the user's free-text request and turns it into a validated
`TravelerProfile`, flagging anything still missing.

**Why this agent has no tools.** It used to be given `validate_required_fields`
and `ask_clarifying_question` and told to call them. Both are pure functions of
data the node already holds, so calling them told the model nothing it did not
already know — but it did give the agent loop something to spin on. Intake runs
on the cheapest model in the tier (`LLM_MODEL_INTAKE`), and small models handed
a deterministic checker call it, read the identical answer as "still not done",
and call it again. The loop only ended at the graph's step limit, which is sized
for the whole eleven-agent pipeline: from the outside the planner simply hung at
its first stage.

So the work is split by what each side is actually good at:

* **The model extracts.** One structured-output call turns free text into a
  `TravelerProfile`. No tools, so there is nothing to loop on.
* **Python decides.** `_finalize` calls the same two tools directly to compute
  `missing_fields` and the clarifying question. The rule about which fields are
  mandatory stays exactly where `intake_tools` says it belongs — in the system,
  not in the model's judgement.

That is also what makes this node's promise to the manager unconditional: once
every required field has a value, intake is complete and the next agents are
dispatched, whatever the model claimed about its own output.
"""

from __future__ import annotations

import logging
import os

from trip_planner.agents.factory import build_structured_agent
from trip_planner import memory
from trip_planner.schemas import AgentName, IntakeResult, TravelerProfile
from trip_planner.tools.intake_tools import (
    REQUIRED_FIELDS,
    ask_clarifying_question,
    validate_required_fields,
)
from trip_planner.state import TripState

logger = logging.getLogger(__name__)

# Ceiling on the agent's own loop. Extraction needs one step, so this only ever
# bounds a misbehaving model. Without it the sub-agent inherits the graph's
# `recursion_limit` (80) and can burn ~40 model calls before anyone notices.
STEP_LIMIT = 8

SYSTEM_PROMPT = """You are the Intake Agent of an autonomous trip planner.

Your only job is to read the traveler's request and return it as a structured
profile. You do this in a single pass: extract, then return the result.

Extract every trip detail the request states: destination, origin, start and
end dates, number of travelers, total budget and its currency, interests and
constraints.

Rules:
- Never invent a detail the traveler did not state. Leave unknown fields null.
- Convert relative dates ("next August") into absolute YYYY-MM-DD dates only
  when the request makes the exact days unambiguous; otherwise leave them null.
- Leave `missing_fields` and `clarifying_question` empty. The system computes
  both from the profile you return, by rule — that is not your decision.
"""


def build_intake_agent():
    """Build the Intake Agent runnable.

    Deliberately tool-free: see this module's docstring.
    """
    return build_structured_agent(
        role="intake",
        tools=[],
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


def _ask(question: str, missing: list[str]) -> str:
    """Suspend the graph and return the traveler's answer.

    Suspending needs both a graph context and a checkpointer to suspend into.
    When either is absent — a direct `intake_node(...)` call, or
    `TRIP_ASK_USER=true` with checkpointing off — `interrupt()` raises, and that
    used to abort the run at its very first stage. Not being able to ask is not
    a failure: the node carries on with what it knows, and the manager stops the
    pipeline cleanly on the incomplete profile.

    Args:
        question: What to put to the traveler.
        missing: The fields the question is about.

    Returns:
        The traveler's answer, or "" when the question could not be asked.
    """
    from langgraph.errors import GraphBubbleUp
    from langgraph.types import interrupt

    try:
        return str(interrupt({"question": question, "missing_fields": missing})).strip()
    except GraphBubbleUp:
        # The suspend itself. It has to travel up untouched.
        raise
    except RuntimeError as exc:
        logger.warning("cannot ask the traveler here (%s); continuing", exc)
        return ""


def _extract(request: str, collector=None) -> IntakeResult:
    """Run the Intake Agent over one request and return the profile it read.

    The call gets its own step budget rather than the graph's, and a failure is
    contained: an empty profile leaves every required field missing, which the
    manager reads as "cannot continue" and stops on cleanly. That is a far
    better outcome than the first stage taking the run down with it.

    Args:
        request: The traveler's request, plus any answer they have given.
        collector: Optional metrics callback.

    Returns:
        The agent's `IntakeResult`, or one with an empty profile if extraction
        failed.
    """
    config: dict = {"recursion_limit": STEP_LIMIT}
    if collector is not None:
        # Route this agent's token usage into the run metrics.
        config["callbacks"] = [collector]

    try:
        result = build_intake_agent().invoke(
            {"messages": [{"role": "user", "content": request}]}, config
        )
        return result["structured_response"]
    except Exception as exc:  # noqa: BLE001 - contained deliberately
        from langgraph.errors import GraphBubbleUp

        if isinstance(exc, GraphBubbleUp):
            raise
        logger.error("intake extraction failed: %s", exc)
        return IntakeResult(profile=TravelerProfile())


def _finalize(intake: IntakeResult) -> IntakeResult:
    """Merge stored preferences in, then decide completeness by rule.

    `validate_required_fields` is called here rather than by the model, so the
    answer is computed once from the profile that actually survived the merge.
    Anything extra the model flagged is kept only while that field is genuinely
    empty, and the question is regenerated to match — it can never go on naming
    a field that memory has since filled.

    Args:
        intake: The agent's raw result.

    Returns:
        The result with the merged profile, and `missing_fields` and
        `clarifying_question` consistent with it.
    """
    profile = memory.apply(intake.profile, memory.load())

    # The tool takes strings; the profile holds real dates and numbers.
    checked = validate_required_fields.invoke(
        {
            "destination": profile.destination,
            "start_date": str(profile.start_date) if profile.start_date else None,
            "end_date": str(profile.end_date) if profile.end_date else None,
            "travelers": profile.travelers,
            "budget_amount": profile.budget_amount,
        }
    )
    missing: list[str] = list(checked["missing_fields"])

    # Keep anything else the model flagged, unless the merge supplied it.
    for field in intake.missing_fields:
        if field not in REQUIRED_FIELDS and getattr(profile, field, None) in (None, ""):
            missing.append(field)

    question = ask_clarifying_question.invoke({"missing_fields": missing}) if missing else None

    return intake.model_copy(
        update={
            "profile": profile,
            "missing_fields": missing,
            "clarifying_question": question,
        }
    )


def intake_node(state: TripState, collector=None) -> dict:
    """Graph node: run the Intake Agent and store its profile in the state.

    Two things happen around the extraction itself:

    * **Stored preferences fill the gaps.** Anything the traveler did not state
      this time is taken from their saved profile — but never overrides what
      they did state.
    * **A missing detail can be asked about.** When required fields are still
      absent and checkpointing is on, the node calls `interrupt()`. The graph
      suspends with its state intact, the UI shows the question, and resuming
      with the answer continues *inside this node*: the answer is folded into
      the request and the profile is re-extracted here.

    The node always reports intake as complete, whatever the profile looks like.
    The manager reads `is_complete` to decide whether the pipeline may continue,
    so a still-incomplete profile stops the run cleanly there — rather than
    being routed back here for a round that would ask the same question again.

    Args:
        state: The shared trip state; reads `user_request` and `user_answer`.
        collector: Optional metrics callback.

    Returns:
        A partial state update with the intake result.
    """
    request = state["user_request"]
    answer = state.get("user_answer", "")

    def with_answer(text: str) -> str:
        return f"{request}\n\nAdditional details from the traveler: {text}"

    intake = _finalize(_extract(with_answer(answer) if answer else request, collector))

    if intake.missing_fields and not answer and _can_ask():
        # Suspends the graph; resuming supplies the traveler's reply.
        reply = _ask(
            intake.clarifying_question or "Could you give me a few more details?",
            intake.missing_fields,
        )
        if reply:
            answer = reply
            # Re-extract here rather than handing back to the manager: doing
            # that left intake incomplete, so the manager routed straight back
            # and the stage ran twice more to reach the same place.
            intake = _finalize(_extract(with_answer(reply), collector))

    update: dict = {"intake": intake, "completed_agents": [AgentName.INTAKE]}
    if answer:
        update["user_answer"] = answer
    return update
