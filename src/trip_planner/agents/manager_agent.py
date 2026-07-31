"""Travel Manager Agent (plan.md agent #1).

The orchestrator. It runs before every stage, inspects the state, and decides
which agent runs next - or that the work is finished.

Which stage may run next is a rule of the system, not a matter of taste: each
stage consumes the previous one's output, so `next_required_agent` decides the
order in Python. The agent still runs, and its `reasoning` is what the traveler
sees, but a confused model cannot send the graph somewhere impossible.
"""

from __future__ import annotations

from langchain.agents import create_agent

from trip_planner.llm import get_model
from trip_planner.schemas import AgentName, ManagerDecision
from trip_planner.state import TripState
from trip_planner.tools import AGENT_SEQUENCE, delegate_to_agent, make_read_trip_state

# How many times the Critic may send the plan back before the planner accepts
# it as good as it will get. Without this, a plan the Critic never likes would
# loop forever.
MAX_REVISIONS = 2

SYSTEM_PROMPT = """You are the Travel Manager Agent, the orchestrator of an
autonomous trip planner. You never do the planning work yourself; you decide
which specialist agent runs next.

Always start by calling `read_trip_state` to see what has already been done.

The agents run in this order, because each one consumes the previous one's
output:
1. `intake` - turns the request into a structured traveler profile.
2. `destination_research` - weather, safety, currency, transport, entry rules.
3. `flights` - finds and ranks real flights.
4. `lodging` - finds and ranks real places to stay.
5. `attractions` - gathers real places matching the traveler's interests.
6. `routing` - splits those places into workable days.
7. `budget` - totals the cost and checks it against the budget.
8. `critic` - verifies the plan is real, possible and within limits.
9. `itinerary` - writes the final plan for the traveler.
10. `calendar` - exports the approved plan as calendar events.

Rules:
- Never run an agent that already appears in `completed_agents`, with one
  exception: when the Critic rejected the plan, `routing` runs again to fix it.
- If intake completed but reported missing fields, the planner cannot continue:
  set `next_agent` to null and explain what the traveler must supply.
- When the Critic reports blockers, the plan goes back to `routing`.
- When every stage has completed, set `next_agent` to null.
- Call `delegate_to_agent` to record your choice, and make your structured
  `next_agent` match what you delegated.
- Explain your choice in `reasoning` in one or two sentences the traveler
  would understand.
"""


def next_required_agent(state: TripState) -> AgentName | None:
    """Decide which stage may run next, by rule.

    Args:
        state: The shared trip state.

    Returns:
        The agent that should run next, or None when the planner is done or
        cannot continue.
    """
    completed = set(state.get("completed_agents", []))

    # Nothing can proceed on an incomplete profile: every later stage needs a
    # destination and dates.
    intake = state.get("intake")
    if intake is not None and not intake.is_complete:
        return None

    # A rejected plan goes back to routing, up to the revision limit.
    critic = state.get("critic")
    if (
        critic is not None
        and not critic.approved
        and state.get("revision_count", 0) < MAX_REVISIONS
    ):
        return AgentName.ROUTING

    for agent in AGENT_SEQUENCE:
        if agent not in completed:
            return agent
    return None


def build_manager_agent(state: TripState):
    """Build the Travel Manager Agent runnable.

    Args:
        state: The state this manager turn should be able to read.

    Returns:
        The agent runnable, with `read_trip_state` bound to `state`.
    """
    return create_agent(
        model=get_model(),
        tools=[make_read_trip_state(state), delegate_to_agent],
        system_prompt=SYSTEM_PROMPT,
        response_format=ManagerDecision,
        name="travel_manager_agent",
    )


def _decision_brief(state: TripState, expected: AgentName | None) -> str:
    """Build the manager's prompt for this turn.

    Args:
        state: The shared trip state.
        expected: The stage the rules say comes next.

    Returns:
        A prompt asking the manager to explain the next step.
    """
    completed = ", ".join(str(name) for name in state.get("completed_agents", [])) or "none"
    if expected is None:
        instruction = (
            "Every required stage is done, or the planner cannot continue. "
            "Set `next_agent` to null and explain why the work stops here."
        )
    else:
        instruction = (
            f"The next stage to run is `{expected}`. Delegate to it and explain "
            f"in one or two sentences why it comes next."
        )

    return (
        f"Trip request: {state.get('user_request', '')}\n"
        f"Agents completed so far: {completed}\n"
        f"Revisions so far: {state.get('revision_count', 0)}\n\n"
        f"{instruction}"
    )


def manager_node(state: TripState) -> dict:
    """Graph node: decide which agent runs next.

    The next stage is fixed by `next_required_agent`; the agent supplies the
    explanation. If the model's choice disagrees with the rules, the rules win.

    Args:
        state: The shared trip state.

    Returns:
        A partial state update holding the routing decision.
    """
    expected = next_required_agent(state)
    agent = build_manager_agent(state)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": _decision_brief(state, expected)}]}
    )
    decision: ManagerDecision = result["structured_response"]

    # The sequence is a system rule, so the model's reasoning is kept but its
    # routing choice is overridden.
    return {
        "manager_decision": ManagerDecision(
            next_agent=expected, reasoning=decision.reasoning
        )
    }


def route_from_manager(state: TripState) -> str:
    """Conditional edge: map the manager's decision to the next node.

    Args:
        state: The shared trip state; reads `manager_decision`.

    Returns:
        The name of the next node, or "finish".
    """
    decision = state.get("manager_decision")
    if decision is None or decision.next_agent is None:
        return "finish"

    return str(decision.next_agent)
