"""Travel Manager Agent (plan.md agent #1).

The orchestrator. It runs before every stage, inspects the state, and decides
which agents run next — or that the work is finished.

Three things make this cheap and correct:

* **Routing is Python, not a prompt.** Each stage consumes the previous one's
  output, so the order is a rule of the system. `next_required_agents()` decides
  it, and the LLM narration is optional (`EXPLAIN_ROUTING=true`) because it used
  to cost ~30 model round trips per plan to produce one sentence per stage.
* **It dispatches in parallel.** After intake, Research, Flights, Lodging and
  Attractions depend on nothing but the profile, so all four are returned at
  once and LangGraph runs them concurrently.
* **It re-checks revised plans.** `plan_version` is bumped whenever Routing
  rebuilds the days; Budget and Critic record the version they examined. Any
  stage left behind must run again, so a revised plan is always re-costed and
  re-reviewed rather than shipped on the strength of a stale verdict.
"""

from __future__ import annotations

import os

from trip_planner.agents.factory import build_structured_agent
from trip_planner.schemas import AgentName, ManagerDecision
from trip_planner.state import TripState
from trip_planner.tools import delegate_to_agent, make_read_trip_state

# How many times the Critic may send the plan back before the planner accepts
# it as good as it will get.
MAX_REVISIONS = 2

# The agents that depend only on the traveler profile, so can all run at once.
PARALLEL_AFTER_INTAKE: tuple[AgentName, ...] = (
    AgentName.DESTINATION_RESEARCH,
    AgentName.FLIGHTS,
    AgentName.LODGING,
    AgentName.ATTRACTIONS,
)

# What must be finished before each remaining stage may run.
_SEQUENTIAL_TAIL: tuple[tuple[AgentName, tuple[AgentName, ...]], ...] = (
    (AgentName.ROUTING, (AgentName.ATTRACTIONS,)),
    (AgentName.BUDGET, (AgentName.ROUTING,)),
    (AgentName.CRITIC, (AgentName.BUDGET,)),
    (AgentName.ITINERARY, (AgentName.CRITIC,)),
    (AgentName.CALENDAR, (AgentName.ITINERARY,)),
)

SYSTEM_PROMPT = """You are the Travel Manager Agent, the orchestrator of an
autonomous trip planner. You never do the planning work yourself.

The routing decision has already been made by the system's rules. Your job is
to explain it to the traveler in one or two plain sentences: what is about to
happen and why it comes next.

Call `read_trip_state` to see what has been done, then answer.
"""


def explain_with_llm() -> bool:
    """Report whether the manager should spend a model call on narration.

    Returns:
        True only when `EXPLAIN_ROUTING` is switched on.
    """
    return os.getenv("EXPLAIN_ROUTING", "").strip().lower() in ("1", "true", "yes")


def next_required_agents(state: TripState) -> list[AgentName]:
    """Decide which stages may run next, by rule.

    Args:
        state: The shared trip state.

    Returns:
        The agents to dispatch — several when they can run concurrently, or an
        empty list when the planner is finished or cannot continue.
    """
    completed = set(state.get("completed_agents", []))

    if AgentName.INTAKE not in completed:
        return [AgentName.INTAKE]

    # Nothing downstream can proceed without a profile. If intake failed, the
    # slot is empty — dispatching anyway is what used to cascade one failure
    # into ten, every downstream agent raising KeyError: 'intake'.
    intake = state.get("intake")
    if intake is None or not intake.is_complete:
        return []

    # Fan out: every one of these needs only the profile.
    ready = [agent for agent in PARALLEL_AFTER_INTAKE if agent not in completed]
    if ready:
        return ready

    plan_version = state.get("plan_version", 0)

    for agent, prerequisites in _SEQUENTIAL_TAIL:
        if not set(prerequisites) <= completed:
            return []

        if agent not in completed:
            return [agent]

        # Re-run a stage that judged an older version of the plan.
        if agent is AgentName.BUDGET and state.get("budget_version", 0) < plan_version:
            return [agent]
        if agent is AgentName.CRITIC and state.get("critic_version", 0) < plan_version:
            return [agent]

        # A rejected plan goes back to Routing, up to the revision limit.
        if agent is AgentName.CRITIC:
            critic = state.get("critic")
            if (
                critic is not None
                and not critic.approved
                and state.get("revision_count", 0) < MAX_REVISIONS
            ):
                return [AgentName.ROUTING]

    return []


def next_required_agent(state: TripState) -> AgentName | None:
    """Return the single next stage, or None.

    Kept for callers and tests that reason about one stage at a time.

    Args:
        state: The shared trip state.

    Returns:
        The first agent that should run, or None when there is nothing to do.
    """
    agents = next_required_agents(state)
    return agents[0] if agents else None


def _template_reasoning(state: TripState, agents: list[AgentName]) -> str:
    """Explain the routing decision without calling a model.

    Args:
        state: The shared trip state.
        agents: The stages about to run.

    Returns:
        A sentence describing what happens next and why.
    """
    if not agents:
        intake = state.get("intake")
        if intake is not None and not intake.is_complete:
            missing = ", ".join(intake.missing_fields)
            return f"Stopping: the traveler still has to supply {missing}."
        critic = state.get("critic")
        if critic is not None and not critic.approved:
            return (
                "Stopping after the revision limit; the plan ships with the "
                "Critic's remaining issues reported rather than hidden."
            )
        return "Every stage is complete, so the trip plan is finished."

    names = ", ".join(str(agent) for agent in agents)
    if len(agents) > 1:
        return (
            f"The traveler profile is ready, so {names} all run at once — none "
            f"of them depends on the others."
        )
    if agents[0] is AgentName.ROUTING and state.get("revision_count", 0) >= 0:
        critic = state.get("critic")
        if critic is not None and not critic.approved:
            return (
                f"The Critic found {len(critic.blockers)} blocker(s), so routing "
                f"rebuilds the days; budget and review then run again on the new plan."
            )
    return f"{names} runs next."


def build_manager_agent(state: TripState):
    """Build the Travel Manager Agent runnable.

    Args:
        state: The state this manager turn should be able to read.

    Returns:
        The agent runnable, with `read_trip_state` bound to `state`.
    """
    return build_structured_agent(
        role="manager",
        tools=[make_read_trip_state(state), delegate_to_agent],
        system_prompt=SYSTEM_PROMPT,
        response_format=ManagerDecision,
        name="travel_manager_agent",
    )


def manager_node(state: TripState, collector=None) -> dict:
    """Graph node: decide which agents run next.

    By default the explanation is generated from a template and this node makes
    **no model calls at all**. `EXPLAIN_ROUTING=true` restores the LLM version
    for demos, where watching the manager reason is the point.

    Args:
        state: The shared trip state.
        collector: Optional metrics callback.

    Returns:
        A partial state update holding the routing decision.
    """
    agents = next_required_agents(state)
    reasoning = _template_reasoning(state, agents)

    if explain_with_llm():
        agent = build_manager_agent(state)
        config = {"callbacks": [collector]} if collector else {}
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Trip request: {state.get('user_request', '')}\n"
                            f"About to run: {', '.join(str(a) for a in agents) or 'nothing'}\n"
                            f"Explain this to the traveler."
                        ),
                    }
                ]
            },
            config,
        )
        reasoning = result["structured_response"].reasoning or reasoning

    return {
        "manager_decision": ManagerDecision(
            next_agent=agents[0] if agents else None,
            next_agents=agents,
            reasoning=reasoning,
        )
    }


def route_from_manager(state: TripState) -> list[str] | str:
    """Conditional edge: map the manager's decision to the next node(s).

    Returning a list makes LangGraph dispatch those nodes concurrently, which
    is what parallelises the four profile-only agents.

    Args:
        state: The shared trip state; reads `manager_decision`.

    Returns:
        The names of the next nodes, or "finish".
    """
    decision = state.get("manager_decision")
    if decision is None or not decision.next_agents:
        return "finish"
    return [str(agent) for agent in decision.next_agents]
