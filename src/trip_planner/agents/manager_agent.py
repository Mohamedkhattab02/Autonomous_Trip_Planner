"""Travel Manager Agent (plan.md agent #1).

The orchestrator. It runs before every stage, inspects the state, and decides
which agent runs next — or that the work is finished.
"""

from __future__ import annotations

from langchain.agents import create_agent

from trip_planner.llm import get_model
from trip_planner.schemas import AgentName, ManagerDecision
from trip_planner.state import TripState
from trip_planner.tools import delegate_to_agent, make_read_trip_state

SYSTEM_PROMPT = """You are the Travel Manager Agent, the orchestrator of an
autonomous trip planner. You never do the planning work yourself; you decide
which specialist agent runs next.

Always start by calling `read_trip_state` to see what has already been done.

The agents available at this stage, in the order they must run:
1. `intake` - turns the traveler's request into a structured profile.
   Run it when it has not run yet.
2. `destination_research` - researches weather, safety, currency, transport
   and entry requirements. Run it once intake is complete and the profile has
   a destination.

Rules:
- Never run an agent that already appears in `completed_agents`.
- Never run `destination_research` before `intake` has completed.
- If intake completed but reported missing fields, the planner cannot continue:
  set `next_agent` to null and explain in `reasoning` what the traveler must supply.
- When both agents have completed, set `next_agent` to null.
- Call `delegate_to_agent` to record your choice before returning it, and make
  your structured `next_agent` match what you delegated.
"""


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


def manager_node(state: TripState) -> dict:
    """Graph node: decide which agent runs next.

    Args:
        state: The shared trip state.

    Returns:
        A partial state update holding the routing decision.
    """
    agent = build_manager_agent(state)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Decide which agent should run next for this trip request:\n"
                        f"{state['user_request']}"
                    ),
                }
            ]
        }
    )
    decision: ManagerDecision = result["structured_response"]

    return {"manager_decision": decision}


def route_from_manager(state: TripState) -> str:
    """Conditional edge: map the manager's decision to the next node.

    A decision to re-run an already completed agent is treated as "finish",
    so a confused manager can never loop the graph forever.

    Args:
        state: The shared trip state; reads `manager_decision`.

    Returns:
        The name of the next node, or "finish".
    """
    decision = state.get("manager_decision")
    if decision is None or decision.next_agent is None:
        return "finish"

    if decision.next_agent in state.get("completed_agents", []):
        return "finish"

    return str(decision.next_agent)
