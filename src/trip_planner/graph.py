"""The LangGraph workflow.

Shape of the graph at stage 1:

    START -> manager -> intake              -> manager -> ...
                     -> destination_research -> manager -> ...
                     -> END

The manager sits at the center: every specialist returns to it, and it decides
whether another stage should run. Adding an agent from plan.md later means
adding one node, one edge back to the manager, and one entry in the routing
map — the rest of the graph is untouched.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from trip_planner.agents.intake_agent import intake_node
from trip_planner.agents.manager_agent import manager_node, route_from_manager
from trip_planner.agents.research_agent import research_node
from trip_planner.schemas import AgentName
from trip_planner.state import TripState

# Node names, kept in one place so the routing map and edges cannot drift apart.
MANAGER = "manager"
INTAKE = str(AgentName.INTAKE)
RESEARCH = str(AgentName.DESTINATION_RESEARCH)


def build_graph() -> StateGraph:
    """Build the (uncompiled) trip planner graph."""
    workflow = StateGraph(TripState)

    workflow.add_node(MANAGER, manager_node)
    workflow.add_node(INTAKE, intake_node)
    workflow.add_node(RESEARCH, research_node)

    workflow.add_edge(START, MANAGER)

    # The manager fans out to whichever specialist it selected.
    workflow.add_conditional_edges(
        MANAGER,
        route_from_manager,
        {
            INTAKE: INTAKE,
            RESEARCH: RESEARCH,
            "finish": END,
        },
    )

    # Every specialist reports back to the manager for the next decision.
    workflow.add_edge(INTAKE, MANAGER)
    workflow.add_edge(RESEARCH, MANAGER)

    return workflow


def compile_graph():
    """Build and compile the graph into a runnable app."""
    return build_graph().compile()


# Module-level app, so `langgraph dev` and scripts share one instance.
app = compile_graph()


def plan_trip(user_request: str) -> TripState:
    """Run the planner end to end on a user's request.

    Args:
        user_request: The traveler's free-text request.

    Returns:
        The final state, holding every agent's structured output.
    """
    return app.invoke(
        {
            "user_request": user_request,
            "messages": [{"role": "user", "content": user_request}],
            "completed_agents": [],
        }
    )
