"""The LangGraph workflow.

Shape of the graph:

    START -> manager -> intake               -> manager -> ...
                     -> destination_research -> manager -> ...
                     -> flights              -> manager -> ...
                     -> lodging              -> manager -> ...
                     -> attractions          -> manager -> ...
                     -> routing              -> manager -> ...
                     -> budget               -> manager -> ...
                     -> critic               -> manager -> ...
                     -> itinerary            -> manager -> ...
                     -> calendar             -> manager -> ...
                     -> END

The manager sits at the center: every specialist returns to it, and it decides
whether another stage should run. That is also how the revision loop works -
when the Critic rejects the plan, the manager simply routes back to `routing`,
so no special edge is needed.

Adding an agent means adding one node, one edge back to the manager, and one
entry in the routing map - the rest of the graph is untouched.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from trip_planner.agents.attractions_agent import attractions_node
from trip_planner.agents.budget_agent import budget_node
from trip_planner.agents.calendar_agent import calendar_node
from trip_planner.agents.critic_agent import critic_node
from trip_planner.agents.flights_agent import flights_node
from trip_planner.agents.intake_agent import intake_node
from trip_planner.agents.itinerary_agent import itinerary_node
from trip_planner.agents.lodging_agent import lodging_node
from trip_planner.agents.manager_agent import manager_node, route_from_manager
from trip_planner.agents.research_agent import research_node
from trip_planner.agents.routing_agent import routing_node
from trip_planner.schemas import AgentName
from trip_planner.state import TripState

# Node names, kept in one place so the routing map and edges cannot drift apart.
MANAGER = "manager"
INTAKE = str(AgentName.INTAKE)
RESEARCH = str(AgentName.DESTINATION_RESEARCH)
FLIGHTS = str(AgentName.FLIGHTS)
LODGING = str(AgentName.LODGING)
ATTRACTIONS = str(AgentName.ATTRACTIONS)
ROUTING = str(AgentName.ROUTING)
BUDGET = str(AgentName.BUDGET)
CRITIC = str(AgentName.CRITIC)
ITINERARY = str(AgentName.ITINERARY)
CALENDAR = str(AgentName.CALENDAR)

# Every specialist, in the order plan.md lists them.
SPECIALISTS: dict[str, callable] = {
    INTAKE: intake_node,
    RESEARCH: research_node,
    FLIGHTS: flights_node,
    LODGING: lodging_node,
    ATTRACTIONS: attractions_node,
    ROUTING: routing_node,
    BUDGET: budget_node,
    CRITIC: critic_node,
    ITINERARY: itinerary_node,
    CALENDAR: calendar_node,
}

# The graph revisits `routing` on a critic revision, so the step limit has to
# exceed 2 x (stages + manager turns). LangGraph's default of 25 is too low.
RECURSION_LIMIT = 60


def build_graph() -> StateGraph:
    """Build the (uncompiled) trip planner graph."""
    workflow = StateGraph(TripState)

    workflow.add_node(MANAGER, manager_node)
    for name, node in SPECIALISTS.items():
        workflow.add_node(name, node)

    workflow.add_edge(START, MANAGER)

    # The manager fans out to whichever specialist it selected.
    workflow.add_conditional_edges(
        MANAGER,
        route_from_manager,
        {**{name: name for name in SPECIALISTS}, "finish": END},
    )

    # Every specialist reports back to the manager for the next decision.
    for name in SPECIALISTS:
        workflow.add_edge(name, MANAGER)

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
            "revision_count": 0,
        },
        config={"recursion_limit": RECURSION_LIMIT},
    )
