"""The LangGraph workflow.

Shape of the graph:

    START -> manager ─┬─> intake ──────────────┐
                      │                        │
                      ├─> destination_research ┤   these four are dispatched
                      ├─> flights              ├─  together and run in
                      ├─> lodging              │   parallel
                      ├─> attractions ─────────┘
                      │
                      ├─> routing -> budget -> critic   (revision loop)
                      ├─> itinerary -> calendar
                      └─> END

The manager stays at the centre, but it is now cheap: routing is decided by
`next_required_agents()` in pure Python, and the LLM narration is opt-in. When
that function returns several agents, the conditional edge returns a list and
LangGraph runs them concurrently — which is what collapses Research, Flights,
Lodging and Attractions from four sequential stages into one.

Every specialist node is wrapped twice: `track` records its cost, and
`resilient_node` retries it and contains a failure so one bad stage cannot
destroy the other ten.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

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
from trip_planner.llm import model_name
from trip_planner.metrics import track
from trip_planner.resilience import resilient_node
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
SPECIALISTS: dict[str, tuple] = {
    INTAKE: (intake_node, AgentName.INTAKE),
    RESEARCH: (research_node, AgentName.DESTINATION_RESEARCH),
    FLIGHTS: (flights_node, AgentName.FLIGHTS),
    LODGING: (lodging_node, AgentName.LODGING),
    ATTRACTIONS: (attractions_node, AgentName.ATTRACTIONS),
    ROUTING: (routing_node, AgentName.ROUTING),
    BUDGET: (budget_node, AgentName.BUDGET),
    CRITIC: (critic_node, AgentName.CRITIC),
    ITINERARY: (itinerary_node, AgentName.ITINERARY),
    CALENDAR: (calendar_node, AgentName.CALENDAR),
}

# The graph revisits routing, budget and critic on a revision, so the step
# limit has to exceed the longest legal path. LangGraph's default of 25 is low.
# Each agent's own loop is bounded separately, by `factory.STEP_LIMITS`.
RECURSION_LIMIT = 80

# Where resumable runs are stored when checkpointing is on.
CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / ".checkpoints" / "trips.sqlite"


@lru_cache(maxsize=1)
def checkpoint_serializer():
    """Return the serializer used for every checkpoint this project writes.

    The state is almost entirely our own Pydantic models, and LangGraph now
    warns on every one it deserializes without being told to expect it:

        Deserializing unregistered type trip_planner.schemas.IntakeResult
        from checkpoint. This will be blocked in a future version.

    The warning is a real deprecation, not noise — resuming a run will stop
    working once the block lands. Declaring the types answers it now, and the
    list is derived from the modules rather than typed out, so a new schema
    cannot be forgotten. Anything outside these two modules still has to be a
    type LangGraph itself considers safe.

    Returns:
        A `JsonPlusSerializer` that expects this project's own types.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from trip_planner import metrics, schemas

    allowed = [
        value
        for module in (schemas, metrics)
        for value in vars(module).values()
        if isinstance(value, type) and value.__module__ == module.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=tuple(allowed))


def _saver(connection):
    """Build a SQLite checkpointer over an open connection.

    Args:
        connection: The SQLite connection to store checkpoints in.

    Returns:
        A `SqliteSaver` using this project's serializer.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver(connection, serde=checkpoint_serializer())


def _wrap(name: str, node, agent: AgentName):
    """Wrap a specialist node with metrics, retry and failure containment.

    Args:
        name: The node's graph name.
        node: The raw node function, taking `(state, collector)`.
        agent: The agent this node runs.

    Returns:
        A node function taking `(state)` alone, as LangGraph expects.
    """
    measured = track(name, model_name(name))(node)
    return resilient_node(agent)(measured)


def _manager(state: TripState) -> dict:
    """Run the manager, measuring it like any other node.

    Args:
        state: The shared trip state.

    Returns:
        The manager's partial state update.
    """
    return track(MANAGER, model_name("manager"))(manager_node)(state)


def build_graph() -> StateGraph:
    """Build the (uncompiled) trip planner graph."""
    workflow = StateGraph(TripState)

    workflow.add_node(MANAGER, _manager)
    for name, (node, agent) in SPECIALISTS.items():
        workflow.add_node(name, _wrap(name, node, agent))

    workflow.add_edge(START, MANAGER)

    # The manager fans out to one specialist, or to several at once when they
    # are independent. Returning a list from the path function is what makes
    # LangGraph execute those branches concurrently.
    workflow.add_conditional_edges(
        MANAGER,
        route_from_manager,
        {**{name: name for name in SPECIALISTS}, "finish": END},
    )

    # Every specialist reports back to the manager for the next decision. When
    # a parallel group runs, the manager resumes once the whole group is done.
    for name in SPECIALISTS:
        workflow.add_edge(name, MANAGER)

    return workflow


def checkpointing_enabled() -> bool:
    """Report whether runs are persisted and can be resumed.

    Returns:
        True unless `TRIP_CHECKPOINTS` is explicitly switched off.
    """
    return os.getenv("TRIP_CHECKPOINTS", "true").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def compile_graph(checkpointer=None):
    """Build and compile the graph into a runnable app.

    Args:
        checkpointer: A LangGraph checkpointer, or None for an in-memory run.

    Returns:
        The compiled app.
    """
    return build_graph().compile(checkpointer=checkpointer)


# Module-level app, so `langgraph dev` and scripts share one instance. It is
# compiled without a checkpointer; `plan_trip` supplies one per run, because a
# SQLite connection cannot be shared safely across threads.
app = compile_graph()


def initial_state(user_request: str) -> dict:
    """Build the starting state for a plan.

    Args:
        user_request: The traveler's free-text request.

    Returns:
        The initial `TripState` values.
    """
    return {
        "user_request": user_request,
        "completed_agents": [],
        "failed_agents": [],
        "metrics": [],
        "plan_version": 0,
        "budget_version": 0,
        "critic_version": 0,
        "revision_count": 0,
    }


@lru_cache(maxsize=1)
def checkpointed_app():
    """Return a graph compiled against a long-lived SQLite checkpointer.

    The connection is deliberately kept open for the life of the process, and
    `check_same_thread=False` because Gradio serves each request on a different
    worker thread. Both are needed for human-in-the-loop: `interrupt()` has to
    suspend into a checkpoint that a *later* HTTP request can resume from.

    Returns:
        The compiled app, or the plain in-memory app when checkpointing is off.
    """
    if not checkpointing_enabled():
        return app

    import sqlite3

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
    return compile_graph(_saver(connection))


def run_config(thread_id: str) -> dict:
    """Build the run config for one plan.

    Args:
        thread_id: Identifies this plan, so it can be resumed.

    Returns:
        A LangGraph config.
    """
    return {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": thread_id},
    }


def plan_trip(user_request: str, thread_id: str | None = None) -> TripState:
    """Run the planner end to end on a user's request.

    Args:
        user_request: The traveler's free-text request.
        thread_id: Resume an earlier run instead of starting a new one. Needs
            checkpointing to be enabled.

    Returns:
        The final state, holding every agent's structured output.
    """
    config: dict = {"recursion_limit": RECURSION_LIMIT}

    if thread_id and checkpointing_enabled():
        import sqlite3

        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        config["configurable"] = {"thread_id": thread_id}
        connection = sqlite3.connect(str(CHECKPOINT_PATH))
        try:
            return compile_graph(_saver(connection)).invoke(
                initial_state(user_request), config
            )
        finally:
            connection.close()

    return app.invoke(initial_state(user_request), config)
