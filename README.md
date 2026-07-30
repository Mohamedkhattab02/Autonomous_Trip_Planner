# Autonomous Trip Planner

A multi-agent trip planner built on LangGraph, with Gemini 2.5 Flash as the
only model. Agents are graph nodes, the state is shared, and every message
between agents is a validated Pydantic model.

## Stage 1 (current)

Three of the eleven agents in `plan.md` are implemented:

| # | Agent | Tools | Output |
|---|-------|-------|--------|
| 1 | Travel Manager | `read_trip_state`, `delegate_to_agent` | `ManagerDecision` |
| 2 | Intake | `validate_required_fields`, `ask_clarifying_question` | `IntakeResult` |
| 3 | Destination Research | `tavily_search`, `get_weather`, `get_entry_requirements`, `get_currency_info` | `DestinationResearch` |

### The graph

```
START -> manager -> intake               -> manager -> ...
                 -> destination_research -> manager -> ...
                 -> END
```

The manager sits at the center. Every specialist returns to it, and it decides
whether another stage should run. Adding an agent later means adding one node,
one edge back to the manager, and one entry in the routing map.

## Layout

```
src/trip_planner/
  schemas.py          Pydantic messages passed between agents
  state.py            The shared LangGraph state
  llm.py              The single Gemini 2.5 Flash configuration
  graph.py            Nodes, edges and the compiled app
  agents/             One module per agent
  tools/              Tools, grouped by the agent that owns them
tests/test_stage1.py  Checks for everything above
scripts/draw_graph.py Renders graph.png
```

## Setup

Put the API keys in `.env` at the project root:

```
GOOGLE_API_KEY=...
TAVILY_API_KEY=...
```

Then install:

```bash
uv sync
```

## Running

```bash
# Plan a trip
uv run main.py

# Draw the graph to graph.png
uv run scripts/draw_graph.py
```

## Tests

```bash
# Fast: graph shape, routing rules, deterministic tools. No network.
uv run pytest tests -m "not live"

# Everything, including real Gemini and Tavily calls.
uv run pytest tests
```

The `live` tests call the real APIs. A free-tier Google API key allows only
**20 Gemini requests per day**, and one full graph run uses several, so those
tests will hit `RESOURCE_EXHAUSTED` quickly unless billing is enabled on the key.
