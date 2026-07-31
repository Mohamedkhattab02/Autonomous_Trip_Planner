# Autonomous Trip Planner

A multi-agent trip planner built on LangGraph, with Gemini 2.5 Flash as the
only model. Agents are graph nodes, the state is shared, and every message
between agents is a validated Pydantic model.

All eleven agents from `plan.md` are implemented, and they work against real
data: live flight prices, real hotel rates, and actual Google Maps places with
coordinates and opening hours.

## The agents

| # | Agent | Tools | Output |
|---|-------|-------|--------|
| 1 | Travel Manager | `read_trip_state`, `delegate_to_agent` | `ManagerDecision` |
| 2 | Intake | `validate_required_fields`, `ask_clarifying_question` | `IntakeResult` |
| 3 | Destination Research | `tavily_search`, `get_weather`, `get_entry_requirements`, `get_currency_info` | `DestinationResearch` |
| 4 | Flights | `search_flights`, `compare_flights`, `get_flight_details` | `FlightsResult` |
| 5 | Lodging | `search_hotels`, `get_hotel_details`, `check_hotel_location`, `compare_hotels` | `LodgingResult` |
| 6 | Attractions | `web_search`, `search_places`, `get_opening_hours`, `get_place_details` | `AttractionsResult` |
| 7 | Routing | `calculate_distance`, `calculate_travel_time`, `cluster_locations`, `check_opening_hours` | `RoutingResult` |
| 8 | Budget | `calculate_total_cost`, `convert_currency`, `estimate_food_cost`, `estimate_local_transport_cost`, `suggest_cheaper_alternatives` | `BudgetResult` |
| 9 | Critic | `verify_place`, `verify_opening_hours`, `validate_schedule`, `validate_budget` | `CriticResult` |
| 10 | Itinerary | `read_agent_results`, `build_daily_itinerary`, `format_trip_plan` | `ItineraryResult` |
| 11 | Calendar | `create_calendar_event`, `update_event`, `export_ics` | `CalendarResult` |

## The graph

```
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
```

The manager sits at the center. Every specialist returns to it, and it decides
whether another stage should run. Adding an agent means adding one node, one
edge back to the manager, and one entry in the routing map.

![The graph](graph.png)

### The revision loop

When the Critic finds a blocker - a place that does not exist, a museum
scheduled on its closing day, overlapping stops, a breached budget - the
manager routes back to `routing` with the issues attached, and the plan is
rebuilt. No extra edge is needed: the manager already reaches every node.

`MAX_REVISIONS` caps this at 2, so a plan the Critic never likes cannot loop
forever. After the cap the trip proceeds, with the remaining issues reported
to the traveler rather than hidden.

## Two design rules

**The stage order is code, not a prompt.** Each stage consumes the previous
one's output, so `next_required_agent()` in `manager_agent.py` decides what
runs next in Python. The manager agent still runs, and its `reasoning` is what
the traveler reads, but a confused model cannot send the graph somewhere
impossible.

**Arithmetic belongs in tools, not in the model.** Ranking flights, measuring
distances, clustering places into days, totalling a budget and detecting
schedule overlaps are all plain Python. A model eyeballing these produces
itineraries that criss-cross a city and budgets that do not add up. The agents
decide *what matters*; the tools decide *what is possible*.

## Layout

```
src/trip_planner/
  schemas.py          Pydantic messages passed between agents
  state.py            The shared LangGraph state
  llm.py              The single Gemini 2.5 Flash configuration
  graph.py            Nodes, edges and the compiled app
  agents/             One module per agent
  tools/              Tools, grouped by the agent that owns them
    serp.py           Shared SerpApi client (flights, hotels, places)
    geo.py            Shared distance and travel-time maths
tests/test_planner.py Checks for everything above
scripts/draw_graph.py Renders graph.png
exports/              Generated .ics calendar files
```

## Setup

Put the API keys in `.env` at the project root:

```
GOOGLE_API_KEY=...     # Gemini
TAVILY_API_KEY=...     # web research
SERPAPI_API_KEY=...    # flights, hotels, places, currency
```

Then install:

```bash
uv sync
```

## Running

```bash
# Plan a trip
uv run main.py

# Plan a specific trip
uv run main.py "5 days in Rome in May 2027, 2 people, 2500 EUR, art and food"

# Draw the graph to graph.png
uv run scripts/draw_graph.py
```

The Calendar Agent writes a `.ics` file to `exports/`, which imports into
Google Calendar, Apple Calendar and Outlook.

## Tests

```bash
# Fast: graph shape, routing rules, every deterministic tool. No network.
uv run pytest tests -m "not live"

# Everything, including real Gemini, Tavily and SerpApi calls.
uv run pytest tests
```

The 51 fast tests need no API keys and cover the graph wiring, the stage
sequencing, the revision cap, and every tool that does arithmetic.

The `live` tests call the real APIs. A free-tier Google API key allows only
**20 Gemini requests per day**, and a full eleven-agent run uses considerably
more than that, so those tests will hit `RESOURCE_EXHAUSTED` quickly unless
billing is enabled on the key.
