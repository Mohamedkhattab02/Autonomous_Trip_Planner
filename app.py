"""Gradio frontend for the Autonomous Trip Planner.

Streams the LangGraph run so the traveler watches each of the eleven agents
finish in turn, rather than staring at a spinner for two minutes.

Usage:
    uv run app.py
"""

from __future__ import annotations

import gradio as gr

from trip_planner.graph import RECURSION_LIMIT, app as graph_app
from trip_planner.mcp_client import describe_calendar_backend
from trip_planner.render import (
    PIPELINE,
    progress_html,
    render_budget,
    render_calendar,
    render_critic,
    render_flights,
    render_itinerary,
    render_lodging,
    render_places,
    render_profile,
    render_research,
)

EXAMPLES = [
    "I want to travel to Lisbon, Portugal from 2026-09-10 to 2026-09-15. "
    "We are 2 travelers flying from Tel Aviv, our total budget is 3000 USD, "
    "and we love food, history and walking tours.",
    "4 days in Rome from 2027-05-03 to 2027-05-07, 2 people from Tel Aviv, "
    "2500 EUR total, we care about art, ancient history and good pasta.",
    "A week in Tokyo from 2027-04-01 to 2027-04-08 for 1 traveler from London, "
    "4000 GBP, interested in temples, ramen and photography.",
]

CSS = """
:root {
    --tp-ink: #0f1729;
    --tp-muted: #64748b;
    --tp-line: rgba(148, 163, 184, 0.22);
    --tp-accent: #6366f1;
}

.gradio-container { max-width: 1240px !important; }

/* ---------- header ---------- */
#tp-header {
    background: linear-gradient(125deg, #4f46e5 0%, #7c3aed 45%, #db2777 100%);
    border-radius: 20px;
    padding: 30px 34px;
    margin-bottom: 20px;
    color: #fff;
    box-shadow: 0 18px 40px -18px rgba(79, 70, 229, 0.75);
}
#tp-header h1 {
    margin: 0;
    font-size: 1.95rem;
    font-weight: 750;
    letter-spacing: -0.028em;
    color: #fff;
}
#tp-header p {
    margin: 8px 0 0;
    opacity: 0.9;
    font-size: 0.97rem;
    line-height: 1.5;
}
#tp-header .tp-tags { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
#tp-header .tp-tag {
    background: rgba(255, 255, 255, 0.17);
    border: 1px solid rgba(255, 255, 255, 0.26);
    border-radius: 999px;
    padding: 4px 13px;
    font-size: 0.78rem;
    font-weight: 500;
    backdrop-filter: blur(6px);
}

/* ---------- pipeline tracker ---------- */
.pipeline {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 4px 0 2px;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 0.83rem;
    font-weight: 550;
    border: 1px solid var(--tp-line);
    transition: all 0.25s ease;
}
.chip-icon { font-size: 0.95rem; line-height: 1; }
.chip-idle { color: var(--tp-muted); opacity: 0.55; }
.chip-active {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-color: transparent;
    color: #fff;
    box-shadow: 0 6px 18px -6px rgba(99, 102, 241, 0.9);
    animation: tp-pulse 1.4s ease-in-out infinite;
}
.chip-done {
    background: rgba(16, 185, 129, 0.13);
    border-color: rgba(16, 185, 129, 0.42);
    color: #059669;
}
.chip-done .chip-label::after { content: " ✓"; font-weight: 700; }
@keyframes tp-pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50%      { transform: scale(1.045); opacity: 0.9; }
}
.dark .chip-done { color: #34d399; }

/* ---------- panels ---------- */
#tp-status {
    font-size: 0.87rem;
    color: var(--tp-muted);
    padding: 2px 0 0 2px;
    min-height: 20px;
}
.tp-panel {
    border: 1px solid var(--tp-line);
    border-radius: 16px;
    padding: 18px 22px;
    background: var(--block-background-fill);
}
.tp-panel h4 { margin-top: 1.1em; }
.tp-panel table { font-size: 0.9rem; }
.tp-panel blockquote {
    border-left: 3px solid var(--tp-accent);
    padding-left: 14px;
    color: var(--tp-muted);
    margin: 14px 0;
}
footer { display: none !important; }
"""

HEADER = """
<div id="tp-header">
  <h1>✈️ Autonomous Trip Planner</h1>
  <p>Eleven specialist agents research, book, route, cost and fact-check your
     trip — then put it straight on your calendar.</p>
  <div class="tp-tags">
    <span class="tp-tag">🕸 LangGraph</span>
    <span class="tp-tag">🧠 Gemini 3.5 Flash</span>
    <span class="tp-tag">🔌 MCP · Google Calendar</span>
    <span class="tp-tag">🌐 Live flight &amp; hotel data</span>
  </div>
</div>
"""


def _sections(state: dict, backend: str) -> list[str]:
    """Render every result panel from the accumulated state.

    Args:
        state: The merged graph state so far.
        backend: Description of the active calendar backend.

    Returns:
        The Markdown for each tab, in the order the UI expects.
    """
    return [
        render_itinerary(state.get("itinerary"), state.get("routing")),
        render_flights(state.get("flights")),
        render_lodging(state.get("lodging")),
        render_places(state.get("attractions")),
        render_budget(state.get("budget")),
        render_critic(state.get("critic")),
        render_calendar(state.get("calendar"), backend),
        render_profile(state.get("intake")),
        render_research(state.get("research")),
    ]


def plan(request: str):
    """Run the planner, yielding UI updates as each agent finishes.

    Args:
        request: The traveler's free-text request.

    Yields:
        The progress tracker, a status line, every result panel, and the
        downloadable .ics file once the Calendar Agent has run.
    """
    labels = {str(name): label for name, label, _ in PIPELINE}
    backend = describe_calendar_backend()
    state: dict = {}

    if not request.strip():
        yield (
            progress_html([]),
            "⚠️ Tell me where you want to go first.",
            *_sections({}, backend),
            gr.update(visible=False),
        )
        return

    yield (
        progress_html([], active="intake"),
        "🚀 Starting — the Travel Manager is deciding what to do first…",
        *_sections({}, backend),
        gr.update(visible=False),
    )

    try:
        stream = graph_app.stream(
            {
                "user_request": request,
                "messages": [{"role": "user", "content": request}],
                "completed_agents": [],
                "revision_count": 0,
            },
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="updates",
        )

        for chunk in stream:
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue

                # Merge this node's partial update into our view of the state.
                for key, value in update.items():
                    if key == "completed_agents":
                        state.setdefault("completed_agents", [])
                        state["completed_agents"] += value
                    else:
                        state[key] = value

                completed = state.get("completed_agents", [])
                if node == "manager":
                    decision = state.get("manager_decision")
                    nxt = decision.next_agent if decision else None
                    status = (
                        f"🧭 Manager → **{labels.get(str(nxt), nxt)}**"
                        if nxt
                        else "🧭 Manager: the plan is complete."
                    )
                    active = str(nxt) if nxt else None
                else:
                    status = f"✅ {labels.get(node, node)} finished."
                    active = None

                if state.get("revision_count"):
                    status += f"  ·  🔄 revision {state['revision_count']}"

                yield (
                    progress_html(completed, active),
                    status,
                    *_sections(state, backend),
                    gr.update(visible=False),
                )

    except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
        message = str(exc)
        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            note = (
                "🛑 **Gemini quota exhausted.** A free-tier key allows 20 requests "
                "per day, and a full eleven-agent run needs more than that. "
                "Enable billing on the key, or try again tomorrow."
            )
        else:
            note = f"🛑 **The run failed:** {message[:400]}"

        yield (
            progress_html(state.get("completed_agents", [])),
            note,
            *_sections(state, backend),
            gr.update(visible=False),
        )
        return

    # Offer the .ics for download when one was written locally.
    calendar = state.get("calendar")
    ics = getattr(calendar, "ics_path", None) if calendar else None
    intake = state.get("intake")

    if intake is not None and not intake.is_complete:
        final = "❓ I need a few more details before I can plan this trip."
    else:
        final = "🎉 **Your trip is ready.** " + (
            "Events were added to your Google Calendar."
            if calendar and not ics and calendar.events
            else "Download the calendar file below."
            if ics
            else ""
        )

    yield (
        progress_html(state.get("completed_agents", [])),
        final,
        *_sections(state, backend),
        gr.update(value=ics, visible=bool(ics)),
    )


# Fonts must be Font objects, not bare strings: Gradio compares the theme
# against its built-ins at launch, and mixing the two breaks that check.
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=(
        gr.themes.GoogleFont("Inter"),
        gr.themes.Font("Segoe UI"),
        gr.themes.Font("system-ui"),
        gr.themes.Font("sans-serif"),
    ),
)


def build_ui() -> gr.Blocks:
    """Assemble the Gradio interface.

    The theme and CSS are applied at launch, which is where Gradio 6 expects
    them, so they are not passed to `Blocks` here.

    Returns:
        The Blocks app, ready to launch.
    """
    with gr.Blocks(title="Autonomous Trip Planner") as demo:
        gr.HTML(HEADER)

        with gr.Row():
            with gr.Column(scale=2):
                request = gr.Textbox(
                    label="Describe your trip",
                    placeholder=(
                        "Where to, when, how many of you, what budget, and what "
                        "you enjoy — one sentence is enough."
                    ),
                    lines=4,
                    autofocus=True,
                )
            with gr.Column(scale=1, min_width=190):
                plan_button = gr.Button(
                    "✨ Plan my trip", variant="primary", size="lg", scale=2
                )
                gr.Markdown(
                    f"<div id='tp-status'>📅 {describe_calendar_backend()}</div>"
                )

        gr.Examples(examples=EXAMPLES, inputs=request, label="Try one of these")

        progress = gr.HTML(progress_html([]))
        status = gr.Markdown("", elem_id="tp-status")

        with gr.Tabs():
            with gr.Tab("📖 Itinerary"):
                itinerary = gr.Markdown(elem_classes="tp-panel")
            with gr.Tab("✈️ Flights"):
                flights = gr.Markdown(elem_classes="tp-panel")
            with gr.Tab("🏨 Stay"):
                lodging = gr.Markdown(elem_classes="tp-panel")
            with gr.Tab("📍 Places"):
                places = gr.Markdown(elem_classes="tp-panel")
            with gr.Tab("💰 Budget"):
                budget = gr.Markdown(elem_classes="tp-panel")
            with gr.Tab("🔍 Review"):
                critic = gr.Markdown(elem_classes="tp-panel")
            with gr.Tab("📅 Calendar"):
                calendar = gr.Markdown(elem_classes="tp-panel")
            with gr.Tab("📋 Your request"):
                profile = gr.Markdown(elem_classes="tp-panel")
            with gr.Tab("🌍 Destination"):
                research = gr.Markdown(elem_classes="tp-panel")

        ics_file = gr.File(label="📎 Add to your calendar (.ics)", visible=False)

        outputs = [
            progress,
            status,
            itinerary,
            flights,
            lodging,
            places,
            budget,
            critic,
            calendar,
            profile,
            research,
            ics_file,
        ]
        plan_button.click(plan, inputs=request, outputs=outputs)
        request.submit(plan, inputs=request, outputs=outputs)

    return demo


def main() -> None:
    """Launch the trip planner UI in a browser."""
    build_ui().launch(theme=THEME, css=CSS, inbrowser=True)


if __name__ == "__main__":
    main()
