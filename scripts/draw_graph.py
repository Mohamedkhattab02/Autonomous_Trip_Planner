"""Render the trip planner graph to graph.png.

Usage:
    uv run scripts/draw_graph.py

`draw_mermaid_png` renders through the mermaid.ink web service, so this needs
an internet connection. The Mermaid source is printed first, so the structure
is visible even if the render fails.
"""

from __future__ import annotations

from pathlib import Path

from trip_planner.graph import app

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "graph.png"


def main() -> None:
    """Print the graph as Mermaid text and save it as a PNG."""
    graph = app.get_graph()

    print("Mermaid source:\n")
    print(graph.draw_mermaid())

    graph.draw_mermaid_png(output_file_path=str(OUTPUT_PATH))
    print(f"\nSaved graph image to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
