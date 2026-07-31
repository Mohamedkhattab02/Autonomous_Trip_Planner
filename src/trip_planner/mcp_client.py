"""MCP (Model Context Protocol) integration.

The Calendar Agent talks to the traveler's real Google Calendar through an MCP
server rather than through hand-written API code. That is the point of MCP: the
server owns the OAuth flow and the Google API calls, and the agent just
discovers whatever tools the server advertises.

Two things make this safe to depend on:

* **It degrades instead of failing.** If no MCP server is configured, or the
  subprocess cannot start, `load_calendar_tools()` falls back to the local
  `.ics` tools. The planner always produces a calendar, with or without OAuth.
* **It bridges async to sync.** MCP tools are async-native, while the LangGraph
  nodes are sync. `run_async()` runs a coroutine safely whether or not the
  caller already has an event loop running.

Configure it with environment variables (see `.env.example`):

    CALENDAR_MCP_ENABLED=true
    GOOGLE_OAUTH_CREDENTIALS=C:/path/to/gcp-oauth.keys.json
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, TypeVar

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

T = TypeVar("T")

# The MCP server the Calendar Agent connects to. Defaults to the community
# Google Calendar server, which is run straight from npm with no install step.
DEFAULT_MCP_COMMAND = "npx"
DEFAULT_MCP_ARGS = "-y @cocal/google-calendar-mcp"

# How long to wait for the server to start and list its tools. An npx cold
# start downloads the package, so this is generous.
STARTUP_TIMEOUT_SECONDS = 120.0


# --------------------------------------------------------------------------
# Async / sync bridge
# --------------------------------------------------------------------------


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from sync code.

    LangGraph nodes are sync functions, but MCP tools are async. Calling
    `asyncio.run()` blindly breaks when a loop is already running (as it is
    inside Gradio's server), so when that happens the coroutine is handed to a
    worker thread that owns its own loop.

    Args:
        coro: The coroutine to run.

    Returns:
        Whatever the coroutine returns.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop in this thread: the simple path.
        return asyncio.run(coro)

    # A loop is already running here, so run the coroutine somewhere else.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# --------------------------------------------------------------------------
# Server configuration
# --------------------------------------------------------------------------


def calendar_server_config() -> dict[str, Any] | None:
    """Build the MCP server configuration from the environment.

    Returns:
        A `MultiServerMCPClient` connection config, or None when Google
        Calendar over MCP has not been switched on.
    """
    if os.getenv("CALENDAR_MCP_ENABLED", "").strip().lower() not in ("1", "true", "yes"):
        return None

    command = os.getenv("CALENDAR_MCP_COMMAND", DEFAULT_MCP_COMMAND)
    args = os.getenv("CALENDAR_MCP_ARGS", DEFAULT_MCP_ARGS).split()

    # `npx` on Windows is a .cmd shim, which needs its full path to spawn.
    resolved = shutil.which(command)
    if resolved is None:
        logger.warning(
            "CALENDAR_MCP_ENABLED is set but '%s' is not on PATH. "
            "Falling back to local .ics export.",
            command,
        )
        return None

    # The server reads its OAuth client secret from this variable.
    env = {**os.environ}
    credentials = os.getenv("GOOGLE_OAUTH_CREDENTIALS")
    if credentials:
        env["GOOGLE_OAUTH_CREDENTIALS"] = credentials

    return {
        "transport": "stdio",
        "command": resolved,
        "args": args,
        "env": env,
    }


def is_calendar_mcp_enabled() -> bool:
    """Report whether Google Calendar over MCP is configured.

    Returns:
        True when the server config could be built.
    """
    return calendar_server_config() is not None


# --------------------------------------------------------------------------
# Tool loading
# --------------------------------------------------------------------------


async def _fetch_calendar_tools() -> list[BaseTool]:
    """Connect to the calendar MCP server and list the tools it advertises.

    Returns:
        The server's tools as LangChain tools, or an empty list on failure.
    """
    config = calendar_server_config()
    if config is None:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({"google_calendar": config})
        tools = await asyncio.wait_for(
            client.get_tools(), timeout=STARTUP_TIMEOUT_SECONDS
        )
    except Exception as exc:  # noqa: BLE001 - any failure degrades the same way
        logger.warning(
            "Could not load Google Calendar MCP tools (%s). "
            "Falling back to local .ics export.",
            exc,
        )
        return []

    logger.info(
        "Loaded %d Google Calendar tools over MCP: %s",
        len(tools),
        ", ".join(tool.name for tool in tools),
    )
    return list(tools)


# Tools are fetched once per process: starting the server is slow, and the
# tool list does not change while the app runs.
_cache: dict[str, list[BaseTool]] = {}
_cache_lock = threading.Lock()


def load_calendar_tools(force_reload: bool = False) -> tuple[list[BaseTool], bool]:
    """Return the Calendar Agent's tools, preferring the MCP server.

    Args:
        force_reload: Ignore the cached tool list and reconnect.

    Returns:
        A `(tools, using_mcp)` pair. When `using_mcp` is False the tools are
        the local `.ics` implementations and no Google Calendar was touched.
    """
    from trip_planner.tools.calendar_tools import CALENDAR_TOOLS

    with _cache_lock:
        if force_reload:
            _cache.clear()

        if "tools" not in _cache:
            _cache["tools"] = run_async(_fetch_calendar_tools())

        mcp_tools = _cache["tools"]

    if mcp_tools:
        return mcp_tools, True
    return CALENDAR_TOOLS, False


def describe_calendar_backend() -> str:
    """Describe which calendar backend is active, for logs and the UI.

    Returns:
        A short human-readable status line.
    """
    tools, using_mcp = load_calendar_tools()
    if using_mcp:
        names = ", ".join(tool.name for tool in tools)
        return f"Google Calendar via MCP ({len(tools)} tools: {names})"
    return "Local .ics export (Google Calendar MCP not configured)"
