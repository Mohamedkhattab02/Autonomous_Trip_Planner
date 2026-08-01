# Autonomous Trip Planner — Gradio interface.
#
# Node is installed alongside Python because the Google Calendar MCP server is
# fetched and run with npx at runtime; without it agent #11 silently falls back
# to writing a .ics file.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so a source change does not re-resolve the whole lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src/ ./src/
COPY app.py main.py ./
COPY scripts/ ./scripts/
RUN uv sync --frozen

# Written at runtime; mounted as volumes in compose so they survive a rebuild.
RUN mkdir -p /app/exports /app/.cache /app/.checkpoints /app/profiles

EXPOSE 7860

# Secrets are injected as environment variables, never baked into the image.
CMD ["uv", "run", "app.py"]
