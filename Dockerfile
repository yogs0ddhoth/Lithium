# ── Stage 1: dependency installer ────────────────────────────────────────────
# Resolve and compile all wheels into a venv before touching application source
# so this expensive layer is cached as long as only app code changes.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# 1. Copy only the files needed to resolve the dependency graph.
#    The local xml-pydantic package must be present because uv reads its
#    metadata to build a wheel (--no-editable → wheel, not symlink).
COPY pyproject.toml uv.lock ./
COPY packages/ packages/

# 2. Install all third-party dependencies (not the project itself).
#    --frozen      : fail if the lockfile is stale (no silent mutations)
#    --no-install-project : skip lithium itself until source is present
#    --no-dev      : exclude dev-only tools (pytest, ruff, …)
#    --extra postgres : include langgraph-checkpoint-postgres + psycopg
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra postgres

# 3. Copy application source, then install the project package.
COPY app/ app/
COPY prompts/ prompts/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra postgres --no-editable

# ── Stage 2: minimal runtime ─────────────────────────────────────────────────
# Python version must exactly match the builder so binary extensions are
# compatible. Use the same Debian release (bookworm) to match glibc.
FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

# Compiled venv from the builder — no compiler, no uv, no build tools needed.
COPY --from=builder /app/.venv /app/.venv

# Application source (not baked into the installed wheel; uvicorn needs it
# importable from the working directory as `app.*`).
COPY --from=builder /app/app ./app
COPY --from=builder /app/prompts ./prompts

# Prepend the venv so `uvicorn`, `python`, etc. resolve from there.
ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANGCHAIN_TRACING_V2=false

# Checkpointer default. Override at deploy time:
#   CHECKPOINTER=postgres + DATABASE_URL  → AsyncPostgresSaver (multi-worker safe)
#   CHECKPOINTER=memory                   → MemorySaver (single worker only)
ENV CHECKPOINTER=memory

EXPOSE 8000

# Fly.io injects $PORT (typically 8080); fall back to 8000 for local use.
# `exec` replaces the shell so uvicorn receives SIGTERM directly.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
