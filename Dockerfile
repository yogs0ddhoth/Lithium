FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Install dependencies first (layer cached until lockfile changes)
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
RUN uv sync --frozen --no-dev --extra postgres

# Copy application source
COPY app/ app/
COPY prompts/ prompts/

ENV PYTHONUNBUFFERED=1 \
    LANGCHAIN_TRACING_V2=false

EXPOSE 8000

# CHECKPOINTER=memory  → single-worker in-process (dev / non-persistent)
# CHECKPOINTER=postgres → requires DATABASE_URL; safe for multiple workers
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
