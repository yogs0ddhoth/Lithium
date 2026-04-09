"""Application entrypoint.

Exports
-------
- ``orient`` — the compiled Orient LangGraph graph (referenced by langgraph.json)
- ``app``    — the FastAPI ASGI application (used by uvicorn)

Running
-------
LangGraph dev server::

    langgraph dev

Uvicorn (production)::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from app.orient.graph import builder
from app.server import app

orient = builder.compile(name="orient")
__all__ = ["orient", "app"]
