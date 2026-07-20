"""Minimal health endpoint for CodeValid infrastructure (§10.2 — allowed seam).

Registers GET /health on the LangGraph API server. This module is referenced
via langgraph.json's "http": {"app": "open_deep_research.server:router"}.

Uses starlette directly (no fastapi dependency required) since langgraph_api
expects a Starlette instance and starlette is already present in the venv.
"""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


async def health(request: Request) -> JSONResponse:
    """Liveness probe required by CodeValid on port 6713."""
    return JSONResponse({"status": "ok"})


router = Starlette(routes=[Route("/health", health)])
