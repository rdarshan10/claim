"""FastAPI application factory."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.v1 import auth, chat, claims, demo, documents, fnol, fnol_agentic, staff
from app.config import get_settings
from app.db import init_db, query

SHIELD_FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3b5bdb">'
    '<path d="M12 1.6 3.6 5v6.4c0 5.2 3.6 10 8.4 11.2 4.8-1.2 8.4-6 8.4-11.2V5L12 1.6Z"/>'
    "</svg>"
).encode("utf-8")

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
           '"msg":"%(message)s"}',
)
logger = logging.getLogger("claimcompanion")


def create_app() -> FastAPI:
    settings = get_settings()
    init_db()

    app = FastAPI(
        title=settings.app_name,
        description="Insurance claim query chatbot with document verification.",
        version="0.1.0-mvp",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:8000",
                       "http://127.0.0.1:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in (auth.router, claims.router, documents.router, chat.router,
                   fnol.router, fnol_agentic.router, staff.router, demo.router):
        app.include_router(router, prefix=settings.api_prefix)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """RFC 7807 problem+json; never leak internals to the caller."""
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "type": "about:blank",
                "title": "Internal Server Error",
                "status": 500,
                "detail": "Something went wrong on our side. Please try again.",
            },
            media_type="application/problem+json",
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        counts = {}
        for table in ("customer", "claim", "document", "kb_chunk", "audit_event"):
            row = query(f"SELECT COUNT(*) AS n FROM {table}")
            counts[table] = row[0]["n"] if row else 0
        return {
            "status": "ok",
            "llm_configured": bool(settings.llm_api_key) and settings.llm_enabled,
            "llm_model": settings.llm_model_primary,
            "counts": counts,
        }

    # ---- static frontend -------------------------------------------------
    # The UI is plain HTML/CSS/ES modules served by this same process: one
    # port, no CORS, and nothing to start separately. Mounted after the API
    # routers so /api/v1 and /health always win over a static path.
    static_dir = Path(__file__).resolve().parents[2] / "frontend" / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        def _page(filename: str) -> FileResponse:
            return FileResponse(static_dir / filename, media_type="text/html")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return _page("index.html")

        @app.get("/portal", include_in_schema=False)
        async def portal() -> FileResponse:
            return _page("portal.html")

        @app.get("/staff", include_in_schema=False)
        async def staff_console() -> FileResponse:
            return _page("staff.html")

        # Cost, savings, AI quality and the audit trail. Manager-only oversight
        # of the operation rather than claim work, so it is its own page.
        # The same shield the pages draw, so the tab icon matches the header.
        # Browsers request this on every page load; without it each navigation
        # logs a 404 that buries real errors in the console.
        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon() -> Response:
            return Response(content=SHIELD_FAVICON, media_type="image/svg+xml")

        @app.get("/oversight", include_in_schema=False)
        async def oversight() -> FileResponse:
            return _page("oversight.html")

        # The insurer's core system. Stands alone — the registration bot drives
        # it in a browser, and staff can open it to see what the bot sees.
        @app.get("/core-system", include_in_schema=False)
        async def core_system() -> FileResponse:
            return _page("core-system.html")

        # Experiment: the scripted bot and the agentic one, side by side.
        @app.get("/rpa-compare", include_in_schema=False)
        async def rpa_compare() -> FileResponse:
            return _page("rpa-compare.html")

        # One run, live: the step log beside the browser it is driving.
        @app.get("/rpa-run", include_in_schema=False)
        async def rpa_run() -> FileResponse:
            return _page("rpa-run.html")
    else:
        @app.get("/", include_in_schema=False)
        async def index() -> dict[str, Any]:
            return {
                "service": settings.app_name,
                "docs": "/docs",
                "health": "/health",
                "ui": "frontend/static is missing — the UI cannot be served.",
            }

    return app


app = create_app()
