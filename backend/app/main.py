"""FastAPI application factory."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import auth, chat, claims, documents, staff
from app.config import get_settings
from app.db import init_db, query

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

    for router in (auth.router, claims.router, documents.router, chat.router, staff.router):
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

    @app.get("/")
    async def index() -> dict[str, Any]:
        return {
            "service": settings.app_name,
            "docs": "/docs",
            "health": "/health",
            "ui": "Run `streamlit run frontend/app.py` for the portal.",
        }

    return app


app = create_app()
