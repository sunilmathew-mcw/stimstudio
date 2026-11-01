"""FastAPI Application Factory and Server Launcher for Stimulus Forge."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from stimulus_forge.api.routes import (
    chat,
    concepts,
    health,
    images,
    pipeline,
    providers,
    system,
    telemetry,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Stimulus Forge API",
        description=(
            "Headless REST & Real-Time Streaming Service for Cognitive Stimulus Generation, "
            "Multimodal Image Acquisition, and FinOps Telemetry."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Enable CORS for web clients & microservices
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API v1 route blueprints
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(system.router, prefix="/api/v1")
    app.include_router(providers.router, prefix="/api/v1")
    app.include_router(concepts.router, prefix="/api/v1")
    app.include_router(images.router, prefix="/api/v1")
    app.include_router(pipeline.router, prefix="/api/v1")
    app.include_router(telemetry.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")

    # Mount Web Client static files if directory exists
    web_dir = Path(__file__).resolve().parent.parent / "web"
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


app = create_app()


def main() -> None:
    """CLI entrypoint for running the Stimulus Forge API server."""
    parser = argparse.ArgumentParser(description="Run the Stimulus Forge API server.")
    parser.add_argument(
        "--host",
        default=os.getenv("STIMULUS_FORGE_HOST", "0.0.0.0"),
        help="Host address to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("STIMULUS_FORGE_PORT", "8000")),
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development only)",
    )

    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Running the API server requires: pip install -e '.[server]' or pip install uvicorn fastapi"
        ) from exc

    print(f"\n========================================================")
    print(f"  Stimulus Forge API & Web Server")
    print(f"  Interactive Docs (Swagger): http://{args.host}:{args.port}/docs")
    print(f"  Alternative Docs (Redoc):   http://{args.host}:{args.port}/redoc")
    print(f"  Web Dashboard UI:           http://{args.host}:{args.port}/")
    print(f"========================================================\n")

    uvicorn.run("stimulus_forge.api.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
