"""FinOps and telemetry inspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from stimulus_forge.api.schemas import TelemetrySummary
from stimulus_forge.api.telemetry import telemetry


router = APIRouter(prefix="/telemetry", tags=["FinOps & Observability"])


@router.get("/stats", response_model=TelemetrySummary, summary="Retrieve Aggregated FinOps & Latency Stats")
def get_telemetry_stats() -> TelemetrySummary:
    """Retrieve cost avoidance, token consumption, and p95 latency across all model backends."""
    return telemetry.get_summary()


@router.post("/reset", summary="Reset Telemetry Counters")
def reset_telemetry() -> dict[str, str]:
    """Clear in-memory telemetry records."""
    telemetry.reset()
    return {"status": "success", "message": "Telemetry metrics reset."}
