"""Health and system diagnostics endpoints."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

from fastapi import APIRouter


router = APIRouter(prefix="/health", tags=["Diagnostics"])


@router.get("", summary="Basic Health Check")
def health_check() -> dict[str, str]:
    """Return 200 OK if service is responding."""
    return {"status": "ok", "service": "stimstudio-api"}


@router.get("/diagnostics", summary="Hardware and System Diagnostics")
def system_diagnostics() -> dict[str, Any]:
    """Inspect local hardware acceleration, Python runtime, and host capabilities."""
    accel = "cpu"
    torch_available = False
    cuda_available = False
    mps_available = False

    try:
        import torch

        torch_available = True
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if cuda_available:
            accel = f"cuda ({torch.cuda.get_device_name(0)})"
        elif mps_available:
            accel = "mps (Apple Silicon Metal)"
    except ImportError:
        pass

    return {
        "status": "healthy",
        "version": "0.1.0",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 1,
        "torch_available": torch_available,
        "hardware_acceleration": accel,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
    }
