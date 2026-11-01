"""System utilities: native macOS file browser, model scanner, and path validation."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

from stimulus_forge.api.schemas import ScannedModelInfo


router = APIRouter(prefix="/system", tags=["System & File Browser"])


class ValidateModelRequest(BaseModel):
    path: str


class ValidateModelResponse(BaseModel):
    valid: bool
    path: str
    name: str = ""
    size_mb: float = 0.0
    message: str = ""


@router.api_route("/browse-file", methods=["GET", "POST"], summary="Open Native OS File Picker")
@router.api_route("/browse-file/", methods=["GET", "POST"], include_in_schema=False)
def browse_file_dialog(model_type: str = Query("all", description="Model type: 'llm', 'image', or 'all'")) -> dict[str, Any]:
    """Launch native macOS file picker dialog to select a model file."""
    if sys.platform != "darwin":
        return {
            "status": "unsupported",
            "path": None,
            "message": "Native file browser is supported on macOS. Enter path manually.",
        }

    if model_type == "image":
        file_types = '{"safetensors", "ckpt", "pt"}'
        prompt_text = "Select Local Diffusion Model (.safetensors / .ckpt)"
    elif model_type == "llm":
        file_types = '{"gguf", "bin"}'
        prompt_text = "Select Local GGUF Model (.gguf)"
    else:
        file_types = '{"gguf", "bin", "safetensors", "ckpt", "pt"}'
        prompt_text = "Select Model Checkpoint File"

    script = f'POSIX path of (choose file of type {file_types} with prompt "{prompt_text}")'
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            selected_path = proc.stdout.strip()
            return {
                "status": "selected",
                "path": selected_path,
                "message": "File selected successfully.",
            }
        return {
            "status": "cancelled",
            "path": None,
            "message": "File selection was cancelled.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "path": None,
            "message": f"Could not launch native file picker: {exc}",
        }


@router.api_route("/browse-folder", methods=["GET", "POST"], summary="Open Native OS Folder Picker")
@router.api_route("/browse-folder/", methods=["GET", "POST"], include_in_schema=False)
def browse_folder_dialog(prompt: str = Query("Select Image Download Folder")) -> dict[str, Any]:
    """Launch native macOS folder picker dialog to select an image download directory."""
    if sys.platform != "darwin":
        return {
            "status": "unsupported",
            "path": None,
            "message": "Native folder browser is supported on macOS. Enter path manually.",
        }

    script = f'POSIX path of (choose folder with prompt "{prompt}")'
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            selected_path = proc.stdout.strip().rstrip("/")
            return {
                "status": "selected",
                "path": selected_path,
                "message": "Folder selected successfully.",
            }
        return {
            "status": "cancelled",
            "path": None,
            "message": "Folder selection was cancelled.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "path": None,
            "message": f"Could not launch native folder picker: {exc}",
        }


@router.post("/validate-folder", summary="Validate Local Directory Path")
def validate_folder_path(payload: dict[str, str] = Body(...)) -> dict[str, Any]:
    """Validate whether a folder path exists or can be created."""
    path_str = payload.get("path", "").strip()
    if not path_str:
        return {"valid": False, "exists": False, "message": "Path is empty."}
    try:
        p = Path(os.path.expanduser(path_str)).resolve()
        if p.is_dir():
            return {"valid": True, "exists": True, "path": str(p), "message": "Directory exists"}
        if p.parent.is_dir() and os.access(str(p.parent), os.W_OK):
            return {"valid": True, "exists": False, "path": str(p), "message": "Will be created on download"}
        return {"valid": False, "exists": False, "path": str(p), "message": "Path not accessible"}
    except Exception as e:
        return {"valid": False, "exists": False, "message": str(e)}


@router.get("/config-paths", summary="Get Default Application Paths")
def get_config_paths() -> dict[str, Any]:
    """Return default resolved application directory paths."""
    from stimulus_forge.config import config
    task_images = Path(config.get("task_images_folder", "data/task_images")).resolve()
    downloads_default = task_images / "downloads"
    default_dir = str(downloads_default if downloads_default.exists() else task_images)
    return {
        "task_images_folder": str(task_images),
        "default_download_folder": default_dir,
        "default_image_dir": default_dir,
        "current_image_dir": default_dir,
    }


@router.api_route("/scan-models", methods=["GET", "POST"], response_model=list[ScannedModelInfo], summary="Scan Host for GGUF & Diffusion Models")
@router.api_route("/scan-models/", methods=["GET", "POST"], include_in_schema=False)
def scan_local_models() -> list[ScannedModelInfo]:
    """Search conventional directories for existing GGUF and Diffusion model files."""
    search_dirs = [
        Path.home() / "LLM",
        Path.home() / "models",
        Path.home() / "Downloads",
        Path.home() / "Desktop",
        Path.home() / ".cache" / "lm-studio" / "models",
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/Users/smathew/LLM"),
    ]

    discovered: list[ScannedModelInfo] = []
    seen_paths: set[str] = set()

    for base_dir in search_dirs:
        try:
            if not base_dir.is_dir():
                continue
            for ext in ("*.gguf", "*.bin", "*.safetensors", "*.ckpt"):
                for p in list(base_dir.glob(ext)) + list(base_dir.glob(f"*/{ext}")):
                    if p.is_file() and str(p) not in seen_paths:
                        seen_paths.add(str(p))
                        stat = p.stat()
                        size_mb = round(stat.st_size / (1024 * 1024), 1)
                        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                        discovered.append(
                            ScannedModelInfo(
                                name=p.name,
                                path=str(p.resolve()),
                                size_mb=size_mb,
                                last_modified=mtime,
                            )
                        )
        except Exception:
            continue

    return sorted(discovered, key=lambda m: m.name.lower())


@router.api_route("/validate-model", methods=["GET", "POST"], response_model=ValidateModelResponse, summary="Validate Model File Path")
def validate_model_path(request: ValidateModelRequest) -> ValidateModelResponse:
    """Check if the specified path points to an accessible model checkpoint file."""
    if not request.path.strip():
        return ValidateModelResponse(valid=False, path="", message="Path cannot be empty.")

    p = Path(os.path.expanduser(request.path.strip())).resolve()
    if not p.exists():
        return ValidateModelResponse(valid=False, path=str(p), message="File does not exist.")
    if not p.is_file():
        return ValidateModelResponse(valid=False, path=str(p), message="Path is not a regular file.")

    size_mb = round(p.stat().st_size / (1024 * 1024), 1)
    return ValidateModelResponse(
        valid=True,
        path=str(p),
        name=p.name,
        size_mb=size_mb,
        message=f"Valid model checkpoint ({size_mb} MB)",
    )
