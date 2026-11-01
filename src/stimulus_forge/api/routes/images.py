"""Image acquisition, search, generation, and transformation endpoints."""

from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse

from stimulus_forge.api.schemas import (
    ImageCollectionResponse,
    ImageGenerateRequest,
    ImageItem,
    ImageSearchRequest,
)
from stimulus_forge.config import config
from stimulus_forge.services.google_images import GoogleImageService
from stimulus_forge.services.huggingface_images import (
    DEFAULT_HF_IMAGE_MODEL,
    HuggingFaceImageService,
)
from stimulus_forge.services.image_workspace import TaskImagesDownloader
from stimulus_forge.services.local_images import LocalImageService
from stimulus_forge.services.openai_images import OpenAIImageService, image_name_stem
from stimulus_forge.services.questionnaire_intelligence import contains_phi
from stimulus_forge.services.security import detect_prompt_injection
from stimulus_forge.services.text_images import TextImageService


router = APIRouter(prefix="/images", tags=["Images"])


def _get_workspace_dir(custom_path: str | None = None) -> Path:
    if custom_path and str(custom_path).strip():
        base = Path(os.path.expanduser(str(custom_path).strip())).resolve()
    else:
        base = Path(config.get("task_images_folder", "data/task_images"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sanitize_subject_id(subject_id: str | None) -> str:
    """Sanitize subject/patient ID to a safe, valid directory name."""
    if not subject_id or not str(subject_id).strip():
        return ""
    clean = re.sub(r"[^\w.-]+", "_", str(subject_id).strip())
    clean = clean.strip("._")
    if clean.lower() in {"", ".", ".."}:
        return ""
    return clean


@router.post("/search", response_model=ImageCollectionResponse, summary="Download Safe Images via Bing")
def search_images(request: ImageSearchRequest) -> ImageCollectionResponse:
    """Acquire real-world visual stimulus candidates using strict SafeSearch Bing downloader."""
    if contains_phi(request.concept) or contains_phi(request.related_concept or "") or contains_phi(request.name or "") or (request.context and contains_phi(request.context)):
        raise HTTPException(
            status_code=400,
            detail="Image query contains sensitive personal identifiers (PHI) and cannot be searched.",
        )
    is_inj, pattern = detect_prompt_injection(request.concept)
    if not is_inj and request.related_concept:
        is_inj, pattern = detect_prompt_injection(request.related_concept)
    if not is_inj and request.context:
        is_inj, pattern = detect_prompt_injection(request.context)
    if is_inj:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt injection or instruction override detected in image query ('{pattern}').",
        )
    start_time = time.perf_counter()
    workspace = _get_workspace_dir(request.output_dir)
    downloader = TaskImagesDownloader()

    safe_subject = _sanitize_subject_id(request.subject_id)
    concept_slug = request.concept.lower().replace(" ", "_")
    if safe_subject:
        if request.related_concept and request.related_concept.strip():
            rel_slug = request.related_concept.lower().replace(" ", "_")
            output_dir = workspace / safe_subject / concept_slug / rel_slug
        else:
            output_dir = workspace / safe_subject / concept_slug
    else:
        output_dir = workspace / "downloads" / concept_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    min_w = max(16, int(request.min_width or request.target_width or 160))
    min_h = max(16, int(request.min_height or request.target_height or 160))
    target_w = max(16, int(request.target_width or min_w))
    target_h = max(16, int(request.target_height or min_h))

    saved_paths: list[str] = []
    try:
        saved_paths = downloader.download(
            concept=request.concept,
            related_concept=request.related_concept or "",
            name=request.name or "",
            limit=request.limit,
            output_dir=str(output_dir),
            provider=request.provider,
            safe_search="strict",  # always enforced; never fail on UI value
            safe_query_exclusions=request.safe_query_exclusions,
            filter=request.filter,
            timeout=12,
            verbose=False,
            min_width=min_w,
            min_height=min_h,
            context=request.context or "",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image download failed: {exc}") from exc

    # Only return images downloaded by THIS request (never stale files).
    image_items: list[ImageItem] = []
    target_files = [Path(p) for p in (saved_paths or []) if Path(p).exists()]

    for file in target_files:
        face_detected = False
        if request.crop_faces:
            try:
                face_detected = downloader.crop_face(str(file), target_size=(target_w, target_h))
            except Exception:
                face_detected = False
        elif request.detect_faces:
            try:
                det = downloader.detect_faces(str(file), max_faces=request.max_faces)
                face_detected = bool(det.get("face_detected", False))
            except Exception:
                face_detected = False

        image_items.append(
            ImageItem(
                id=str(uuid.uuid4())[:8],
                filename=file.name,
                local_path=str(file),
                url=f"/api/v1/images/file?path={file.as_posix()}",
                concept=request.concept,
                related_concept=request.related_concept or "",
                source=request.provider,
                moderated=True,
                face_detected=face_detected,
            )
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return ImageCollectionResponse(
        concept=request.concept,
        related_concept=request.related_concept or "",
        images=image_items[: request.limit],
        total_count=len(image_items),
        latency_ms=round(latency_ms, 2),
    )


@router.post("/generate", response_model=ImageCollectionResponse, summary="Synthesize Stimulus Image via GenAI")
def generate_images(request: ImageGenerateRequest) -> ImageCollectionResponse:
    """Generate visual stimulus using OpenAI DALL-E or Local Diffusion model."""
    if contains_phi(request.prompt) or contains_phi(request.concept) or contains_phi(request.related_concept or "") or contains_phi(request.name or "") or contains_phi(request.description or "") or (request.context and contains_phi(request.context)):
        raise HTTPException(
            status_code=400,
            detail="Image prompt contains sensitive personal identifiers (PHI) and cannot be generated.",
        )
    is_inj, pattern = detect_prompt_injection(request.prompt)
    if not is_inj and request.concept:
        is_inj, pattern = detect_prompt_injection(request.concept)
    if not is_inj and request.context:
        is_inj, pattern = detect_prompt_injection(request.context)
    if is_inj:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt injection or instruction override detected in image prompt ('{pattern}').",
        )
    start_time = time.perf_counter()
    workspace = _get_workspace_dir(request.output_dir)

    safe_subject = _sanitize_subject_id(request.subject_id)
    concept_name = request.concept or request.prompt
    concept_slug = concept_name.lower().replace(" ", "_") if concept_name else "stimulus"
    if safe_subject:
        if request.related_concept and request.related_concept.strip():
            rel_slug = request.related_concept.lower().replace(" ", "_")
            output_dir = workspace / safe_subject / "generated" / concept_slug / rel_slug
        else:
            output_dir = workspace / safe_subject / "generated" / concept_slug
    else:
        output_dir = workspace / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    if request.provider == "openai":
        try:
            q = "high" if request.quality in {"hd", "high"} else ("medium" if request.quality == "medium" else "low")
            service = OpenAIImageService(
                model=request.model,
                quality=q,
                size=request.size if request.size in {"1024x1024", "512x512"} else "1024x1024",
                api_key=request.api_key,
            )
            saved_paths = service.generate(
                concept=concept_name,
                related_concept=request.related_concept,
                name=request.name,
                description=request.description,
                output_dir=output_dir,
                limit=request.count,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"OpenAI image generation failed: {exc}") from exc
    elif request.provider == "google":
        try:
            model_name = request.model if request.model and "imagen" in request.model else "imagen-3.0-generate-002"
            service = GoogleImageService(
                model=model_name,
                api_key=request.api_key,
            )
            saved_paths = service.generate(
                concept=concept_name,
                related_concept=request.related_concept,
                name=request.name,
                description=request.description,
                output_dir=output_dir,
                limit=request.count,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Google Imagen generation failed: {exc}") from exc
    elif request.provider == "local":
        try:
            model_path = (
                request.local_model_path
                or os.getenv("STIMULUS_FORGE_LOCAL_IMAGE_MODEL")
                or config.get("local_image_model", "")
            )
            service = LocalImageService(model_path=model_path)
            saved_paths = service.generate(
                concept=concept_name,
                related_concept=request.related_concept,
                name=request.name,
                description=request.description,
                output_dir=output_dir,
                limit=request.count,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Local diffusion generation failed: {exc}") from exc
    elif request.provider == "huggingface":
        try:
            hf_model = (
                request.model
                if request.model and request.model != "dall-e-3"
                else (
                    os.getenv("STIMULUS_FORGE_HF_IMAGE_MODEL")
                    or config.get("huggingface_image_model", DEFAULT_HF_IMAGE_MODEL)
                )
            )
            service = HuggingFaceImageService(
                model=hf_model,
                api_key=request.api_key,
            )
            saved_paths = service.generate(
                concept=concept_name,
                related_concept=request.related_concept,
                name=request.name,
                description=request.description,
                output_dir=output_dir,
                limit=request.count,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Hugging Face image generation failed: {exc}"
            ) from exc
    elif request.provider == "text":
        try:
            if request.size and "x" in request.size:
                parts = request.size.split("x")
                img_size = (int(parts[0]), int(parts[1]))
            elif request.target_width and request.target_height and request.target_width > 160:
                img_size = (int(request.target_width), int(request.target_height))
            else:
                img_size = (512, 512)

            service = TextImageService(
                bg_color=request.text_bg_color,
                text_color=request.text_color,
                target_size=img_size,
                font_size=request.text_font_size,
            )
            saved_paths = service.generate(
                concept=concept_name,
                related_concept=request.related_concept,
                name=request.name,
                description=request.description,
                output_dir=output_dir,
                limit=request.count,
                target_size=img_size,
                bg_color=request.text_bg_color,
                text_color=request.text_color,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Text stimulus image generation failed: {exc}"
            ) from exc
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image generation provider: '{request.provider}'.",
        )

    downloader = TaskImagesDownloader()
    target_w = max(16, int(request.target_width or 160))
    target_h = max(16, int(request.target_height or 160))

    image_items: list[ImageItem] = []
    for path in saved_paths:
        face_detected = False
        if request.crop_faces:
            try:
                face_detected = downloader.crop_face(str(path), target_size=(target_w, target_h))
            except Exception:
                face_detected = False
        elif request.detect_faces:
            try:
                det = downloader.detect_faces(str(path), max_faces=request.max_faces)
                face_detected = bool(det.get("face_detected", False))
            except Exception:
                face_detected = False

        image_items.append(
            ImageItem(
                id=str(uuid.uuid4())[:8],
                filename=path.name,
                local_path=str(path),
                url=f"/api/v1/images/file?path={path.as_posix()}",
                concept=concept_name,
                source=request.provider,
                moderated=True,
                face_detected=face_detected,
            )
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return ImageCollectionResponse(
        concept=concept_name,
        images=image_items,
        total_count=len(image_items),
        latency_ms=round(latency_ms, 2),
    )


@router.get("/file", summary="Serve Image Binary")
def get_image_file(path: str = Query(..., description="Absolute or relative path to image")) -> FileResponse:
    """Serve an image file from the workspace for UI display."""
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found.")

    suffix = file_path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    return FileResponse(file_path, media_type=media_type)


def _resolve_image_path(path: str) -> Path:
    """Resolve an image path from absolute path, URL with query param, or relative workspace path."""
    if "path=" in path:
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(path)
        qs = parse_qs(parsed.query)
        if "path" in qs:
            path = qs["path"][0]

    file_path = Path(path).resolve()
    if file_path.is_file():
        return file_path

    roots = [_get_workspace_dir(), Path("data/image_workspace").resolve()]
    for root in roots:
        for sub in ["downloads", "generated", "bing", "wikimedia", "duckduckgo", "openverse", ""]:
            cand = (root / sub / path).resolve()
            if cand.is_file():
                return cand

    filename = Path(path).name
    for root in roots:
        downloads_dir = root / "downloads"
        if downloads_dir.is_dir():
            for found in downloads_dir.rglob(filename):
                if found.is_file():
                    return found
        if root.is_dir():
            for found in root.rglob(filename):
                if found.is_file():
                    return found

    return file_path


def _parse_crop_dimensions(
    width: int | None,
    height: int | None,
    payload: dict[str, Any] | None,
) -> tuple[int, int]:
    w = width
    h = height
    if payload:
        if w is None:
            w = payload.get("target_width") or payload.get("width")
        if h is None:
            h = payload.get("target_height") or payload.get("height")
        target_size = payload.get("target_size")
        if (w is None or h is None) and target_size is not None:
            if isinstance(target_size, (list, tuple)) and len(target_size) >= 2:
                w = w or int(target_size[0])
                h = h or int(target_size[1])
            elif isinstance(target_size, str) and "x" in target_size.lower():
                parts = target_size.lower().split("x", 1)
                w = w or int(parts[0].strip())
                h = h or int(parts[1].strip())
            elif isinstance(target_size, (int, float)):
                w = w or int(target_size)
                h = h or int(target_size)
    final_w = max(16, min(4096, int(w if w else 160)))
    final_h = max(16, min(4096, int(h if h else 160)))
    return final_w, final_h


@router.post("/crop-face", summary="Run Face Detection and Auto-Crop")
def crop_face(
    path: str = Query(None, description="Path to image to face-crop"),
    width: int | None = Query(None, ge=16, le=4096, description="Target crop width in pixels"),
    height: int | None = Query(None, ge=16, le=4096, description="Target crop height in pixels"),
    payload: dict[str, Any] = Body(None, description="Optional JSON payload with image_path and dimensions"),
) -> dict[str, Any]:
    """Detect frontal face and crop to normalized target_width x target_height stimulus."""
    target_path_str = path
    if not target_path_str and payload:
        target_path_str = payload.get("image_path") or payload.get("path")
    if not target_path_str:
        raise HTTPException(status_code=400, detail="Missing image path.")

    file_path = _resolve_image_path(target_path_str)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Target image file not found.")

    target_w, target_h = _parse_crop_dimensions(width, height, payload)
    face_box = None
    if payload:
        face_box = payload.get("face_box") or payload.get("face")

    try:
        downloader = TaskImagesDownloader()
        cropped = downloader.crop_face(
            str(file_path),
            target_size=(target_w, target_h),
            face_box=face_box,
        )
        if not cropped:
            return {
                "status": "no_face",
                "message": "No prominent frontal face detected to crop.",
                "path": str(file_path),
                "url": f"/api/v1/images/file?path={file_path.as_posix()}",
                "width": target_w,
                "height": target_h,
            }
        return {
            "status": "cropped",
            "message": f"Face cropped and image normalized to {target_w}x{target_h}.",
            "path": str(file_path),
            "url": f"/api/v1/images/file?path={file_path.as_posix()}",
            "width": target_w,
            "height": target_h,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face crop failed: {exc}") from exc


@router.get("/detect-face", summary="Detect Face Bounding Boxes (ROI)")
def detect_face(
    path: str = Query(..., description="Path to image for face detection"),
    max_faces: int = Query(5, ge=1, le=10, description="Maximum faces to detect (capped at 10)"),
    min_face_size: int = Query(80, ge=16, le=1024, description="Minimum face width/height in pixels"),
    target_width: int = Query(160, ge=16, le=4096, description="Target crop width in pixels"),
    target_height: int = Query(160, ge=16, le=4096, description="Target crop height in pixels"),
) -> dict[str, Any]:
    """Detect up to 10 prominent frontal faces and return bounding box ROIs without modifying the image."""
    file_path = _resolve_image_path(path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Target image file not found.")

    try:
        downloader = TaskImagesDownloader()
        result = downloader.detect_faces(
            str(file_path),
            max_faces=min(10, max_faces),
            min_face_size=min_face_size,
            target_size=(target_width, target_height),
        )
        return {
            "status": "success",
            **result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face detection failed: {exc}") from exc

