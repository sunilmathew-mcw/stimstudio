"""End-to-end async stimulus batch pipeline with job state tracking and SSE streaming."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from stimulus_forge.api.routes.concepts import generate_concepts
from stimulus_forge.api.routes.images import search_images
from stimulus_forge.api.schemas import (
    ConceptGenerateRequest,
    ImageSearchRequest,
    PipelineBatchRequest,
    PipelineJobStatus,
)


router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# In-memory store for async jobs
_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, PipelineJobStatus] = {}


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _JOBS_LOCK:
        if job_id in _JOBS:
            job = _JOBS[job_id]
            data = job.model_dump()
            data.update(kwargs)
            data["updated_at"] = datetime.now(timezone.utc)
            _JOBS[job_id] = PipelineJobStatus(**data)


def _execute_pipeline_worker(job_id: str, request: PipelineBatchRequest) -> None:
    """Synchronous worker thread executing the multi-step stimulus creation pipeline."""
    try:
        _update_job(
            job_id,
            status="processing",
            current_step="Initializing batch pipeline",
            progress_percent=5.0,
            total_concepts=len(request.concepts),
        )

        all_results: dict[str, Any] = {}
        total_images = 0
        concepts_processed = 0
        total_seeds = len(request.concepts)

        for idx, seed in enumerate(request.concepts):
            seed = seed.strip()
            if not seed:
                continue

            # Step 1: Generate related concepts
            _update_job(
                job_id,
                current_step=f"Generating related concepts for '{seed}' ({idx+1}/{total_seeds})",
                progress_percent=round(10.0 + (idx / total_seeds) * 40.0, 1),
            )

            concept_req = ConceptGenerateRequest(
                concept=seed,
                count=request.concepts_per_seed,
                provider=request.text_provider,
                model=request.text_model,
                use_cache=True,
            )
            concept_resp = generate_concepts(concept_req)
            related = concept_resp.concepts

            seed_summary = {
                "seed": seed,
                "related_concepts": [c.model_dump() for c in related],
                "images": [],
            }

            # Step 2: Download or synthesize images
            _update_job(
                job_id,
                current_step=f"Acquiring images for '{seed}' visual concepts",
                progress_percent=round(50.0 + (idx / total_seeds) * 45.0, 1),
            )

            if request.image_source in {"bing", "wikimedia", "duckduckgo", "openverse"}:
                for rel_item in related:
                    img_req = ImageSearchRequest(
                        concept=seed,
                        related_concept=rel_item.name,
                        name=rel_item.char,
                        limit=request.images_per_concept,
                        provider=request.image_source,
                        safe_search="strict",
                        min_width=request.target_width,
                        min_height=request.target_height,
                        detect_faces=request.detect_faces,
                        max_faces=request.max_faces,
                        crop_faces=request.crop_faces,
                        target_width=request.target_width,
                        target_height=request.target_height,
                        output_dir=request.output_dir,
                        subject_id=request.subject_id,
                    )
                    try:
                        img_resp = search_images(img_req)
                        total_images += len(img_resp.images)
                        seed_summary["images"].extend([img.model_dump() for img in img_resp.images])
                    except Exception:
                        pass
            elif request.image_source in {"openai", "google", "local", "huggingface", "text"}:
                for rel_item in related:
                    gen_req = ImageGenerateRequest(
                        prompt=f"{rel_item.name} {rel_item.char}".strip(),
                        concept=seed,
                        related_concept=rel_item.name,
                        name=rel_item.char,
                        description=rel_item.desc,
                        count=request.images_per_concept,
                        provider=request.image_source,
                        local_model_path=request.local_model_path,
                        detect_faces=request.detect_faces,
                        max_faces=request.max_faces,
                        crop_faces=request.crop_faces,
                        target_width=request.target_width,
                        target_height=request.target_height,
                        output_dir=request.output_dir,
                        subject_id=request.subject_id,
                    )
                    try:
                        img_resp = generate_images(gen_req)
                        total_images += len(img_resp.images)
                        seed_summary["images"].extend([img.model_dump() for img in img_resp.images])
                    except Exception:
                        pass

            concepts_processed += 1
            all_results[seed] = seed_summary
            _update_job(
                job_id,
                completed_concepts=concepts_processed,
                total_images_generated=total_images,
            )

        _update_job(
            job_id,
            status="completed",
            progress_percent=100.0,
            current_step="Completed stimulus pipeline",
            completed_at=datetime.now(timezone.utc),
            summary={
                "seeds_processed": len(all_results),
                "total_images": total_images,
                "results": all_results,
            },
        )

    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            current_step=f"Failed: {exc}",
            completed_at=datetime.now(timezone.utc),
        )


@router.post("/batch", response_model=PipelineJobStatus, summary="Trigger Asynchronous Batch Pipeline")
def start_batch_pipeline(
    request: PipelineBatchRequest,
    background_tasks: BackgroundTasks,
) -> PipelineJobStatus:
    """Submit an asynchronous job to generate related concepts and acquire images for stimulus sets."""
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    job = PipelineJobStatus(
        job_id=job_id,
        status="queued",
        progress_percent=0.0,
        current_step="Queued for processing",
        total_concepts=len(request.concepts),
        completed_concepts=0,
        total_images_generated=0,
        created_at=now,
        updated_at=now,
    )

    with _JOBS_LOCK:
        _JOBS[job_id] = job

    # Dispatch to background task execution
    background_tasks.add_task(_execute_pipeline_worker, job_id, request)

    return job


@router.get("/status/{job_id}", response_model=PipelineJobStatus, summary="Check Pipeline Job Status")
def get_job_status(job_id: str) -> PipelineJobStatus:
    """Retrieve the real-time execution status and result summary of a pipeline job."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Pipeline job not found.")
    return job


@router.get("/stream/{job_id}", summary="Stream Real-Time Job Progress (SSE)")
async def stream_job_progress(job_id: str):
    """Server-Sent Events (SSE) endpoint to stream live pipeline execution updates."""
    with _JOBS_LOCK:
        if job_id not in _JOBS:
            raise HTTPException(status_code=404, detail="Pipeline job not found.")

    async def event_generator():
        while True:
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
            if not job:
                break

            data = job.model_dump_json()
            yield f"data: {data}\n\n"

            if job.status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs", response_model=list[PipelineJobStatus], summary="List Recent Pipeline Jobs")
def list_jobs() -> list[PipelineJobStatus]:
    """Return list of all tracked pipeline jobs."""
    with _JOBS_LOCK:
        return list(_JOBS.values())
