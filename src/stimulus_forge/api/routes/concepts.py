"""Endpoints for structured related-concept generation and RAG file extraction."""

from __future__ import annotations

import os
import time
from typing import Any
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from stimulus_forge.config import config

from stimulus_forge.api.cache import concept_cache
from stimulus_forge.api.schemas import (
    BatchConceptRequest,
    BatchConceptResponse,
    ConceptGenerateRequest,
    ConceptGenerateResponse,
    ConceptItem,
)
from stimulus_forge.api.telemetry import telemetry
from stimulus_forge.services.agentic_rag import (
    extract_concepts_with_rag,
    extract_raw_text,
    is_straightforward_concept_list,
)
from stimulus_forge.services.questionnaire_intelligence import contains_phi, parse_questionnaire
from stimulus_forge.services.security import detect_prompt_injection
from stimulus_forge.services.concept_generation import (
    DEFAULT_CONCEPT_COUNT,
    RelatedConceptGenerator,
)


router = APIRouter(prefix="/concepts", tags=["Concepts"])


def _estimate_tokens(concept: str, count: int, output_concepts: list[dict[str, str]]) -> tuple[int, int]:
    """Estimate prompt and completion token counts when provider doesn't report exact usage."""
    prompt_tokens = 280 + len(concept.split())
    completion_tokens = max(20, sum(len(c.get("name", "").split()) + len(c.get("description", "").split()) + 15 for c in output_concepts))
    return prompt_tokens, completion_tokens


@router.post("/generate", response_model=ConceptGenerateResponse, summary="Generate Structured Related Concepts")
def generate_concepts(request: ConceptGenerateRequest) -> ConceptGenerateResponse:
    """Generate high-confidence, visually distinctive related concepts for a seed concept."""
    if contains_phi(request.concept) or (request.context and contains_phi(request.context)):
        raise HTTPException(
            status_code=400,
            detail="Concept contains sensitive personal information or identifiers (PHI) and cannot be processed.",
        )
    is_inj, pattern = detect_prompt_injection(request.concept)
    if not is_inj and request.context:
        is_inj, pattern = detect_prompt_injection(request.context)
    if is_inj:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt injection or instruction override detected in concept ('{pattern}').",
        )
    start_time = time.perf_counter()
    req_context = request.context or ""

    # 1. Check Cache
    if request.use_cache:
        cached_data = concept_cache.get(request.concept, request.count, context=req_context)
        if cached_data is not None:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            telemetry.record_concept_generation(
                provider=request.provider,
                model=request.model or "cached",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                cached=True,
            )
            return ConceptGenerateResponse(
                concept=request.concept,
                provider=request.provider,
                model=request.model or "cached",
                concepts=[ConceptItem(**c) for c in cached_data],
                cached=True,
                latency_ms=round(latency_ms, 2),
                prompt_tokens=0,
                completion_tokens=0,
                estimated_cost_usd=0.0,
            )

    # 2. Invoke Selected Provider
    try:
        default_google_model = os.getenv(
            "STIMULUS_FORGE_GOOGLE_MODEL",
            config.get("google_concept_model", "gemini-3.8-flash"),
        )
        model_name = request.model or (
            "gpt-4o"
            if request.provider == "openai"
            else "claude-3-5-sonnet-20241022"
            if request.provider == "anthropic"
            else default_google_model
            if request.provider == "google"
            else "local-gguf"
        )

        generator = RelatedConceptGenerator(
            provider=request.provider,
            openai_model=model_name,
            local_model_path=request.local_model_path or "",
            api_key=request.api_key or "",
            base_url=request.base_url or "",
        )
        generated = generator.generate(request.concept, request.count, context=req_context)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Concept generation failed via [{request.provider}]: {exc}",
        ) from exc

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    prompt_tokens, completion_tokens = _estimate_tokens(request.concept, request.count, generated)

    # 3. Record Telemetry & FinOps
    metric = telemetry.record_concept_generation(
        provider=request.provider,
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        cached=False,
    )

    # 4. Update Cache with fresh results
    concept_cache.set(request.concept, request.count, generated, context=req_context)

    return ConceptGenerateResponse(
        concept=request.concept,
        provider=request.provider,
        model=model_name,
        concepts=[ConceptItem(**c) for c in generated],
        cached=False,
        latency_ms=round(latency_ms, 2),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost_usd=metric.estimated_cost_usd,
    )


@router.post("/cache/clear", summary="Clear Related Concept Cache")
def clear_concept_cache() -> dict[str, Any]:
    """Evict all cached related concepts so subsequent generations run fresh."""
    cleared_count = concept_cache.size()
    concept_cache.clear()
    telemetry.reset_cache_stats()
    return {
        "status": "success",
        "cleared_entries": cleared_count,
        "message": f"Cleared {cleared_count} cached concept entry(ies).",
    }



@router.post("/batch", response_model=BatchConceptResponse, summary="Batch Generate Across Concepts")
def batch_generate_concepts(request: BatchConceptRequest) -> BatchConceptResponse:
    """Run batch concept generation across multiple seeds with individual error isolation."""
    start_time = time.perf_counter()
    results: dict[str, list[ConceptItem]] = {}
    errors: dict[str, str] = {}
    total_tokens = 0
    total_cost = 0.0

    for concept in request.concepts:
        clean_concept = concept.strip()
        if not clean_concept:
            continue
        if contains_phi(clean_concept):
            errors[clean_concept] = "Concept contains sensitive personal identifiers (PHI) and cannot be processed."
            continue
        is_inj, _ = detect_prompt_injection(clean_concept)
        if is_inj:
            errors[clean_concept] = "Prompt injection or instruction override detected in concept."
            continue
        try:
            single_req = ConceptGenerateRequest(
                concept=clean_concept,
                count=request.count,
                provider=request.provider,
                model=request.model,
                local_model_path=request.local_model_path,
                api_key=request.api_key,
                base_url=request.base_url,
                use_cache=request.use_cache,
            )
            resp = generate_concepts(single_req)
            results[clean_concept] = resp.concepts
            total_tokens += resp.prompt_tokens + resp.completion_tokens
            total_cost += resp.estimated_cost_usd
        except Exception as exc:
            errors[clean_concept] = str(exc)

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0

    return BatchConceptResponse(
        total_requested=len(request.concepts),
        successful=len(results),
        results=results,
        errors=errors,
        total_latency_ms=round(total_latency_ms, 2),
        total_estimated_cost_usd=round(total_cost, 4),
        total_tokens=total_tokens,
    )


@router.api_route("/upload", methods=["GET", "POST"], summary="Upload File & Extract Concepts (with Agentic RAG Fallback)")
@router.api_route("/upload/", methods=["GET", "POST"], include_in_schema=False)
async def upload_concept_file(
    file: UploadFile | None = File(None),
    provider: str = Form("local"),
    model: str = Form(""),
    api_key: str = Form(""),
    base_url: str = Form(""),
    local_model_path: str = Form(""),
    subject_id: str = Form(""),
) -> dict[str, Any]:
    """Parse concepts from uploaded file, with Agentic RAG intelligence for complex/unstructured documents."""
    if file is None:
        return {
            "status": "ready",
            "message": "Concept file upload endpoint is active. Send a multipart POST with 'file'.",
            "concepts": [],
        }
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}") from exc

    filename = file.filename or "file.txt"
    sub_val = subject_id if isinstance(subject_id, str) else str(getattr(subject_id, "default", "") or "")
    prov_val = provider if isinstance(provider, str) else str(getattr(provider, "default", "local") or "local")
    mod_val = model if isinstance(model, str) else str(getattr(model, "default", "") or "")
    key_val = api_key if isinstance(api_key, str) else str(getattr(api_key, "default", "") or "")
    url_val = base_url if isinstance(base_url, str) else str(getattr(base_url, "default", "") or "")
    path_val = local_model_path if isinstance(local_model_path, str) else str(getattr(local_model_path, "default", "") or "")

    # 1. Specialized Questionnaire Intelligence for Clinical / Neuroscience Survey Spreadsheets
    q_res = parse_questionnaire(content, filename=filename, subject_id=sub_val)
    if q_res.is_questionnaire and (q_res.selected_concepts or q_res.all_concepts):
        concepts = q_res.selected_concepts
        excerpts = q_res.selected_excerpts
        method = "questionnaire_rag"

        # Questionnaire processing runs 100% locally: zero clinical responses or PHI leave the machine.
        # Build full lookup across all subjects
        all_excerpts = {}
        for s in q_res.subjects:
            all_excerpts.update(q_res.excerpts_by_subject.get(s, {}))

        return {
            "filename": filename,
            "concepts": concepts,
            "method": method,
            "rag_excerpts": excerpts,
            "subjects": q_res.subjects,
            "selected_subject": q_res.selected_subject,
            "concepts_by_subject": q_res.concepts_by_subject,
            "excerpts_by_subject": q_res.excerpts_by_subject,
            "all_concepts": q_res.all_concepts,
            "all_excerpts": all_excerpts,
            "message": f"Extracted {len(concepts)} visual concepts for {q_res.selected_subject or 'all'} from {filename}.",
        }

    # 2. General Document Parsing (Text, docx, json, standard csv)
    raw_text = extract_raw_text(filename, content)

    # Check if straightforward list
    is_straightforward, direct_concepts = is_straightforward_concept_list(raw_text)
    if is_straightforward and len(direct_concepts) >= 2:
        return {
            "filename": filename,
            "concepts": direct_concepts,
            "method": "deterministic",
            "rag_excerpts": {},
            "subjects": [],
            "selected_subject": "",
            "concepts_by_subject": {},
            "message": f"Successfully parsed {len(direct_concepts)} concepts directly from {filename}.",
        }

    # 3. Execute Agentic RAG Document Intelligence for unstructured documents
    try:
        generator = RelatedConceptGenerator(
            provider=provider,
            openai_model=model or "gpt-4o",
            local_model_path=local_model_path,
            api_key=api_key,
            base_url=base_url,
        )
        extracted_concepts, excerpts = extract_concepts_with_rag(raw_text, generator, max_concepts=30)
    except Exception:
        # Fallback regex extraction if generator fails
        from stimulus_forge.services.agentic_rag import _fallback_regex_extraction
        extracted_concepts, excerpts = _fallback_regex_extraction(raw_text, limit=25)

    if not extracted_concepts:
        raise HTTPException(
            status_code=422,
            detail="Could not extract visual concepts from this document. Please check file contents.",
        )

    return {
        "filename": filename,
        "concepts": extracted_concepts,
        "method": "agentic_rag",
        "rag_excerpts": excerpts,
        "subjects": [],
        "selected_subject": "",
        "concepts_by_subject": {},
        "message": f"Extracted {len(extracted_concepts)} visual concepts from {filename} using Agentic RAG.",
    }
