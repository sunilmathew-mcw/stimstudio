"""Providers discovery and capability inspection endpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from stimulus_forge.api.schemas import ProviderInfo
from stimulus_forge.config import config


router = APIRouter(prefix="/providers", tags=["Providers"])


def _check_local_llm_available() -> tuple[bool, str]:
    from stimulus_forge.config import config

    model_path = os.getenv("STIMULUS_FORGE_LLM_MODEL", "") or config.get("llm_model", "")
    if not model_path or not Path(model_path).is_file():
        candidates = [
            Path.home() / "LLM" / "Qwen2.5-7B-Instruct-GGUF" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
            Path.home() / "LLM" / "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        ]
        for candidate in candidates:
            if candidate.is_file():
                model_path = str(candidate)
                break

    has_weights = bool(model_path and Path(model_path).is_file())
    try:
        import llama_cpp  # noqa: F401
        has_lib = True
    except ImportError:
        has_lib = False

    if has_weights and has_lib:
        return True, f"Configured GGUF model: {Path(model_path).name}"
    if not has_lib:
        return False, "llama-cpp-python not installed (pip install -e '.[llm]')"
    return False, "No GGUF model weights found in ~/LLM or STIMULUS_FORGE_LLM_MODEL"


def _check_local_diffusers_available() -> tuple[bool, str]:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        return True, "Diffusers and PyTorch runtime available"
    except ImportError:
        return False, "Diffusers not installed (pip install -e '.[local-images]')"


def _get_available_google_models() -> list[str]:
    defaults = [
        "gemini-3.8-flash",
        "gemini-3.8-pro",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return defaults
    try:
        from google import genai
        client = genai.Client(api_key=key)
        discovered: list[str] = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            clean = name.removeprefix("models/")
            if "gemini" in clean.lower() and not any(x in clean.lower() for x in ["embed", "imagen", "aqa"]):
                discovered.append(clean)
        if discovered:
            return list(dict.fromkeys(defaults + discovered))
    except Exception:
        pass
    return defaults


@router.get("", response_model=list[ProviderInfo], summary="List AI Providers & Capabilities")
def list_providers() -> list[ProviderInfo]:
    """Inspect availability, active credentials, and pricing models for all providers."""
    local_available, local_notes = _check_local_llm_available()
    local_img_available, local_img_notes = _check_local_diffusers_available()

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_google = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    has_compat = bool(os.getenv("STIMULUS_FORGE_COMPAT_BASE_URL"))
    has_hf = bool(
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_API_KEY")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    return [
        ProviderInfo(
            id="local",
            name="Local Llama.cpp (GGUF)",
            type="text",
            available=local_available,
            default_model="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
            supports_structured_outputs=True,
            pricing_per_1m_input_usd=0.0,
            pricing_per_1m_output_usd=0.0,
            notes=f"Zero cloud leakage, HIPAA/IRB air-gapped compliant. {local_notes}",
        ),
        ProviderInfo(
            id="openai",
            name="OpenAI Structured Outputs",
            type="text",
            available=has_openai,
            default_model=os.getenv("STIMULUS_FORGE_CONCEPT_MODEL", "gpt-4o"),
            available_models=[
                "gpt-5.6-luna",
                "gpt-4o",
                "gpt-4o-mini",
            ],
            supports_structured_outputs=True,
            pricing_per_1m_input_usd=2.50,
            pricing_per_1m_output_usd=10.00,
            notes="API key detected" if has_openai else "Set OPENAI_API_KEY",
        ),
        ProviderInfo(
            id="anthropic",
            name="Anthropic Claude (Messages API)",
            type="text",
            available=has_anthropic,
            default_model="claude-3-5-sonnet-20241022",
            available_models=[
                "claude-sonnet-4-6",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
            ],
            supports_structured_outputs=True,
            pricing_per_1m_input_usd=3.00,
            pricing_per_1m_output_usd=15.00,
            notes="API key detected" if has_anthropic else "Set ANTHROPIC_API_KEY",
        ),
        ProviderInfo(
            id="google",
            name="Google Gemini Flash",
            type="text",
            available=has_google,
            default_model=os.getenv(
                "STIMULUS_FORGE_GOOGLE_MODEL",
                config.get("google_concept_model", "gemini-3.8-flash"),
            ),
            available_models=_get_available_google_models(),
            supports_structured_outputs=True,
            pricing_per_1m_input_usd=0.10,
            pricing_per_1m_output_usd=0.40,
            notes="API key detected" if has_google else "Set GEMINI_API_KEY or GOOGLE_API_KEY",
        ),
        ProviderInfo(
            id="compatible",
            name="OpenAI-Compatible / Hosted Llama",
            type="text",
            available=has_compat,
            default_model=os.getenv("STIMULUS_FORGE_COMPAT_MODEL", "meta-llama/llama-3.3-70b-instruct"),
            available_models=[
                os.getenv("STIMULUS_FORGE_COMPAT_MODEL", "meta-llama/llama-3.3-70b-instruct"),
            ],
            supports_structured_outputs=True,
            pricing_per_1m_input_usd=0.20,
            pricing_per_1m_output_usd=0.20,
            notes="Base URL configured" if has_compat else "Set STIMULUS_FORGE_COMPAT_BASE_URL",
        ),
        ProviderInfo(
            id="bing",
            name="Bing Image Search (Strict SafeSearch)",
            type="image",
            available=True,
            default_model="web-scraper-v1",
            supports_structured_outputs=False,
            pricing_per_1m_input_usd=0.0,
            pricing_per_1m_output_usd=0.0,
            notes="Zero cost. Real-world visual stimulus candidates with deduplication.",
        ),
        ProviderInfo(
            id="openai_image",
            name="OpenAI DALL-E 3 / GPT-Image",
            type="image",
            available=has_openai,
            default_model=os.getenv("STIMULUS_FORGE_IMAGE_MODEL", "dall-e-3"),
            supports_structured_outputs=False,
            pricing_per_1m_input_usd=0.0,
            pricing_per_1m_output_usd=0.0,
            notes="Approx. $0.040 per generated stimulus image",
        ),
        ProviderInfo(
            id="google_image",
            name="Google Imagen 3 (Cloud)",
            type="image",
            available=has_google,
            default_model=os.getenv("STIMULUS_FORGE_GOOGLE_IMAGE_MODEL", "imagen-3.0-generate-002"),
            supports_structured_outputs=False,
            pricing_per_1m_input_usd=0.0,
            pricing_per_1m_output_usd=0.0,
            notes="Approx. $0.030 per generated stimulus image via Google GenAI",
        ),
        ProviderInfo(
            id="local_image",
            name="Local Diffusers (Stable Diffusion)",
            type="image",
            available=local_img_available,
            default_model="runwayml/stable-diffusion-v1-5",
            supports_structured_outputs=False,
            pricing_per_1m_input_usd=0.0,
            pricing_per_1m_output_usd=0.0,
            notes=local_img_notes,
        ),
        ProviderInfo(
            id="huggingface_image",
            name="Hugging Face Serverless (Cloud)",
            type="image",
            available=has_hf,
            default_model=os.getenv("STIMULUS_FORGE_HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell"),
            available_models=[
                "black-forest-labs/FLUX.1-schnell",
                "stabilityai/stable-diffusion-3.5-large",
                "stabilityai/stable-diffusion-xl-base-1.0",
            ],
            supports_structured_outputs=False,
            pricing_per_1m_input_usd=0.0,
            pricing_per_1m_output_usd=0.0,
            notes="Zero cost (Free Tier Serverless Inference API). Set HF_TOKEN." if not has_hf else "HF token detected. Free Serverless Inference.",
        ),
    ]
