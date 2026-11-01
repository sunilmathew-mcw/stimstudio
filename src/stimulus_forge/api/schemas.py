"""Pydantic V2 schemas for the Stimulus Forge API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class ConceptItem(BaseModel):
    """A concrete visual association for a cognitive stimulus."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Canonical name of the entity, object, or person")
    char: str = Field(
        default="",
        description="Character or role name when the entity is a performer; otherwise empty string",
    )
    description: str = Field(
        default="",
        description="Concise factual description of the visual relationship",
    )


class ConceptGenerateRequest(BaseModel):
    """Request payload for generating related concepts."""

    model_config = ConfigDict(extra="forbid")

    concept: str = Field(min_length=1, max_length=200, description="Seed concept")
    count: int = Field(default=5, ge=1, le=50, description="Target number of concepts")
    context: str | None = Field(default=None, description="Optional category or document source excerpt to disambiguate the concept")
    provider: Literal["local", "openai", "anthropic", "google", "compatible"] = Field(
        default="local",
        description="Inference provider to execute the request",
    )
    model: str | None = Field(default=None, description="Explicit model identifier override")
    local_model_path: str | None = Field(
        default=None,
        description="Absolute or relative path to local GGUF model file on host",
    )
    api_key: str | None = Field(
        default=None,
        description="Optional session API key (never persisted to disk)",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for OpenAI-compatible endpoint",
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to check and update the semantic concept cache",
    )


class ConceptGenerateResponse(BaseModel):
    """Response payload containing generated concepts and execution metadata."""

    concept: str
    provider: str
    model: str
    concepts: list[ConceptItem]
    cached: bool = False
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


class BatchConceptRequest(BaseModel):
    """Request payload for batch concept generation across multiple seeds."""

    concepts: list[str] = Field(min_length=1, max_length=100)
    count: int = Field(default=10, ge=1, le=25)
    provider: Literal["local", "openai", "anthropic", "google", "compatible"] = "local"
    model: str | None = None
    local_model_path: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    use_cache: bool = True


class BatchConceptResponse(BaseModel):
    """Response payload for multi-concept batch generation."""

    total_requested: int
    successful: int
    results: dict[str, list[ConceptItem]]
    errors: dict[str, str] = {}
    total_latency_ms: float
    total_estimated_cost_usd: float
    total_tokens: int


class ImageSearchRequest(BaseModel):
    """Request parameters for Bing SafeSearch image downloads."""

    concept: str
    related_concept: str | None = ""
    name: str | None = ""
    context: str | None = Field(default="", description="Optional category or document source excerpt to disambiguate the concept")
    limit: int = Field(default=5, ge=1, le=30)
    provider: Literal["bing", "wikimedia", "duckduckgo", "openverse"] = "bing"
    safe_search: Literal["strict", "moderate", "off"] = "strict"
    filter: str = ""
    safe_query_exclusions: str | None = Field(
        default=None,
        description="Negative keyword exclusions string (e.g. '-porn -xxx -nude')",
    )
    min_width: int = Field(default=160, ge=16, le=4096, description="Minimum image width in pixels")
    min_height: int = Field(default=160, ge=16, le=4096, description="Minimum image height in pixels")
    detect_faces: bool = Field(default=False, description="Run face detection on downloaded images")
    max_faces: int = Field(default=5, ge=1, le=10, description="Maximum faces to detect per image")
    crop_faces: bool = Field(default=False, description="Auto-crop primary detected face to target_width x target_height")
    target_width: int = Field(default=160, ge=16, le=4096, description="Target crop width in pixels")
    target_height: int = Field(default=160, ge=16, le=4096, description="Target crop height in pixels")
    output_dir: str | None = Field(default=None, description="Custom directory where images will be downloaded")
    subject_id: str | None = Field(
        default=None,
        description="Subject or patient ID (e.g. SUBJ-07) used to create storage directory",
    )


class ImageGenerateRequest(BaseModel):
    """Request parameters for text-to-image synthesis."""

    prompt: str = Field(default="", max_length=1000)
    concept: str = ""
    related_concept: str = ""
    name: str = ""
    description: str = ""
    context: str | None = Field(default="", description="Optional category or document source excerpt to disambiguate the concept")
    count: int = Field(default=1, ge=1, le=30)
    provider: Literal["openai", "google", "local", "huggingface", "text"] = "openai"
    api_key: str | None = None
    model: str = "dall-e-3"
    local_model_path: str | None = None
    quality: Literal["standard", "hd", "low", "medium", "high"] = "standard"
    size: Literal["1024x1024", "512x512", "1792x1024", "1024x1792"] = "1024x1024"
    text_bg_color: str = Field(default="#FFFFFF", description="Background color for text stimulus")
    text_color: str = Field(default="#000000", description="Text color for text stimulus")
    text_font_size: int | None = Field(default=None, description="Font size for text stimulus (auto-fit if None)")
    detect_faces: bool = Field(default=False, description="Run face detection on generated images")
    max_faces: int = Field(default=5, ge=1, le=10, description="Maximum faces to detect per image")
    crop_faces: bool = Field(default=False, description="Auto-crop primary detected face to target_width x target_height")
    target_width: int = Field(default=160, ge=16, le=4096, description="Target crop width in pixels")
    target_height: int = Field(default=160, ge=16, le=4096, description="Target crop height in pixels")
    output_dir: str | None = Field(default=None, description="Custom directory where generated images will be stored")
    subject_id: str | None = Field(
        default=None,
        description="Subject or patient ID (e.g. SUBJ-07) used to create storage directory",
    )


class ImageItem(BaseModel):
    """Metadata for a downloaded or generated stimulus image."""

    id: str
    filename: str
    url: str | None = None
    local_path: str | None = None
    concept: str
    related_concept: str = ""
    source: str
    moderated: bool = True
    face_detected: bool = False


class ImageCollectionResponse(BaseModel):
    """Response payload for image acquisition."""

    concept: str
    related_concept: str = ""
    images: list[ImageItem]
    total_count: int
    latency_ms: float


class PipelineBatchRequest(BaseModel):
    """End-to-end task stimulus pipeline request."""

    concepts: list[str] = Field(min_length=1, max_length=50)
    concepts_per_seed: int = Field(default=5, ge=1, le=15)
    images_per_concept: int = Field(default=3, ge=1, le=10)
    text_provider: Literal["local", "openai", "anthropic", "google", "compatible"] = "local"
    text_model: str | None = None
    local_model_path: str | None = None
    image_source: Literal["bing", "wikimedia", "duckduckgo", "openverse", "openai", "google", "local", "huggingface", "text"] = "bing"
    detect_faces: bool = False
    max_faces: int = Field(default=5, ge=1, le=10, description="Maximum faces to detect per image")
    crop_faces: bool = False
    target_width: int = Field(default=160, ge=16, le=4096)
    target_height: int = Field(default=160, ge=16, le=4096)
    safe_search: str = "strict"
    output_dir: str | None = Field(default=None, description="Custom directory where task images will be stored")
    subject_id: str | None = Field(
        default=None,
        description="Subject or patient ID (e.g. SUBJ-07) used to create storage directory",
    )


class PipelineJobStatus(BaseModel):
    """Current execution status of an async pipeline task."""

    job_id: str
    status: Literal["queued", "processing", "completed", "failed", "cancelled"]
    progress_percent: float = 0.0
    current_step: str = ""
    total_concepts: int = 0
    completed_concepts: int = 0
    total_images_generated: int = 0
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    summary: dict[str, Any] = {}


class ProviderInfo(BaseModel):
    """Runtime availability and configuration for an AI model provider."""

    id: str
    name: str
    type: Literal["text", "image", "multimodal"]
    available: bool
    default_model: str
    available_models: list[str] = Field(default_factory=list)
    supports_structured_outputs: bool
    pricing_per_1m_input_usd: float = 0.0
    pricing_per_1m_output_usd: float = 0.0
    notes: str = ""


class TelemetryMetric(BaseModel):
    """Record of a single model inference transaction."""

    timestamp: datetime
    provider: str
    model: str
    task_type: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    cached: bool = False


class TelemetrySummary(BaseModel):
    """Aggregated FinOps and performance metrics."""

    total_requests: int
    cache_hits: int
    cache_misses: int
    cache_hit_ratio: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_usd: float
    total_cost_avoided_usd: float
    avg_latency_ms: float
    p95_latency_ms: float
    requests_by_provider: dict[str, int]
    cost_by_provider: dict[str, float]


class ScannedModelInfo(BaseModel):
    """Discovered local model file."""

    name: str
    path: str
    size_mb: float
    last_modified: str


class ChatChoice(BaseModel):
    """Interactive button presented in the conversational assistant."""

    label: str
    action: str
    payload: dict[str, Any] = {}


class ChatMessageRequest(BaseModel):
    """Incoming user message sent to the workflow assistant."""

    message: str
    state: str = "awaiting_concept"
    current_concept: str = ""
    current_related: str = ""
    pending_image_source: str = "bing"
    workspace_concepts: list[str] = []


class ChatMessageResponse(BaseModel):
    """Conversational response from the workflow assistant."""

    speaker: str = "Assistant"
    reply: str
    state: str
    current_concept: str = ""
    current_related: str = ""
    pending_image_source: str = ""
    choices: list[ChatChoice] = []
    ui_action: str | None = None
    ui_payload: dict[str, Any] = {}
