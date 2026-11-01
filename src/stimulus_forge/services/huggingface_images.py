"""Hugging Face Serverless Inference API image generation service for stimulus images."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from stimulus_forge.services.openai_images import (
    build_stimulus_image_prompt,
    image_name_stem,
)

DEFAULT_HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"


class HuggingFaceImageService:
    """Client for Hugging Face Serverless Inference API image generation."""

    def __init__(
        self,
        model: str = DEFAULT_HF_IMAGE_MODEL,
        client: Any | None = None,
        api_key: str | None = None,
    ):
        self.model = model or DEFAULT_HF_IMAGE_MODEL
        self._client = client
        self._api_key = (api_key or "").strip()

    @property
    def api_key(self) -> str:
        return (
            self._api_key
            or os.getenv("HF_TOKEN", "").strip()
            or os.getenv("HUGGINGFACE_API_KEY", "").strip()
            or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
            or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        )

    @api_key.setter
    def api_key(self, value: str | None) -> None:
        cleaned = (value or "").strip()
        if cleaned != getattr(self, "_api_key", ""):
            self._api_key = cleaned
            self._client = None

    @property
    def is_configured(self) -> bool:
        return self._client is not None or bool(self.api_key)

    @property
    def is_available(self) -> bool:
        """Whether generation can run without missing token or package."""
        return self._client is not None or (
            bool(self.api_key)
            and importlib.util.find_spec("huggingface_hub") is not None
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            effective_key = self.api_key
            if not effective_key:
                raise RuntimeError(
                    "Set HF_TOKEN or enter your Hugging Face API token before using Hugging Face image generation."
                )
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise RuntimeError(
                    'Hugging Face inference requires huggingface_hub: pip install huggingface_hub'
                ) from exc
            self._client = InferenceClient(api_key=effective_key)
        return self._client

    def generate(
        self,
        concept: str,
        output_dir: str | Path,
        limit: int,
        related_concept: str = "",
        name: str = "",
        description: str = "",
        on_image: Callable[[Path], None] | None = None,
        should_continue: Callable[[], bool] | None = None,
    ) -> list[Path]:
        """Generate and save one or more safe square stimulus images via Hugging Face Serverless API."""
        if limit < 1:
            return []

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stem = image_name_stem(concept, related_concept, name)
        next_index = 1
        while (directory / f"{stem}_{next_index}.png").exists():
            next_index += 1
        saved: list[Path] = []
        prompt = build_stimulus_image_prompt(
            concept, related_concept, name, description
        )

        for _ in range(limit):
            if should_continue is not None and not should_continue():
                break

            try:
                result = self.client.text_to_image(
                    prompt=prompt,
                    model=self.model,
                )
            except Exception as exc:
                if saved:
                    break
                raise RuntimeError(f"Hugging Face image generation failed: {exc}") from exc

            if not isinstance(result, Image.Image):
                continue

            path = directory / f"{stem}_{next_index}.png"
            result.convert("RGB").save(path, format="PNG")
            saved.append(path)
            if on_image is not None:
                on_image(path)
            next_index += 1

        if not saved and (should_continue is None or should_continue()):
            raise RuntimeError("The Hugging Face model returned no usable images.")
        return saved
