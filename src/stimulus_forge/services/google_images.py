"""Google Imagen image generation service for stimulus images."""

from __future__ import annotations

import importlib.util
from io import BytesIO
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from stimulus_forge.services.openai_images import (
    build_stimulus_image_prompt,
    image_name_stem,
)


class GoogleImageService:
    """Lazy Google GenAI client used for Imagen image generation."""

    def __init__(
        self,
        model: str = "imagen-3.0-generate-002",
        client: Any | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self._client = client
        self._api_key = (api_key or "").strip()

    @property
    def api_key(self) -> str:
        return (
            self._api_key
            or os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
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
        """Whether image generation can run without missing key or dependency."""
        return self._client is not None or (
            bool(self.api_key)
            and importlib.util.find_spec("google.genai") is not None
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            effective_key = self.api_key
            if not effective_key:
                raise RuntimeError(
                    "Set GEMINI_API_KEY / GOOGLE_API_KEY or enter your Google API key before using Google image generation."
                )
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    'Google image generation requires: pip install -e ".[cloud]"'
                ) from exc
            self._client = genai.Client(api_key=effective_key)
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
        """Generate and save one or more safe square stimulus images with Google Imagen."""
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

        while len(saved) < limit:
            if should_continue is not None and not should_continue():
                break
            batch_size = min(4, limit - len(saved))

            try:
                response = self.client.models.generate_images(
                    model=self.model,
                    prompt=prompt,
                    config={
                        "number_of_images": batch_size,
                        "output_mime_type": "image/png",
                        "aspect_ratio": "1:1",
                        "person_generation": "ALLOW_ADULT",
                    },
                )
            except Exception as exc:
                if saved:
                    break
                raise RuntimeError(f"Google Imagen generation failed: {exc}") from exc

            generated_images = getattr(response, "generated_images", []) or []
            if not generated_images:
                break

            for item in generated_images:
                if should_continue is not None and not should_continue():
                    break
                img_obj = getattr(item, "image", None)
                if img_obj is None:
                    continue
                image_bytes = getattr(img_obj, "image_bytes", None)
                if not image_bytes:
                    continue
                try:
                    with Image.open(BytesIO(image_bytes)) as generated_image:
                        generated_image.verify()
                except Exception:
                    continue
                path = directory / f"{stem}_{next_index}.png"
                path.write_bytes(image_bytes)
                saved.append(path)
                if on_image is not None:
                    on_image(path)
                next_index += 1

        if not saved and (should_continue is None or should_continue()):
            raise RuntimeError("The Google Imagen model returned no usable images.")
        return saved
