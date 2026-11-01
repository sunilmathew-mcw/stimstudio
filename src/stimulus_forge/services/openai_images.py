"""OpenAI image generation and safety checks for stimulus images."""

from __future__ import annotations

import base64
import importlib.util
from io import BytesIO
import os
from pathlib import Path
import re
from typing import Any, Callable

from PIL import Image

from stimulus_forge.services.security import sanitize_prompt_text


def image_name_stem(concept: str, related_concept: str = "", name: str = "") -> str:
    """Build a stable filename stem compatible with the existing workspace."""
    clean = lambda value: re.sub(r"[^a-zA-Z0-9]", "", value.replace(" ", "").lower())
    parts = [clean(concept)]
    if related_concept:
        parts.append(clean(related_concept))
    if name and clean(name) not in parts[1:]:
        parts.append(clean(name))
    return "~".join(part for part in parts if part) or "stimulus"


def build_stimulus_image_prompt(
    concept: str,
    related_concept: str = "",
    name: str = "",
    description: str = "",
) -> str:
    """Create a consistent prompt optimized for recognizable research stimuli."""
    safe_concept = sanitize_prompt_text(concept)
    safe_rel = sanitize_prompt_text(related_concept)
    safe_name = sanitize_prompt_text(name)
    safe_desc = sanitize_prompt_text(description, max_length=200)

    if safe_rel and safe_name:
        subject = f'"{safe_rel}" as portrayed by {safe_name}'
    else:
        subject = f'"{safe_rel or safe_name or safe_concept}"'

    context = ""
    if safe_rel or safe_name:
        context = f' It must be clearly associated with the parent concept "{safe_concept}".'
    if safe_desc:
        context += f" Relationship context: {safe_desc}."

    return (
        f"Create one safe-for-work, photorealistic square stimulus image whose unmistakable "
        f"main subject is {subject}.{context} Use a single centered subject, a simple natural "
        "background, even lighting, and a clear unobstructed view. The image must be easy to "
        "recognize at thumbnail size. No collage, split screen, border, caption, logo, watermark, "
        "written words, nudity, sexualized pose, or graphic content."
    )


class OpenAIImageService:
    """Lazy OpenAI client used for image generation and Bing-result moderation."""

    def __init__(
        self,
        model: str = "gpt-image-2",
        quality: str = "low",
        size: str = "1024x1024",
        client: Any | None = None,
        api_key: str | None = None,
    ):
        if quality not in {"low", "medium", "high", "auto"}:
            raise ValueError("Image quality must be low, medium, high, or auto.")
        self.model = model
        self.quality = quality
        self.size = size
        self._client = client
        self._api_key = (api_key or "").strip()

    @property
    def api_key(self) -> str:
        return self._api_key or os.getenv("OPENAI_API_KEY", "").strip()

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
        """Whether optional moderation can run without prompting an install error."""
        return self._client is not None or (
            bool(self.api_key)
            and importlib.util.find_spec("openai") is not None
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            effective_key = self.api_key
            if not effective_key:
                raise RuntimeError(
                    "Set OPENAI_API_KEY or enter your OpenAI API key before using AI image generation."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    'AI image generation requires: pip install -e ".[cloud]"'
                ) from exc
            self._client = OpenAI(api_key=effective_key)
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
        """Generate and save one or more safe, square PNG stimulus images."""
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
            batch_size = min(10, limit - len(saved))
            response = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                n=batch_size,
                size=self.size,
                quality=self.quality,
                moderation="auto",
            )
            batch_saved = 0
            for item in response.data:
                if should_continue is not None and not should_continue():
                    break
                encoded = getattr(item, "b64_json", None)
                if not encoded:
                    continue
                try:
                    image_bytes = base64.b64decode(encoded, validate=True)
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
                batch_saved += 1
            if batch_saved == 0:
                break

        if not saved and should_continue is not None and not should_continue():
            return []
        if not saved:
            raise RuntimeError("The image model returned no usable images.")
        return saved

    def is_sexually_safe(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> bool:
        """Reject sexual imagery using the multimodal moderation model."""
        encoded = base64.b64encode(image_bytes).decode("ascii")
        response = self.client.moderations.create(
            model="omni-moderation-latest",
            input=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            ],
        )
        result = response.results[0]
        categories = getattr(result, "categories", None)
        if categories is None:
            return not bool(getattr(result, "flagged", True))
        if hasattr(categories, "model_dump"):
            category_values = categories.model_dump(by_alias=True)
        elif isinstance(categories, dict):
            category_values = categories
        else:
            category_values = vars(categories)
        return not any(
            bool(category_values.get(key))
            for key in ("sexual", "sexual/minors", "sexual_minors")
        )
