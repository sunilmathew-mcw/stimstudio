"""Optional local diffusion-model image generation."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Callable

from stimulus_forge.config import config
from stimulus_forge.services.openai_images import (
    image_name_stem,
)


LOCAL_IMAGE_NEGATIVE_PROMPT = (
    "nudity, sexual content, graphic violence, gore, text, caption, watermark, "
    "logo, collage, split screen, blurry, distorted, low quality"
)


def build_local_image_prompt(
    concept: str,
    related_concept: str = "",
    name: str = "",
    description: str = "",
) -> str:
    """Build a compact prompt that fits Stable Diffusion's CLIP token limit."""
    if related_concept and name:
        subject = f"{related_concept}, portrayed by {name}"
    else:
        subject = related_concept or name or concept
    relationship = f", associated with {concept}" if subject != concept else ""
    context = f", {description}" if description else ""
    return (
        f"safe-for-work photorealistic square image of {subject}{relationship}{context}. "
        "Single centered subject, simple natural background, even lighting, clear view, "
        "recognizable at thumbnail size."
    )


class LocalImageService:
    """Lazily load a local Diffusers-compatible checkpoint or model directory."""

    def __init__(
        self,
        model_path: str = "",
        device: str = "auto",
        pipeline: Any | None = None,
    ):
        resolved = (
            model_path
            or os.getenv("STIMULUS_FORGE_LOCAL_IMAGE_MODEL")
            or config.get("local_image_model", "")
        )
        self.model_path = resolved
        self.device = device
        self._pipeline = pipeline

    @property
    def is_configured(self) -> bool:
        return self._pipeline is not None or bool(self.model_path.strip())

    @property
    def missing_dependencies(self) -> tuple[str, ...]:
        """Return optional packages required by the local image backend."""
        if self._pipeline is not None:
            return ()
        return tuple(
            name
            for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors")
            if importlib.util.find_spec(name) is None
        )

    @property
    def is_available(self) -> bool:
        return self.is_configured and not self.missing_dependencies

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if not self.model_path.strip():
            raise RuntimeError(
                "No local image model configured. Please select an existing .safetensors checkpoint."
            )

        try:
            import torch
            from diffusers import (
                DiffusionPipeline,
                StableDiffusionPipeline,
                StableDiffusionXLPipeline,
            )
            try:
                from diffusers import StableDiffusion3Pipeline
            except ImportError:
                StableDiffusion3Pipeline = None
        except ImportError as exc:
            raise RuntimeError(
                'Local image generation requires: pip install -e ".[local-images]"'
            ) from exc

        if self.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device

        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
        model_path = Path(self.model_path).expanduser()
        if not model_path.exists() and not ("/" in self.model_path and not self.model_path.startswith(".")):
            raise RuntimeError(
                f"Local diffusion model file not found at: '{self.model_path}'. "
                "Please verify the file path or choose a different image source (Bing or OpenAI)."
            )

        filename_or_id = (model_path.name if model_path.exists() else self.model_path).casefold()
        is_sd3 = any(marker in filename_or_id for marker in ("sd3", "sd-3", "stable-diffusion-3"))
        is_sdxl = any(marker in filename_or_id for marker in ("sdxl", "ssd-1b", "xl_"))

        if is_sd3:
            if StableDiffusion3Pipeline is None:
                raise RuntimeError(
                    "StableDiffusion3Pipeline is not available in your diffusers installation. "
                    "Please update diffusers to support SD 3.5."
                )
            dtype = torch.bfloat16 if device in {"cuda", "mps"} else torch.float32

            # Check if an ungated SD 3.5 medium snapshot is cached locally
            hf_cache_snapshots = list(
                Path("~/.cache/huggingface/hub/models--ckpt--stable-diffusion-3.5-medium/snapshots")
                .expanduser()
                .glob("*")
            )
            if hf_cache_snapshots and hf_cache_snapshots[0].is_dir():
                pipeline = StableDiffusion3Pipeline.from_pretrained(
                    str(hf_cache_snapshots[0]),
                    text_encoder_3=None,
                    tokenizer_3=None,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            elif model_path.is_dir():
                pipeline = StableDiffusion3Pipeline.from_pretrained(
                    str(model_path.resolve()),
                    text_encoder_3=None,
                    tokenizer_3=None,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            else:
                repo_id = (
                    self.model_path
                    if ("/" in self.model_path and not self.model_path.startswith(".") and "stabilityai" not in self.model_path)
                    else "ckpt/stable-diffusion-3.5-medium"
                )
                pipeline = StableDiffusion3Pipeline.from_pretrained(
                    repo_id,
                    text_encoder_3=None,
                    tokenizer_3=None,
                    torch_dtype=dtype,
                )
        elif model_path.is_file():
            pipeline_class = (
                StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline
            )
            pipeline = pipeline_class.from_single_file(
                str(model_path.resolve()), torch_dtype=dtype
            )
        else:
            source = str(model_path.resolve()) if model_path.exists() else self.model_path
            pipeline = DiffusionPipeline.from_pretrained(source, torch_dtype=dtype)
        self._pipeline = pipeline.to(device)
        if hasattr(self._pipeline, "set_progress_bar_config"):
            self._pipeline.set_progress_bar_config(disable=True)
        return self._pipeline

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
        """Generate safe-prompted images and save them using workspace naming rules."""
        if limit < 1:
            return []

        pipeline = self._load_pipeline()
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stem = image_name_stem(concept, related_concept, name)
        next_index = 1
        while (directory / f"{stem}_{next_index}.png").exists():
            next_index += 1

        prompt = build_local_image_prompt(
            concept, related_concept, name, description
        )
        saved: list[Path] = []
        is_sd3 = "StableDiffusion3" in getattr(getattr(pipeline, "__class__", None), "__name__", "")
        for _ in range(limit):
            if should_continue is not None and not should_continue():
                break
            gen_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "negative_prompt": LOCAL_IMAGE_NEGATIVE_PROMPT,
                "num_images_per_prompt": 1,
            }
            if is_sd3:
                gen_kwargs["num_inference_steps"] = 28
                gen_kwargs["guidance_scale"] = 4.5

            result = pipeline(**gen_kwargs)
            images = getattr(result, "images", [])
            if not images:
                continue
            path = directory / f"{stem}_{next_index}.png"
            images[0].convert("RGB").save(path, format="PNG")
            saved.append(path)
            if on_image is not None:
                on_image(path)
            next_index += 1

        if not saved and should_continue is not None and not should_continue():
            return []
        if not saved:
            raise RuntimeError("The local image model returned no usable images.")
        return saved
