"""Text stimulus generation service: creates high-contrast images with centered concept text."""

from __future__ import annotations

import os
from pathlib import Path
import re
import textwrap
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from stimulus_forge.services.openai_images import image_name_stem


def _get_font(size: int, font_name: str | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Attempt to load a clean sans-serif font at the requested size, falling back to default."""
    candidates = []
    if font_name:
        candidates.append(font_name)
    candidates.extend([
        "Helvetica",
        "Helvetica.ttc",
        "Arial",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "FreeSans.ttf",
        "Verdana",
        "Verdana.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ])
    for cand in candidates:
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def render_centered_text_image(
    text: str,
    size: tuple[int, int] = (512, 512),
    bg_color: str = "#FFFFFF",
    text_color: str = "#000000",
    font_name: str | None = None,
    font_size: int | None = None,
) -> Image.Image:
    """Render an image of `size` with `text` horizontally and vertically centered."""
    width, height = max(16, size[0]), max(16, size[1])
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    clean_text = str(text).strip() if text else "Stimulus"
    if not clean_text:
        clean_text = "Stimulus"

    # Target maximum bounding box inside the image (with 15% margin on all sides)
    max_w = width * 0.78
    max_h = height * 0.75

    if font_size and font_size > 0:
        cur_size = font_size
        words = clean_text.split()
        if len(words) > 2 and len(clean_text) > 20:
            lines = textwrap.wrap(clean_text, width=max(12, int(len(clean_text) / 2)))
        else:
            lines = [clean_text]
        wrapped_text = "\n".join(lines)
        font = _get_font(cur_size, font_name)
    else:
        # Dynamically calculate font size and line wrapping
        cur_size = max(16, int(height * 0.14))
        min_size = 14
        while cur_size >= min_size:
            font = _get_font(cur_size, font_name)
            words = clean_text.split()
            if len(words) > 2 and len(clean_text) > 18:
                wrap_w = max(10, int(len(clean_text) / (2 if len(words) <= 4 else 3)))
                lines = textwrap.wrap(clean_text, width=wrap_w)
            else:
                lines = [clean_text]
            wrapped_text = "\n".join(lines)

            bbox = draw.textbbox((0, 0), wrapped_text, font=font, align="center")
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if tw <= max_w and th <= max_h:
                break
            cur_size -= 2

        font = _get_font(cur_size, font_name)

    bbox = draw.textbbox((0, 0), wrapped_text, font=font, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = (width - tw) / 2.0 - bbox[0]
    y = (height - th) / 2.0 - bbox[1]

    draw.text((x, y), wrapped_text, font=font, fill=text_color, align="center")
    return img


class TextImageService:
    """Generates synthetic visual stimulus images with the concept text centered."""

    def __init__(
        self,
        font_name: str | None = None,
        bg_color: str = "#FFFFFF",
        text_color: str = "#000000",
        target_size: tuple[int, int] = (512, 512),
        font_size: int | None = None,
    ):
        self.font_name = font_name
        self.bg_color = bg_color or "#FFFFFF"
        self.text_color = text_color or "#000000"
        self.target_size = target_size
        self.font_size = font_size

    @property
    def is_configured(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return True

    def generate(
        self,
        concept: str,
        output_dir: str | Path,
        limit: int = 1,
        related_concept: str = "",
        name: str = "",
        description: str = "",
        target_size: tuple[int, int] | None = None,
        bg_color: str | None = None,
        text_color: str | None = None,
        on_image: Callable[[Path], None] | None = None,
        should_continue: Callable[[], bool] | None = None,
    ) -> list[Path]:
        """Generate and save PNG stimulus images featuring centered concept text."""
        if limit < 1:
            return []

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)

        stem = image_name_stem(concept, related_concept, name)
        size = target_size or self.target_size or (512, 512)
        base_bg = bg_color or self.bg_color
        base_fg = text_color or self.text_color

        # Determine centered display text: priority is related_concept, then name, then concept
        target_text = (
            related_concept.strip()
            if related_concept and related_concept.strip()
            else (name.strip() if name and name.strip() else concept.strip())
        )
        if not target_text:
            target_text = "Stimulus"

        # Multi-stimulus variations when limit > 1 (e.g. contrast inversion for psychophysics)
        style_variations = [
            (base_bg, base_fg, False),                 # Default
            ("#000000", "#FFFFFF", False),             # Inverted (White on Black)
            ("#E0E0E0", "#111111", False),             # Light neutral gray
            ("#1E1E1E", "#F0F0F0", False),             # Dark slate
            (base_bg, base_fg, True),                  # Uppercase variant
        ]

        # Find next available index in folder
        next_index = 1
        while (directory / f"{stem}_{next_index}.png").exists():
            next_index += 1

        saved: list[Path] = []
        for i in range(limit):
            if should_continue is not None and not should_continue():
                break

            if i < len(style_variations) and base_bg == "#FFFFFF" and base_fg == "#000000":
                bg, fg, uppercase = style_variations[i]
            else:
                bg, fg, uppercase = base_bg, base_fg, False

            render_text = target_text.upper() if uppercase else target_text
            img = render_centered_text_image(
                text=render_text,
                size=size,
                bg_color=bg,
                text_color=fg,
                font_name=self.font_name,
                font_size=self.font_size,
            )

            path = directory / f"{stem}_{next_index}.png"
            next_index += 1
            img.save(path, format="PNG")
            saved.append(path)

            if on_image is not None:
                try:
                    on_image(path)
                except Exception:
                    pass

        return saved
