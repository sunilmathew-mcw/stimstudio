# Author: Sunil Mathew
# Date: 12 Nov 2024
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
from typing import Any
try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
from PIL import Image

from stimulus_forge.config import config
from stimulus_forge.services.bing_image_search import BingDownloader, build_bing_query

_FACE_DETECTOR_CACHE = None


def _get_face_detector():
    global _FACE_DETECTOR_CACHE
    if _FACE_DETECTOR_CACHE is None:
        try:
            from mtcnn import MTCNN
            _FACE_DETECTOR_CACHE = MTCNN()
        except Exception:
            _FACE_DETECTOR_CACHE = False
    return _FACE_DETECTOR_CACHE


MAX_DETECTED_FACES = 10
DEFAULT_MIN_FACE_PIXELS = 80
MIN_FACE_IMAGE_RATIO = 0.08
HEAD_EXPANSION_FACTOR = 1.6
MIN_HEAD_IMAGE_AREA_RATIO = 1.0 / 25.0  # Expanded head box must be >= 1/25th (4%) of reference image area
FOREGROUND_AREA_RATIO = 0.35


def _box_iou(a: dict, b: dict) -> float:
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _filter_and_rank_faces(
    candidates: list[dict],
    img_width: int,
    img_height: int,
    max_faces: int = MAX_DETECTED_FACES,
    min_face_size: int = DEFAULT_MIN_FACE_PIXELS,
) -> list[dict]:
    """Filter out small/background crowd faces (<80x80 or <1/25th image scale), deduplicate overlaps, and cap at max_faces."""
    if not candidates:
        return []

    max_cap = max(1, min(int(max_faces or MAX_DETECTED_FACES), MAX_DETECTED_FACES))
    min_px = max(16, int(min_face_size or DEFAULT_MIN_FACE_PIXELS))
    short_dim = max(1, min(img_width, img_height))
    # Enforce both hard pixel floor (default 80x80) and relative edge floor (8% of shorter image edge)
    effective_min_edge = max(min_px, int(short_dim * MIN_FACE_IMAGE_RATIO))
    # Enforce 1/25th area constraint on the full head region ((1.6*w)*(1.6*h) >= (short_dim^2)/25)
    min_face_area = max(
        min_px * min_px,
        int(((short_dim * short_dim) * MIN_HEAD_IMAGE_AREA_RATIO) / (HEAD_EXPANSION_FACTOR ** 2)),
    )

    valid: list[dict] = []
    for f in candidates:
        w = int(f.get("w", 0))
        h = int(f.get("h", 0))
        if w < effective_min_edge or h < effective_min_edge:
            continue
        if (w * h) < min_face_area:
            continue
        aspect = w / h if h > 0 else 0.0
        if aspect < 0.55 or aspect > 1.55:
            continue
        valid.append(f)

    if not valid:
        return []

    # Foreground dominance filter: reject background crowd faces that are much smaller
    # than the primary foreground subject face(s) in the scene.
    max_area = max(f["w"] * f["h"] for f in valid)
    min_foreground_area = max_area * FOREGROUND_AREA_RATIO
    foreground = [f for f in valid if (f["w"] * f["h"]) >= min_foreground_area]

    # Rank by prominence (area weighted by confidence and slight centrality bonus)
    cx_img = img_width / 2.0
    cy_img = img_height / 2.0
    diag = max(1.0, (img_width ** 2 + img_height ** 2) ** 0.5)

    def _salience(f: dict) -> float:
        area = float(f["w"] * f["h"])
        conf = float(f.get("confidence", 0.90))
        fcx = f["x"] + f["w"] / 2.0
        fcy = f["y"] + f["h"] / 2.0
        dist_norm = (((fcx - cx_img) ** 2 + (fcy - cy_img) ** 2) ** 0.5) / diag
        centrality_bonus = 1.0 + 0.15 * (1.0 - min(1.0, dist_norm))
        return area * conf * centrality_bonus

    foreground.sort(key=_salience, reverse=True)

    # Non-maximum suppression (IoU > 0.4) to avoid duplicate boxes on the same face
    deduped: list[dict] = []
    for f in foreground:
        if any(_box_iou(f, kept) > 0.4 for kept in deduped):
            continue
        deduped.append(f)
        if len(deduped) >= max_cap:
            break

    return deduped


def compute_face_crop_box(
    img_width: int,
    img_height: int,
    face: dict[str, Any],
    target_size: tuple[int, int] = (160, 160),
    expansion_factor: float = 0.60,
) -> tuple[int, int, int, int]:
    """Compute crop bounding box (crop_x1, crop_y1, crop_x2, crop_y2) centered on the face.

    Mirrors the native desktop Stimulus Forge cropping algorithm:
    1. Centers the crop window on the detected face center.
    2. Expands the crop window by 60% (+0.6 * max(w, h)) to comfortably frame the head,
       hair, chin, and neck without awkward edge cutoffs.
    3. Enforces the target aspect ratio (square 1:1 by default) so resizing never distorts,
       stretches, or squashes the face.
    4. Shifts the window within image bounds if near borders to preserve full dimensions.
    """
    x = int(face.get("x", 0))
    y = int(face.get("y", 0))
    w = int(face.get("w", 0))
    h = int(face.get("h", 0))

    if w <= 0 or h <= 0 or img_width <= 0 or img_height <= 0:
        return 0, 0, max(1, img_width), max(1, img_height)

    center_x = x + w / 2.0
    center_y = y + h / 2.0

    target_w = max(16, int(target_size[0]))
    target_h = max(16, int(target_size[1]))
    target_aspect = target_w / target_h

    # Base expansion matching desktop: 60% padding around max(w, h)
    face_max = max(w, h)
    base_dim = int(round(face_max * (1.0 + expansion_factor)))

    if abs(target_aspect - 1.0) < 1e-4:
        # Default square crop (as in native desktop client)
        crop_w = base_dim
        crop_h = base_dim
        max_sq = min(img_width, img_height)
        if crop_w > max_sq:
            crop_w = max_sq
            crop_h = max_sq
    elif target_aspect > 1.0:
        crop_h = base_dim
        crop_w = int(round(crop_h * target_aspect))
        if crop_w > img_width:
            crop_w = img_width
            crop_h = int(round(crop_w / target_aspect))
        if crop_h > img_height:
            crop_h = img_height
            crop_w = int(round(crop_h * target_aspect))
    else:
        crop_w = base_dim
        crop_h = int(round(crop_w / target_aspect))
        if crop_h > img_height:
            crop_h = img_height
            crop_w = int(round(crop_h * target_aspect))
        if crop_w > img_width:
            crop_w = img_width
            crop_h = int(round(crop_w / target_aspect))

    crop_w = max(1, min(crop_w, img_width))
    crop_h = max(1, min(crop_h, img_height))

    # Center crop box around face center
    crop_x1 = int(round(center_x - crop_w / 2.0))
    crop_y1 = int(round(center_y - crop_h / 2.0))
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    # Boundary shifting (exact desktop logic: shift away from edges while keeping crop_w x crop_h)
    if crop_x1 < 0:
        shift = -crop_x1
        crop_x1 = 0
        crop_x2 = min(img_width, crop_x2 + shift)
    elif crop_x2 > img_width:
        shift = crop_x2 - img_width
        crop_x2 = img_width
        crop_x1 = max(0, crop_x1 - shift)

    if crop_y1 < 0:
        shift = -crop_y1
        crop_y1 = 0
        crop_y2 = min(img_height, crop_y2 + shift)
    elif crop_y2 > img_height:
        shift = crop_y2 - img_height
        crop_y2 = img_height
        crop_y1 = max(0, crop_y1 - shift)

    return crop_x1, crop_y1, crop_x2, crop_y2


def detect_face_rois(
    image_path: str,
    max_faces: int = MAX_DETECTED_FACES,
    min_face_size: int = DEFAULT_MIN_FACE_PIXELS,
    target_size: tuple[int, int] = (160, 160),
) -> dict:
    """Detect prominent frontal faces in an image (max 10) and filter out background crowds."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        pixels = np.array(rgb)

    min_px = max(16, int(min_face_size or DEFAULT_MIN_FACE_PIXELS))
    effective_min_edge = max(min_px, int(min(width, height) * MIN_FACE_IMAGE_RATIO))

    raw_faces: list[dict] = []
    detector_used = "none"
    detector = _get_face_detector()
    if detector:
        try:
            results = detector.detect_faces(pixels)
            detector_used = "mtcnn"
            for r in results:
                box = r.get("box", [])
                conf = float(r.get("confidence", 0.0))
                if len(box) == 4 and conf >= 0.90:
                    x, y, w, h = [int(v) for v in box]
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)
                    if w < effective_min_edge or h < effective_min_edge:
                        continue
                    keypoints = r.get("keypoints") or {}
                    left_eye = keypoints.get("left_eye")
                    right_eye = keypoints.get("right_eye")
                    if left_eye and right_eye:
                        eye_dx = abs(float(right_eye[0]) - float(left_eye[0]))
                        if eye_dx < w * 0.20:
                            continue
                    raw_faces.append({
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "confidence": round(conf, 4),
                    })
        except Exception as e:
            print(f"[Face Detection] MTCNN error: {e}")

    if detector_used == "none" and hasattr(cv2, "CascadeClassifier"):
        try:
            cascade_path = os.path.join(
                os.path.dirname(__file__), "..", "assets", "haarcascade_frontalface_default.xml"
            )
            if os.path.exists(cascade_path):
                cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
                detected = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(effective_min_edge, effective_min_edge),
                )
                detector_used = "haar"
                for (x, y, w, h) in detected:
                    raw_faces.append({
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "confidence": 0.90,
                    })
        except Exception as e:
            print(f"[Face Detection] Cascade error: {e}")

    faces = _filter_and_rank_faces(
        raw_faces,
        img_width=width,
        img_height=height,
        max_faces=max_faces,
        min_face_size=min_px,
    )

    for f in faces:
        crop_x1, crop_y1, crop_x2, crop_y2 = compute_face_crop_box(
            img_width=width,
            img_height=height,
            face=f,
            target_size=target_size,
        )
        f["crop_box"] = {
            "x": crop_x1,
            "y": crop_y1,
            "w": crop_x2 - crop_x1,
            "h": crop_y2 - crop_y1,
        }

    return {
        "width": width,
        "height": height,
        "faces": faces,
        "count": len(faces),
        "face_detected": len(faces) > 0,
        "detector": detector_used,
        "path": image_path,
    }


class TaskImagesDownloader:
    """
    TaskImagesDownloader class to download images from Bing Image Search
    """
    def __init__(self):
        self.task_images_path = config['task_images_folder']

    def download_images(
        self,
        output_dir,
        names,
        filter="",
        limit=5,
        force_replace=False,
        verbose=True,
        image_validator=None,
        on_image=None,
        should_continue=None,
        seen_images=None,
        seen_images_lock=None,
        safe_query_exclusions=None,
        min_width=160,
        min_height=160,
        contexts=None,
    ):
        """
        Download images from Bing Image Search

        Args:
        	names: List of names
        """
        if force_replace and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        multi_thread = True
        ctx_map = contexts or {}
        if multi_thread:
            with ThreadPoolExecutor(max_workers=None) as executor:
                futures = {
                    executor.submit(
                        self.download,
                        name,
                        limit=limit,
                        output_dir=output_dir,
                        filter=filter,
                        safe_search="strict",
                        safe_query_exclusions=safe_query_exclusions,
                        image_validator=image_validator,
                        on_image=on_image,
                        should_continue=should_continue,
                        seen_images=seen_images,
                        seen_images_lock=seen_images_lock,
                        force_replace=False,
                        timeout=60,
                        verbose=verbose,
                        min_width=min_width,
                        min_height=min_height,
                        context=ctx_map.get(name, ""),
                    ): name
                    for name in names
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[Error] Failed to download {name}: {e}")
        else:
            for name in names:
                self.download(
                    name,
                    limit=limit,
                    output_dir=output_dir,
                    filter=filter,
                    safe_search="strict",
                    safe_query_exclusions=safe_query_exclusions,
                    image_validator=image_validator,
                    on_image=on_image,
                    should_continue=should_continue,
                    seen_images=seen_images,
                    seen_images_lock=seen_images_lock,
                    force_replace=False,
                    timeout=60,
                    verbose=verbose,
                    min_width=min_width,
                    min_height=min_height,
                    context=ctx_map.get(name, ""),
                )

    def download(
        self,
        concept,
        related_concept="",
        name="",
        limit=100,
        output_dir="dataset",
        safe_search="strict",
        safe_query_exclusions=None,
        image_validator=None,
        on_image=None,
        should_continue=None,
        seen_images=None,
        seen_images_lock=None,
        force_replace=False,
        timeout=10,
        filter="",
        verbose=True,
        provider="bing",
        min_width=160,
        min_height=160,
        context="",
    ):
        query = build_bing_query(concept, related_concept, name, context=context)

        if force_replace:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print('[Error]Failed to create directory.', e)
            
        if verbose:
            print(f"Downloading Images to {output_dir}")
            print(f"Image search query: {query} (provider={provider})")
        bing = BingDownloader(
            query=query,
            concept=concept,
            related_concept=related_concept,
            name=name,
            limit=limit,
            output_dir=output_dir,
            safe_search=safe_search,
            safe_query_exclusions=safe_query_exclusions,
            image_validator=image_validator,
            on_image=on_image,
            should_continue=should_continue,
            seen_images=seen_images,
            seen_images_lock=seen_images_lock,
            timeout=timeout,
            filter=filter,
            verbose=verbose,
            provider=provider,
            min_width=min_width,
            min_height=min_height,
            context=context,
        )
        return bing.run()
    
    def detect_faces(
        self,
        image_path: str,
        max_faces: int = MAX_DETECTED_FACES,
        min_face_size: int = DEFAULT_MIN_FACE_PIXELS,
        target_size: tuple[int, int] = (160, 160),
    ) -> dict:
        """Detect frontal faces in the given image file and return bounding boxes."""
        return detect_face_rois(
            image_path,
            max_faces=max_faces,
            min_face_size=min_face_size,
            target_size=target_size,
        )

    def crop_face(
        self,
        image_path: str,
        target_size: tuple[int, int] = (160, 160),
        max_faces: int = MAX_DETECTED_FACES,
        min_face_size: int = DEFAULT_MIN_FACE_PIXELS,
        face_box: dict[str, Any] | None = None,
    ) -> bool:
        """Crop primary detected face from an image and save normalized to target_size (width, height).

        Mirrors native desktop Stimulus Forge behavior:
        - Locks crop aspect ratio to target_size (1:1 square by default) to prevent stretching.
        - Expands 60% around the face to frame head, hair, chin, and neck naturally.
        - Clamps and shifts within image borders without distorting aspect ratio.
        """
        primary = face_box
        if not primary:
            detection = self.detect_faces(
                image_path,
                max_faces=max_faces,
                min_face_size=min_face_size,
                target_size=target_size,
            )
            faces = detection.get("faces", [])
            if not faces:
                return False
            primary = max(faces, key=lambda f: f["w"] * f["h"])

        target_w = max(16, int(target_size[0]))
        target_h = max(16, int(target_size[1]))

        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            crop_x1, crop_y1, crop_x2, crop_y2 = compute_face_crop_box(
                img_width=rgb.width,
                img_height=rgb.height,
                face=primary,
                target_size=(target_w, target_h),
            )
            cropped = rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            resample_filter = getattr(Image, "Resampling", Image).LANCZOS
            resized = cropped.resize((target_w, target_h), resample_filter)
            resized.save(image_path, format="PNG")
        return True


if __name__ == "__main__":
    names = ["obama", "prince", "marquette"]
    task = TaskImagesDownloader()
    task.download_images(names=names, output_dir="dataset", force_replace=True, verbose=True)
