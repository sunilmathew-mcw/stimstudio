"""Command-line interface (CLI) for headless stimulus generation.

Designed for HPC clusters, multi-GPU servers, and batch pipelines.
Supports local GGUF inference (e.g. 27B parameter models on GPUs with llama-cpp),
OpenAI-compatible inference engines (e.g. vLLM, Ollama, llama-server on compute nodes),
and cloud providers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from stimulus_forge.config import config
from stimulus_forge.services.concept_generation import (
    DEFAULT_CONCEPT_COUNT,
    RelatedConceptGenerator,
)
from stimulus_forge.services.concept_loader import load_concepts


def check_gpu_environment() -> dict[str, Any]:
    """Inspect local hardware, PyTorch CUDA capability, and llama.cpp GPU acceleration."""
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "torch_available": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "llama_cpp_available": False,
        "llama_gpu_offload": False,
    }

    try:
        import torch

        info["torch_available"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            count = torch.cuda.device_count()
            info["device_count"] = count
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except ImportError:
        pass

    try:
        import llama_cpp

        info["llama_cpp_available"] = True
        if hasattr(llama_cpp, "llama_supports_gpu_offload"):
            info["llama_gpu_offload"] = bool(llama_cpp.llama_supports_gpu_offload())
    except ImportError:
        pass

    return info


def print_gpu_diagnostic(info: dict[str, Any]) -> None:
    """Print human-readable diagnostic report for HPC cluster environments."""
    print("=" * 65)
    print(" StimStudio HPC & GPU Acceleration Diagnostic")
    print("=" * 65)
    print(f" Python Version        : {info['python_version']}")
    print(f" PyTorch Installed     : {'Yes' if info['torch_available'] else 'No'}")
    print(f" CUDA Available (PyTorch): {'Yes' if info['cuda_available'] else 'No'}")

    if info["cuda_available"]:
        print(f" Detected GPU Count    : {info['device_count']}")
        for dev in info["devices"]:
            print(f"   - [GPU {dev['id']}] {dev['name']} ({dev['total_memory_gb']} GB VRAM, CC {dev['compute_capability']})")
    else:
        print("   (No CUDA GPUs detected via PyTorch)")

    print(f" llama-cpp-python      : {'Installed' if info['llama_cpp_available'] else 'Not installed'}")
    gpu_offload_status = (
        "Supported (CUDA / Metal / Vulkan enabled)"
        if info["llama_gpu_offload"]
        else "CPU only (compile with CUDA: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python)"
    )
    print(f" llama GPU Offload     : {gpu_offload_status}")
    print("-" * 65)

    # Recommendations for 27B parameter models
    total_vram = sum(d["total_memory_gb"] for d in info.get("devices", []))
    print(" 27B Parameter Model Feasibility Analysis:")
    if total_vram >= 32.0:
        print(f" [PASS] Total VRAM ({total_vram:.1f} GB) is sufficient to run 27B models (Q4_K_M ~17GB, Q8_0 ~29GB)")
        print("        with 100% layer offload to GPU (-ngl -1) or via vLLM / llama-server.")
    elif total_vram >= 16.0:
        print(f" [PASS] Total VRAM ({total_vram:.1f} GB) can host 27B quantized models (Q4_K_M ~17GB)")
        print("        with partial or full layer offload, or via 4-bit AWQ / GPTQ.")
    elif total_vram > 0:
        print(f" [WARN] Total VRAM ({total_vram:.1f} GB) is tight for 27B. Recommend offloading partial layers")
        print("        or hosting 27B via an OpenAI-compatible compute server (e.g., vLLM with --base-url).")
    else:
        print(" [NOTE] No GPU detected. Local 27B will execute on CPU threads (slower) or connect via --base-url to an HPC model server.")
    print("=" * 65)


def collect_concepts(args: argparse.Namespace) -> list[str]:
    """Collect and deduplicate concepts from file, arguments, or standard input."""
    raw_concepts: list[str] = []

    # 1. From file
    if args.concepts_file:
        if args.concepts_file.strip() == "-":
            for line in sys.stdin:
                line_str = line.strip()
                if line_str and not line_str.startswith("#"):
                    raw_concepts.append(line_str)
        else:
            file_path = Path(args.concepts_file).expanduser().resolve()
            if not file_path.is_file():
                raise FileNotFoundError(f"Concepts file not found: {file_path}")
            raw_concepts.extend(load_concepts(file_path, subject_id=args.subject_id))

    # 2. From direct CLI arguments
    if args.concepts:
        for item in args.concepts:
            if item.strip() == "-":
                for line in sys.stdin:
                    line_str = line.strip()
                    if line_str and not line_str.startswith("#"):
                        raw_concepts.append(line_str)
            else:
                for part in item.split(","):
                    clean = part.strip()
                    if clean:
                        raw_concepts.append(clean)

    # 3. Read from stdin if piped and no arguments were passed
    if not raw_concepts and not sys.stdin.isatty():
        for line in sys.stdin:
            line_str = line.strip()
            if line_str and not line_str.startswith("#"):
                raw_concepts.append(line_str)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_concepts: list[str] = []
    for c in raw_concepts:
        key = c.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            unique_concepts.append(c.strip())

    return unique_concepts


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="stimstudio-cli",
        description="StimStudio CLI: Headless cognitive stimulus generation for HPC and batch clusters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check GPU & CUDA acceleration capabilities for 27B models
  stimstudio-cli --check-gpu

  # Generate stimuli using a local 27B GGUF model on HPC GPU:
  stimstudio-cli --concepts "Albert Einstein" "Marie Curie" \\
                 --provider local \\
                 --model /scratch/models/gemma-2-27b-it.Q4_K_M.gguf \\
                 --gpu-layers -1 --output results.json

  # Read a list of concepts from file and use an OpenAI-compatible vLLM server:
  stimstudio-cli -f data/concepts.txt \\
                 --provider compatible \\
                 --base-url http://localhost:8000/v1 \\
                 --model google/gemma-2-27b-it \\
                 -o stimuli.csv

  # Pipe concepts through standard input:
  cat concepts.txt | stimstudio-cli - --provider google -o output.json
""",
    )

    # Concept input arguments
    input_group = parser.add_argument_group("Concept Inputs")
    input_group.add_argument(
        "concepts",
        nargs="*",
        help="One or more seed concepts, or '-' to read from stdin.",
    )
    input_group.add_argument(
        "-f",
        "--concepts-file",
        dest="concepts_file",
        help="Path to concepts list file (.txt, .csv, .xlsx, or clinical questionnaire), or '-' for stdin.",
    )
    input_group.add_argument(
        "--context",
        default="",
        help="Optional domain/category context to guide generation (e.g. 'Neurology visual screening').",
    )
    input_group.add_argument(
        "--subject-id",
        dest="subject_id",
        default="",
        help="Participant or subject ID (e.g. 'SUBJ001') for folder organization and traceability.",
    )

    # Model and Provider arguments
    model_group = parser.add_argument_group("Model & Inference Configuration")
    model_group.add_argument(
        "-p",
        "--provider",
        choices=["local", "compatible", "openai", "anthropic", "google"],
        default="local",
        help="Inference engine: 'local' (llama.cpp GGUF), 'compatible' (vLLM/Ollama/llama-server), or cloud (default: local).",
    )
    model_group.add_argument(
        "-m",
        "--model",
        default="",
        help="Model identifier (e.g. 'google/gemma-2-27b-it' for compatible, or path to .gguf for local).",
    )
    model_group.add_argument(
        "--model-path",
        dest="model_path",
        default="",
        help="Path to local GGUF model file (overrides config/env STIMULUS_FORGE_LLM_MODEL).",
    )
    model_group.add_argument(
        "-ngl",
        "--gpu-layers",
        dest="gpu_layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU for local GGUF models (-1 = all layers, default: -1).",
    )
    model_group.add_argument(
        "-c",
        "--ctx-size",
        dest="ctx_size",
        type=int,
        default=4096,
        help="Context window size in tokens for local GGUF models (default: 4096).",
    )
    model_group.add_argument(
        "--base-url",
        dest="base_url",
        default="",
        help="Base URL for OpenAI-compatible inference servers (e.g., http://localhost:8000/v1 for vLLM).",
    )
    model_group.add_argument(
        "--api-key",
        dest="api_key",
        default="",
        help="API key for cloud providers or secured compatible endpoints.",
    )

    # Generation parameters
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "-n",
        "--count",
        type=int,
        default=DEFAULT_CONCEPT_COUNT,
        help=f"Number of related stimuli to generate per seed (default: {DEFAULT_CONCEPT_COUNT}).",
    )
    gen_group.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        help="Bypass semantic LRU cache and force fresh inference.",
    )

    # Optional image acquisition
    img_group = parser.add_argument_group("Image Acquisition (Optional)")
    img_group.add_argument(
        "--with-images",
        dest="with_images",
        action="store_true",
        help="Download or synthesize visual images for generated concepts.",
    )
    img_group.add_argument(
        "--image-source",
        dest="image_source",
        choices=["bing", "wikimedia", "duckduckgo", "openverse", "openai", "google", "local", "text"],
        default="bing",
        help="Source for image acquisition (default: bing).",
    )
    img_group.add_argument(
        "--images-per-concept",
        dest="images_per_concept",
        type=int,
        default=1,
        help="Number of image candidates per related concept (default: 1).",
    )
    img_group.add_argument(
        "--detect-faces",
        dest="detect_faces",
        action="store_true",
        help="Perform automated face detection on candidate images.",
    )
    img_group.add_argument(
        "--crop-faces",
        dest="crop_faces",
        action="store_true",
        help="Crop and align detected faces into standard square stimuli.",
    )

    # Output options
    out_group = parser.add_argument_group("Output & Formatting")
    out_group.add_argument(
        "-o",
        "--output",
        dest="output_file",
        help="File path to write output (.json or .csv). Defaults to stdout if omitted.",
    )
    out_group.add_argument(
        "--format",
        choices=["json", "csv", "table"],
        default=None,
        help="Output format: json, csv, or table (auto-detected from output file extension if omitted).",
    )
    out_group.add_argument(
        "--output-dir",
        dest="output_dir",
        default="",
        help="Workspace directory for stimulus images and artifacts (default: data/workspace).",
    )
    out_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode: suppress progress messages and banners, emit only final output.",
    )
    out_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose mode: print detailed token usage, FinOps cost estimation, and latency.",
    )

    # Diagnostics
    parser.add_argument(
        "--check-gpu",
        "--check-env",
        dest="check_gpu",
        action="store_true",
        help="Check GPU, CUDA, and llama.cpp acceleration status and exit.",
    )

    return parser


def run_cli(argv: list[str] | None = None) -> int:
    """Execute the StimStudio CLI workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check_gpu:
        diag = check_gpu_environment()
        print_gpu_diagnostic(diag)
        return 0

    # Collect concepts
    try:
        concepts = collect_concepts(args)
    except Exception as exc:
        print(f"[ERROR] Failed to load concepts: {exc}", file=sys.stderr)
        return 1

    if not concepts:
        print("[ERROR] No concepts provided. Specify seed concepts, a file (-f), or pass through stdin.", file=sys.stderr)
        parser.print_usage(file=sys.stderr)
        return 1

    # Output format deduction
    output_format = args.format
    if not output_format:
        if args.output_file:
            ext = Path(args.output_file).suffix.lower()
            output_format = "csv" if ext == ".csv" else "json"
        else:
            output_format = "table"

    if not args.quiet:
        print("=" * 65)
        print(" StimStudio CLI - Cognitive Stimulus Generation")
        print("=" * 65)
        print(f" Seed Concepts : {len(concepts)} unique seeds")
        print(f" Provider      : {args.provider}")
        model_display = args.model or args.model_path or config.get("llm_model") or "default"
        print(f" Model         : {model_display}")
        if args.provider == "local":
            print(f" GPU Offload   : {args.gpu_layers} layers (Context: {args.ctx_size} tokens)")
        elif args.provider == "compatible":
            print(f" Base URL      : {args.base_url or 'http://localhost:8000/v1'}")
        if args.subject_id:
            print(f" Subject ID    : {args.subject_id}")
        if args.context:
            print(f" Category/Ctx  : {args.context}")
        print(f" Concepts/Seed : {args.count}")
        print(f" Acquire Images: {'Yes (' + args.image_source + ')' if args.with_images else 'No'}")
        print("-" * 65)

    # Resolve model path
    resolved_model_path = args.model_path
    if not resolved_model_path and args.provider == "local" and args.model:
        if Path(args.model).is_file():
            resolved_model_path = args.model

    # Instantiate generator
    try:
        generator = RelatedConceptGenerator(
            provider=args.provider,
            openai_model=args.model or "gpt-4o",
            local_model_path=resolved_model_path,
            local_gpu_layers=args.gpu_layers,
            api_key=args.api_key,
            base_url=args.base_url,
            n_ctx=args.ctx_size,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to initialize model generator: {exc}", file=sys.stderr)
        return 1

    # Processing loop
    results: list[dict[str, Any]] = []
    total_associations = 0
    total_images = 0
    start_total_time = time.perf_counter()

    for idx, seed in enumerate(concepts, start=1):
        if not args.quiet:
            print(f"[{idx}/{len(concepts)}] Generating {args.count} stimuli for: \"{seed}\" ...", end="", flush=True)

        t0 = time.perf_counter()
        try:
            items = generator.generate(seed, count=args.count, context=args.context)
            dt = time.perf_counter() - t0
            total_associations += len(items)
            if not args.quiet:
                print(f" Done ({len(items)} items in {dt:.2f}s)")
        except Exception as exc:
            if not args.quiet:
                print(" FAILED")
            print(f"[ERROR] Generation failed for '{seed}': {exc}", file=sys.stderr)
            items = []

        seed_entry: dict[str, Any] = {
            "seed": seed,
            "related_concepts": items,
            "images": [],
        }

        # Optional multimodal image acquisition
        if args.with_images and items:
            from stimulus_forge.api.routes.images import search_images, generate_images
            from stimulus_forge.api.schemas import ImageSearchRequest, ImageGenerateRequest

            for item in items:
                name_str = item.get("name", "")
                char_str = item.get("char", "")
                desc_str = item.get("description", "")
                target_query = char_str or name_str

                if not args.quiet:
                    print(f"   ↳ Acquiring image for: {target_query} ...", end="", flush=True)

                if args.image_source in {"bing", "wikimedia", "duckduckgo", "openverse"}:
                    req = ImageSearchRequest(
                        concept=seed,
                        related_concept=name_str,
                        name=char_str,
                        limit=args.images_per_concept,
                        provider=args.image_source,
                        detect_faces=args.detect_faces,
                        crop_faces=args.crop_faces,
                        output_dir=args.output_dir or None,
                        subject_id=args.subject_id or None,
                    )
                    try:
                        resp = search_images(req)
                        seed_entry["images"].extend([im.model_dump() for im in resp.images])
                        total_images += len(resp.images)
                        if not args.quiet:
                            print(f" ({len(resp.images)} found)")
                    except Exception as err:
                        if not args.quiet:
                            print(f" (search failed: {err})")
                else:
                    g_req = ImageGenerateRequest(
                        prompt=f"{target_query} {desc_str}".strip(),
                        concept=seed,
                        related_concept=name_str,
                        name=char_str,
                        description=desc_str,
                        count=args.images_per_concept,
                        provider=args.image_source,
                        detect_faces=args.detect_faces,
                        crop_faces=args.crop_faces,
                        output_dir=args.output_dir or None,
                        subject_id=args.subject_id or None,
                    )
                    try:
                        resp = generate_images(g_req)
                        seed_entry["images"].extend([im.model_dump() for im in resp.images])
                        total_images += len(resp.images)
                        if not args.quiet:
                            print(f" ({len(resp.images)} synthesized)")
                    except Exception as err:
                        if not args.quiet:
                            print(f" (gen failed: {err})")

        results.append(seed_entry)

    total_duration = time.perf_counter() - start_total_time

    if not args.quiet:
        print("=" * 65)
        print(f" Complete: {len(results)} seeds processed, {total_associations} related stimuli generated in {total_duration:.2f}s.")
        if args.with_images:
            print(f" Images Acquired: {total_images}")
        print("=" * 65)

    # Format and emit output
    output_payload = {
        "subject_id": args.subject_id,
        "provider": args.provider,
        "model": args.model or args.model_path or "default",
        "total_seeds": len(results),
        "total_associations": total_associations,
        "duration_seconds": round(total_duration, 2),
        "results": results,
    }

    if output_format == "json":
        json_str = json.dumps(output_payload, indent=2, ensure_ascii=False)
        if args.output_file:
            out_path = Path(args.output_file).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_str, encoding="utf-8")
            if not args.quiet:
                print(f"[OK] Results written to: {out_path}")
        else:
            print(json_str)

    elif output_format == "csv":
        rows: list[dict[str, str]] = []
        for entry in results:
            seed = entry["seed"]
            for rel in entry.get("related_concepts", []):
                rows.append({
                    "seed": seed,
                    "name": rel.get("name", ""),
                    "character": rel.get("char", ""),
                    "description": rel.get("description", ""),
                })
        
        if args.output_file:
            out_path = Path(args.output_file).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["seed", "name", "character", "description"])
                writer.writeheader()
                writer.writerows(rows)
            if not args.quiet:
                print(f"[OK] Results written to: {out_path}")
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=["seed", "name", "character", "description"])
            writer.writeheader()
            writer.writerows(rows)

    else:
        # Table / human readable
        for entry in results:
            print(f"\n Seed: {entry['seed'].upper()}")
            for i, rel in enumerate(entry.get("related_concepts", []), start=1):
                char_info = f" (as {rel['char']})" if rel.get("char") else ""
                desc_info = f" — {rel['description']}" if rel.get("description") else ""
                print(f"   {i:2d}. {rel.get('name')}{char_info}{desc_info}")

    return 0


def main() -> int:
    """CLI script entry point."""
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
