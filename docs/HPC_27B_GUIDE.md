# Running StimStudio on HPC Clusters with 27B Parameter Models

This guide details how to run headless cognitive stimulus generation on High Performance Computing (HPC) clusters utilizing large GPUs (e.g., NVIDIA A100 40GB/80GB, H100 80GB, L40S 48GB, RTX 6000 Ada) and 27B parameter open-weights models (such as **Google Gemma 2 27B** or **Qwen 2.5 27B**).

---

## 🎯 Why 27B Parameter Models?

For cognitive neuroscience and psychological experiments, related concept associations must balance:
- **Broad World Knowledge & Visual Granularity**: Identifying recognizable physical associations (people, characters, artifacts, landmarks) rather than abstract semantic traits.
- **Strict Schema Adherence**: Generating structured outputs with accurate role separation (`name`, `char`, `description`).
- **Air-Gapped Privacy & HIPAA/IRB Compliance**: Running offline without sending clinical patient questionnaire seeds to commercial cloud APIs.

27B parameter models strike the optimal balance between 70B+ model capability and rapid, resource-efficient local GPU deployment.

---

## 🖥️ GPU Sizing & Quantization Guide for 27B Models

| Model & Quantization | Approximate VRAM Needed | Recommended HPC GPU | Layer Offload (`-ngl`) |
| :--- | :--- | :--- | :--- |
| **Gemma 2 27B (Q4_K_M)** | ~17 - 18 GB | A100 (40GB/80GB), L40S (48GB), RTX 6000 (48GB), V100 (32GB) | `-ngl -1` (100% on GPU) |
| **Gemma 2 27B (Q8_0)** | ~29 - 30 GB | A100 (40GB/80GB), H100 (80GB), L40S (48GB), RTX 6000 (48GB) | `-ngl -1` (100% on GPU) |
| **Gemma 2 27B (FP16 / BF16)** | ~54 - 58 GB | A100 (80GB), H100 (80GB), or $2\times$ A100 (40GB) | Tensor Parallel via vLLM |
| **Qwen 2.5 27B (Q4_K_M)** | ~17 - 18 GB | A100 (40GB/80GB), L40S, RTX 6000 Ada, V100 (32GB) | `-ngl -1` (100% on GPU) |
| **Qwen 2.5 27B (Q8_0)** | ~29 - 30 GB | A100 (40GB/80GB), H100 (80GB), L40S (48GB) | `-ngl -1` (100% on GPU) |

---

## 🚀 Two Execution Modes on HPC

StimStudio CLI supports two distinct execution patterns on HPC:

### Pattern 1: Direct Local GGUF via `llama-cpp-python` (Zero Setup)
Load the `.gguf` file directly in the python process and offload all layers to GPU VRAM with `-ngl -1`.

```bash
# 1. Download model weights (e.g. from Hugging Face)
huggingface-cli download bartowski/gemma-2-27b-it-GGUF \
    gemma-2-27b-it-Q4_K_M.gguf \
    --local-dir /scratch/models/

# 2. Run StimStudio CLI
stimstudio-cli \
    --concepts-file data/concepts.txt \
    --provider local \
    --model-path /scratch/models/gemma-2-27b-it-Q4_K_M.gguf \
    --gpu-layers -1 \
    --count 10 \
    --output stimuli_results.json
```

### Pattern 2: OpenAI-Compatible Server on Compute Node (vLLM / llama-server)
Run a dedicated inference engine on the compute node (especially useful for multi-GPU tensor parallelism or FP16 models), then connect StimStudio CLI to the local port:

```bash
# 1. Start vLLM on compute node
vllm serve google/gemma-2-27b-it \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 &

# 2. Run StimStudio CLI pointing to vLLM
stimstudio-cli \
    --concepts-file data/concepts.txt \
    --provider compatible \
    --base-url http://localhost:8000/v1 \
    --model google/gemma-2-27b-it \
    --count 10 \
    --output stimuli_results.json
```

---

## 🛠️ HPC Environment Setup

### 1. CUDA & Environment Installation

When installing on an HPC cluster with CUDA GPUs, ensure `llama-cpp-python` is compiled with CUDA acceleration enabled:

```bash
# Load cluster CUDA module
module load cuda/12.2

# Create environment
conda create -n stimstudio python=3.10 -y
conda activate stimstudio

# Clone and install dependencies
cd stimstudio
pip install -e '.[server,cloud]'

# Install CUDA-accelerated llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

### 2. Verify GPU Acceleration

Run the built-in diagnostic tool to verify CUDA detection and GPU VRAM:

```bash
stimstudio-cli --check-gpu
```

Expected output on an A100 GPU node:
```text
=================================================================
 StimStudio HPC & GPU Acceleration Diagnostic
=================================================================
 Python Version        : 3.10.x
 PyTorch Installed     : Yes
 CUDA Available (PyTorch): Yes
 Detected GPU Count    : 1
   - [GPU 0] NVIDIA A100-SXM4-80GB (80.0 GB VRAM, CC 8.0)
 llama-cpp-python      : Installed
 llama GPU Offload     : Supported (CUDA / Metal / Vulkan enabled)
-----------------------------------------------------------------
 27B Parameter Model Feasibility Analysis:
 [PASS] Total VRAM (80.0 GB) is sufficient to run 27B models (Q4_K_M ~17GB, Q8_0 ~29GB)
        with 100% layer offload to GPU (-ngl -1) or via vLLM / llama-server.
=================================================================
```

---

## 📋 Providing Concepts to the CLI

You can provide concepts in three flexible ways:

### 1. From a File (`.txt`, `.csv`, `.xlsx`, or questionnaire)
```bash
stimstudio-cli -f my_concepts.txt --provider local -m model.gguf -ngl -1
```

Supports:
- Text files (`.txt`), one concept per line
- CSV/Excel spreadsheets (`.csv`, `.xlsx`), automatically detecting column headers like `name`, `concept`, or `stimulus`
- Clinical neuroscience questionnaires (e.g. epilepsy/cognitive screening protocols)

### 2. Direct Command-Line Arguments
```bash
stimstudio-cli "Albert Einstein" "Marie Curie" "Isaac Newton" \
    --provider local \
    --model-path /scratch/models/gemma-2-27b-it-Q4_K_M.gguf \
    --output scientists.json
```

### 3. Piped from Standard Input (`stdin`)
```bash
cat concepts.txt | stimstudio-cli - --provider local -m model.gguf -ngl -1 -o output.csv
```

---

## ⚡ Slurm Batch Script Example

Submit a batch job using the provided template [`scripts/slurm_run_27b.sh`](file:///Users/smathew/Library/CloudStorage/OneDrive-mcw.edu/WiredBrainsLab/stimstudio/scripts/slurm_run_27b.sh):

```bash
sbatch scripts/slurm_run_27b.sh data/concepts.txt data/results.json
```

Or run interactively in a Slurm GPU session:
```bash
srun --partition=gpu --gres=gpu:1 --mem=64G --cpus-per-task=8 --pty bash
conda activate stimstudio
stimstudio-cli -f data/concepts.txt -p local -m /scratch/models/gemma-2-27b-it.Q4_K_M.gguf -ngl -1 -o results.json
```

---

## 📊 Export Formats

### JSON (`--format json` or `-o results.json`)
```json
{
  "subject_id": "SUBJ001",
  "provider": "local",
  "model": "gemma-2-27b-it.Q4_K_M.gguf",
  "total_seeds": 1,
  "total_associations": 2,
  "duration_seconds": 1.45,
  "results": [
    {
      "seed": "Albert Einstein",
      "related_concepts": [
        {
          "name": "General Relativity",
          "char": "",
          "description": "Einstein's landmark geometric theory of gravitation"
        },
        {
          "name": "Institute for Advanced Study",
          "char": "",
          "description": "Princeton research center where Einstein worked"
        }
      ],
      "images": []
    }
  ]
}
```

### CSV (`--format csv` or `-o stimuli.csv`)
```csv
seed,name,character,description
Albert Einstein,General Relativity,,Einstein's landmark geometric theory of gravitation
Albert Einstein,Institute for Advanced Study,,Princeton research center where Einstein worked
```
