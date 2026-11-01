#!/usr/bin/env bash
#SBATCH --job-name=stimstudio-27b
#SBATCH --output=stimstudio_%j.log
#SBATCH --error=stimstudio_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # 1 GPU (A100-40GB/80GB, H100, or L40S recommended for 27B models)
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# ==============================================================================
# StimStudio HPC Slurm Batch Runner for 27B Parameter Models
# ==============================================================================

set -euo pipefail

echo "================================================================="
echo " Job ID           : ${SLURM_JOB_ID:-local}"
echo " Compute Host     : $(hostname)"
echo " Start Time       : $(date)"
echo " Allocated GPUs   : ${CUDA_VISIBLE_DEVICES:-None}"
echo "================================================================="

# 1. Load HPC Environment Modules (adjust to match your cluster's module naming)
if command -v module &> /dev/null; then
    module purge || true
    module load cuda/12.2 || module load cuda/12.1 || module load cuda/11.8 || true
fi

# 2. Activate Conda / Python Virtual Environment
# Replace with the path to your cluster's conda or venv:
# source /path/to/miniconda3/bin/activate concept-studio

# 3. Check GPU status and CUDA acceleration
stimstudio-cli --check-gpu

# 4. Configuration Variables
CONCEPTS_FILE="${1:-data/concepts.txt}"
OUTPUT_FILE="${2:-data/stimuli_27b_results.json}"
MODEL_PATH="${MODEL_PATH:-/scratch/models/gemma-2-27b-it.Q4_K_M.gguf}"
SUBJECT_ID="${SUBJECT_ID:-HPC_BATCH_01}"

echo "Processing concepts from: ${CONCEPTS_FILE}"
echo "Output will be saved to : ${OUTPUT_FILE}"
echo "Using 27B model GGUF    : ${MODEL_PATH}"

# ==============================================================================
# OPTION A: Direct Local GGUF Execution via llama.cpp GPU Offload
# ==============================================================================
# -ngl -1 offloads all model layers to the allocated GPU VRAM
stimstudio-cli \
    --concepts-file "${CONCEPTS_FILE}" \
    --provider local \
    --model-path "${MODEL_PATH}" \
    --gpu-layers -1 \
    --ctx-size 4096 \
    --count 10 \
    --subject-id "${SUBJECT_ID}" \
    --output "${OUTPUT_FILE}" \
    --format json

# ==============================================================================
# OPTION B: Running via Local vLLM Inference Server on HPC Node (Alternative)
# ==============================================================================
# If running vLLM for high-throughput serving:
# 1. Start vLLM in the background:
#    vllm serve google/gemma-2-27b-it --port 8000 --gpu-memory-utilization 0.90 &
#    VLLM_PID=$!
#    sleep 25
# 2. Execute StimStudio CLI pointing to local endpoint:
#    stimstudio-cli \
#        --concepts-file "${CONCEPTS_FILE}" \
#        --provider compatible \
#        --base-url "http://localhost:8000/v1" \
#        --model "google/gemma-2-27b-it" \
#        --output "${OUTPUT_FILE}"
# 3. Clean up server:
#    kill ${VLLM_PID} || true

echo "================================================================="
echo " StimStudio batch job finished successfully at: $(date)"
echo " Results stored in: ${OUTPUT_FILE}"
echo "================================================================="
