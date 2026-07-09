#!/bin/bash
# ============================================================
# MT5-base Reproduction — GermanQuAD (Phase 1)
# Run on gpunode1 with Elasticsearch running
# ============================================================
set -euo pipefail

PROJECT_DIR="/home/research5/dharmik_workspace/Task-Arithmetic"
CACHE_DIR="${PROJECT_DIR}/offline_cache"
RESULTS_DIR="${PROJECT_DIR}/Scripts/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${RESULTS_DIR}/mt5_gpunode_${TIMESTAMP}.log"

mkdir -p "${RESULTS_DIR}"

echo "=============================================="
echo "MT5-base Reproduction — GermanQuAD"
echo "Started: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Cache:   ${CACHE_DIR}"
echo "Output:  ${RESULTS_DIR}"
echo "Log:     ${LOG_FILE}"
echo "=============================================="

cd "${PROJECT_DIR}"
# Source conda profile setup to allow activation inside non-interactive shells
if [ -f "/home/research5/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/research5/miniconda3/etc/profile.d/conda.sh"
fi
conda activate task

# --- Check prerequisites ---
echo ""
echo "[0/3] Checking prerequisites..."

# GPU
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB')
"

# Elasticsearch
python -c "
from elasticsearch import Elasticsearch
es = Elasticsearch('http://localhost:9200', timeout=10)
assert es.ping(), 'Elasticsearch not running!'
print('Elasticsearch: OK')
"

# Cache
echo ""
echo "Cache contents:"
ls -lh "${CACHE_DIR}/models/" 2>/dev/null || echo "  No models found — run download_mt5.py first!"
ls -lh "${CACHE_DIR}/datasets/germanquad/" 2>/dev/null || echo "  No GermanQuAD found — run download_mt5.py first!"

# --- Step 1: BM25 baseline ---
echo ""
echo "[1/3] Running BM25 (save to cache)..."
python Scripts/reproduce_mt5_offline.py \
    --cache_dir "${CACHE_DIR}" \
    --dataset germanquad \
    --mode save_bm25 \
    2>&1 | tee -a "${LOG_FILE}"

# --- Step 2: Baselines (BM25 + Θ_T alone) ---
echo ""
echo "[2/3] Running baselines (BM25 + Θ_T MonoT5)..."
python Scripts/reproduce_mt5_offline.py \
    --cache_dir "${CACHE_DIR}" \
    --dataset germanquad \
    --mode baselines \
    --device cuda:0 \
    --skip_bm25 \
    --output_dir "${RESULTS_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

# --- Step 3: Task arithmetic (α=1.0) ---
echo ""
echo "[3/3] Running task arithmetic (α=1.0, zero-shot)..."
python Scripts/reproduce_mt5_offline.py \
    --cache_dir "${CACHE_DIR}" \
    --dataset germanquad \
    --mode test \
    --alfa 1.0 \
    --device cuda:0 \
    --skip_bm25 \
    --output_dir "${RESULTS_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "=============================================="
echo "Done: $(date)"
echo "Results in: ${RESULTS_DIR}"
echo "Log: ${LOG_FILE}"
echo "=============================================="
