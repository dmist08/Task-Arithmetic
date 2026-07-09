#!/bin/bash
# Run the full offline reproduction on gpunode1 inside tmux.
#
# Usage (on login node):
#   ssh gpunode1
#   tmux new -s task-arith
#   cd /path/to/Task-Arithmetic
#   bash Scripts/run_on_gpunode.sh
#
# To detach: Ctrl+B then D
# To reattach: tmux attach -t task-arith
# To check progress: tail -f Scripts/results/offline_dev_sweep.log

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="${PROJECT_DIR}/offline_cache"
OUTPUT_DIR="${SCRIPT_DIR}/results"
DEVICE="cuda:0"

# Force offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=============================================="
echo "Task Arithmetic — Offline Reproduction"
echo "Started: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Cache:   ${CACHE_DIR}"
echo "Output:  ${OUTPUT_DIR}"
echo "Device:  ${DEVICE}"
echo "=============================================="


# Verify GPU
echo ""
echo "[0/4] GPU check..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB')"

# Verify cache exists
if [ ! -d "${CACHE_DIR}/models" ] || [ ! -d "${CACHE_DIR}/datasets" ]; then
    echo "ERROR: offline_cache not found at ${CACHE_DIR}"
    echo "Run download_all.py on login node first."
    exit 1
fi

if [ ! -f "${CACHE_DIR}/bm25_cache/bm25_scifact_dev.json" ]; then
    echo "ERROR: BM25 cache not found. Run save_bm25 mode on login node first."
    exit 1
fi

echo ""
echo "Cache contents:"
ls -lh "${CACHE_DIR}/models/"
ls -lh "${CACHE_DIR}/bm25_cache/"
echo ""

cd "${SCRIPT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# --- Step 1: Dev sweep (α = 0.0 to 1.0) ---
echo "=============================================="
echo "[1/4] Dev sweep (α = 0.0 to 1.0)..."
echo "Started: $(date)"
echo "=============================================="

python reproduce_roberta_offline.py \
    --cache_dir "${CACHE_DIR}" \
    --mode dev_sweep \
    --skip_bm25 \
    --device "${DEVICE}" \
    --batch_size 128

echo ""
echo "[1/4] Dev sweep DONE at $(date)"
echo "Results: ${OUTPUT_DIR}/dev_sweep_results.csv"
cat "${OUTPUT_DIR}/dev_sweep_results.csv"
echo ""

# --- Step 2: Baselines ---
echo "=============================================="
echo "[2/4] Baselines (BM25, Θ_T)..."
echo "Started: $(date)"
echo "=============================================="

python reproduce_roberta_offline.py \
    --cache_dir "${CACHE_DIR}" \
    --mode baselines \
    --skip_bm25 \
    --device "${DEVICE}"

echo ""
echo "[2/4] Baselines DONE at $(date)"
echo ""

# --- Step 3: Test with α=1.0 (fully zero-shot) ---
echo "=============================================="
echo "[3/4] Test with α=1.0 (fully zero-shot)..."
echo "Started: $(date)"
echo "=============================================="

python reproduce_roberta_offline.py \
    --cache_dir "${CACHE_DIR}" \
    --mode test \
    --alfa 1.0 \
    --skip_bm25 \
    --device "${DEVICE}"

echo ""
echo "[3/4] Test α=1.0 DONE at $(date)"
echo ""

# --- Step 4: Test with α=0.3 (best from paper) ---
echo "=============================================="
echo "[4/4] Test with α=0.3 (best from paper)..."
echo "Started: $(date)"
echo "=============================================="

python reproduce_roberta_offline.py \
    --cache_dir "${CACHE_DIR}" \
    --mode test \
    --alfa 0.3 \
    --skip_bm25 \
    --device "${DEVICE}"

echo ""
echo "[4/4] Test α=0.3 DONE at $(date)"
echo ""

# --- Summary ---
echo "=============================================="
echo "ALL DONE — $(date)"
echo "=============================================="
echo ""
echo "Output files:"
ls -lh "${OUTPUT_DIR}"/*.csv "${OUTPUT_DIR}"/*.log 2>/dev/null
echo ""
echo "To view dev sweep:  cat ${OUTPUT_DIR}/dev_sweep_results.csv"
echo "To view test:       cat ${OUTPUT_DIR}/test_results.csv"
echo "To view logs:       cat ${OUTPUT_DIR}/offline_*.log"
