# Reproduction Setup Guide — RoBERTa-base on SciFact

Two workflows: **Online** (node with internet + GPU) and **Offline** (RTX 6000 / gpunode1 without internet).

## Prerequisites

- NVIDIA GPU with >= 8GB VRAM (RoBERTa-base is ~125M params x3 models)
- Python 3.9+, conda
- Java 11+ (required for Elasticsearch, login node only)

## Step 1: Create conda environment (login node)

```bash
conda create -n task-arithmetic python=3.11 -y
conda activate task-arithmetic
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify GPU (on gpunode1):
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## OFFLINE WORKFLOW (for RTX 6000 / gpunode1 without internet)

### Step 2: Download everything (login node — has internet)

```bash
conda activate task-arithmetic
cd Task-Arithmetic
python Scripts/download_all.py --cache_dir ./offline_cache
```

This downloads to `offline_cache/`:
- `datasets/scifact/` and `datasets/nfcorpus/` (BEIR datasets)
- `models/roberta-base/` (Θ₀, ~500MB)
- `models/allenai__biomed_roberta_base/` (Θ_D, ~500MB)
- `models/cross-encoder__stsb-roberta-base/` (Θ_T, ~500MB)

Total: ~1.5 GB

### Step 3: Run BM25 and cache results (login node — needs Elasticsearch)

```bash
# Start Elasticsearch (one-time)
bash Scripts/setup_elasticsearch.sh

# Save BM25 results for all splits
cd Scripts
python reproduce_roberta_offline.py --cache_dir ../offline_cache --mode save_bm25
```

This saves JSON files to `offline_cache/bm25_cache/`.

### Step 4: Run on gpunode1 (fully offline)

```bash
# SSH into gpunode1
conda activate task-arithmetic
cd Task-Arithmetic/Scripts

# α-sweep on dev sets
python reproduce_roberta_offline.py \
    --cache_dir ../offline_cache \
    --mode dev_sweep \
    --skip_bm25 \
    --device cuda:0

# Test with best α
python reproduce_roberta_offline.py \
    --cache_dir ../offline_cache \
    --mode test \
    --alfa 0.3 \
    --skip_bm25 \
    --device cuda:0

# Baselines
python reproduce_roberta_offline.py \
    --cache_dir ../offline_cache \
    --mode baselines \
    --skip_bm25 \
    --device cuda:0
```

Output: `results/dev_sweep_results.csv`, `results/test_results.csv`

---

## ONLINE WORKFLOW (node with internet + GPU, e.g. H200)

### Step 2: Start Elasticsearch

```bash
bash Scripts/setup_elasticsearch.sh
curl http://localhost:9200
```

### Step 3: Run (downloads models/datasets automatically)

```bash
cd Scripts

# α-sweep
python reproduce_roberta_scifact.py --mode dev_sweep --device cuda:0

# Test
python reproduce_roberta_scifact.py --mode test --alfa 0.3 --device cuda:0

# Baselines
python reproduce_roberta_scifact.py --mode baselines --device cuda:0
```

## Step 5: Compare against Table 1

Expected results for RoBERTa-base on SciFact (from the paper):

| Variant          | P@10 | NDCG@3 | NDCG@10 | MAP@100 |
|------------------|------|--------|---------|---------|
| BM25             | .091 | .637   | .691    | .649    |
| Θ₀ (pretrained)  | .090 | .633   | .686    | .646    |
| Θ_T (msmarco)    | .095 | .655   | .707    | .662    |
| Θ′ (α=1)         | .092 | .649   | .700    | .659    |
| Θ′ (α=0.3, best) | .096 | .669   | .720    | .676    |

Your results should be within ±0.01–0.02 of these.

## IMPORTANT: The Θ_T Model ID Question

The script defaults to `cross-encoder/stsb-roberta-base` for Θ_T because the
paper's "msmarco-RoBERTa" cross-encoder could not be identified on HuggingFace.

**Before running**, try to verify the correct model. Check:

1. The paper's supplementary material / appendix for exact HF model IDs
2. Email the authors (Braga et al.) — they may share their checkpoint
3. Check if `cross-encoder/ms-marco-cross-encoder-roberta-base` exists (it doesn't as of July 2026, but may be uploaded)

If the model is wrong, the pipeline will still run end-to-end, but NDCG scores
will differ from Table 1. The task arithmetic *mechanism* (subtraction/addition/α-scaling)
is independent of which Θ_T is used — so you can still verify the pipeline works.

To use a different Θ_T:

```bash
python reproduce_roberta_scifact.py --mode test --alfa 0.3 \
  --model_base_path "your/model-id-here" --device cuda:0
```

## Troubleshooting

### "Connection refused" from BM25
Elasticsearch isn't running. Start it with `bash Scripts/setup_elasticsearch.sh`.

### Out of Memory
RoBERTa-base models are small (~500MB each). If you hit OOM:
- Check that no other processes are using the GPU: `nvidia-smi`
- Reduce batch size: `--batch_size 32`

### "monot5 not found"
`pip install monot5` — this is imported by utils.py even though we don't use it
for the RoBERTa experiment. You can also comment out the monot5 import in utils.py.

### Scores wildly different from paper
Most likely cause: wrong Θ_T model. The key.replace('roberta.','') mapping in
TaskVectorRoBERTa assumes the cross-encoder uses a RoBERTa backbone. If the model
uses BERT keys (prefixed with 'bert.'), the task vector won't be applied and you'll
get Θ_T baseline scores regardless of α.

Check the log for "X keys merged, Y keys kept as-is". If "0 keys merged", the
prefix stripping is wrong — adjust the replace() call in the script.
