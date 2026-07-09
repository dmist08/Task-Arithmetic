# Reproduction Setup Guide — RoBERTa-base on SciFact

Step-by-step instructions to run the Task Arithmetic reproduction on your H200 cluster.

## Prerequisites

- NVIDIA GPU with >= 8GB VRAM (RoBERTa-base is ~125M params x3 models, well under any modern GPU)
- Python 3.9+
- Java 11+ (required for Elasticsearch)

## Step 0: Clone and switch to branch

```bash
git clone https://github.com/dmist08/Task-Arithmetic.git
cd Task-Arithmetic
```

## Step 1: Create conda environment

```bash
conda create -n task-arithmetic python=3.11 -y
conda activate task-arithmetic

# Install PyTorch with CUDA (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt
```

### Verify GPU access

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

## Step 2: Start Elasticsearch (required for BM25)

The BEIR library's BM25 implementation uses Elasticsearch under the hood.

```bash
# One-time setup
bash Scripts/setup_elasticsearch.sh

# Verify it's running
curl http://localhost:9200
```

If your cluster already has Elasticsearch or doesn't allow installing software, you can use Docker:

```bash
docker run -d --name elasticsearch -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:7.17.9
```

## Step 3: Download datasets only (optional pre-step)

```bash
cd Scripts
python reproduce_roberta_scifact.py --mode download_only
```

This downloads SciFact (~5K docs) and NFCorpus (~3.6K docs) from BEIR.

## Step 4: Run the reproduction

### 4a: α-sweep on dev sets (find best α)

```bash
python reproduce_roberta_scifact.py --mode dev_sweep --device cuda:0
```

This sweeps α from 0.0 to 1.0 in steps of 0.1, evaluating on:
- SciFact train (20% of queries, following the base paper)
- NFCorpus dev

Output: `results/dev_sweep_results.csv`

Expected: best α should be around 0.3 for RoBERTa-base.

### 4b: Test evaluation with chosen α

```bash
python reproduce_roberta_scifact.py --mode test --alfa 0.3 --device cuda:0
```

Output: `results/test_results.csv`

### 4c: Baselines (BM25, Θ_T alone)

```bash
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
