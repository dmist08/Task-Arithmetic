"""
Reproduce Table 1 (RoBERTa-base row) from Braga et al. SIGIR 2025:
"Investigating Task Arithmetic for Zero-Shot Information Retrieval"

Pipeline:
  1. Download SciFact + NFCorpus from BEIR
  2. BM25 first-stage retrieval (top 100) — requires local Elasticsearch
  3. Load three models: Θ₀ (roberta-base), Θ_D (BioMed-RoBERTa), Θ_T (msmarco cross-encoder)
  4. Compute task vector: τ_D = Θ_D − Θ₀
  5. Merge: Θ′ = Θ_T + α·τ_D for α ∈ {0, 0.1, ..., 1.0}
  6. Re-rank BM25 top-100 with Θ′
  7. Fuse BM25 + rerank scores (min-max norm, weighted sum)
  8. Evaluate: P@10, NDCG@3, NDCG@10, MAP@100

Expected results (from Table 1, SciFact):
  BM25:              P@10=.091, NDCG@3=.637, NDCG@10=.691, MAP@100=.649
  Θ_T (α=0):         P@10=.095, NDCG@3=.655, NDCG@10=.707, MAP@100=.662
  Θ′ (α=1):          P@10=.092, NDCG@3=.649, NDCG@10=.700, MAP@100=.659
  Θ′ (α=0.3, best):  P@10=.096, NDCG@3=.669, NDCG@10=.720, MAP@100=.676

Usage:
  # Step 1: Start Elasticsearch (run once)
  bash Scripts/setup_elasticsearch.sh

  # Step 2: Run α-sweep on SciFact dev (to find best α)
  python Scripts/reproduce_roberta_scifact.py --mode dev_sweep --device cuda:0

  # Step 3: Run final evaluation on SciFact test with chosen α
  python Scripts/reproduce_roberta_scifact.py --mode test --alfa 0.3 --device cuda:0

  # Step 4 (optional): Run baselines on SciFact test
  python Scripts/reproduce_roberta_scifact.py --mode baselines --device cuda:0

Model triplet for RoBERTa-base (from Table 1):
  Θ₀ = roberta-base  (Liu et al., 2019)
  Θ_D = allenai/biomed_roberta_base  (BioMed-RoBERTa, Gururangan et al., 2020)
  Θ_T = cross-encoder/stsb-roberta-base  (Reimers & Gurevych, 2019)
       ^^^ SEE NOTE BELOW — this is our best guess; verify against paper.

NOTE ON Θ_T (msmarco-RoBERTa):
  The base paper says "msmarco-RoBERTa" citing Reimers & Gurevych (2019).
  The cross-encoder/ HuggingFace org has NO RoBERTa checkpoint for MS-MARCO.
  Available cross-encoder RoBERTa models:
    - cross-encoder/stsb-roberta-base  (STS-B, not MS-MARCO — but same architecture)
    - cross-encoder/stsb-roberta-large (STS-B, large — wrong size)
  The sentence-transformers/ org has bi-encoder variants (NOT cross-encoders):
    - sentence-transformers/msmarco-roberta-base-ance-firstp (bi-encoder)

  The code's TaskVectorBERT.apply_to() uses beir's CrossEncoder(), which wraps
  AutoModelForSequenceClassification — so Θ_T MUST be a cross-encoder checkpoint.

  Our default: cross-encoder/stsb-roberta-base — same architecture as roberta-base,
  same hidden_size/layers/heads/vocab, is a cross-encoder. The STS-B vs MS-MARCO
  training difference means our scores may differ slightly from the paper.

  If you find the exact msmarco-roberta cross-encoder, pass it via --model_base_path.

  ALTERNATIVE HYPOTHESIS: The authors may have trained their own msmarco-roberta
  cross-encoder using sentence-transformers' cross-encoder training script on MS-MARCO.
  If so, we cannot reproduce without that checkpoint.
"""

import json
import logging
import os
import sys
import time
import pathlib
import csv

import click
import torch
import numpy as np

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from transformers import AutoModel

from ranx import Qrels, Run, compare, fuse

from utils import seed_everything

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Vector for RoBERTa (encoder-only cross-encoder)
# ---------------------------------------------------------------------------
class TaskVectorRoBERTa:
    """
    Computes τ = Θ_D − Θ₀ using AutoModel (encoder-only, no classification head).
    Applies τ to a cross-encoder Θ_T loaded via beir's CrossEncoder wrapper.

    Key mapping: CrossEncoder wraps AutoModelForSequenceClassification, whose
    encoder keys are prefixed with 'roberta.' (e.g. 'roberta.encoder.layer.0...').
    AutoModel keys have no such prefix ('encoder.layer.0...').
    The code strips the 'roberta.' prefix when matching, following the original
    utils.py TaskVectorBERT logic (which strips 'bert.').
    """

    def __init__(self, pretrained_checkpoint, finetuned_checkpoint, device="cuda"):
        with torch.no_grad():
            log_gpu("Before loading Θ₀", device)
            pretrained_sd = AutoModel.from_pretrained(pretrained_checkpoint).state_dict()
            log_gpu("After loading Θ₀", device)

            finetuned_sd = AutoModel.from_pretrained(finetuned_checkpoint).state_dict()
            log_gpu("After loading Θ_D", device)

            self.vector = {}
            for key in pretrained_sd:
                self.vector[key] = finetuned_sd[key].to(device) - pretrained_sd[key].to(device)

            log_gpu("After computing task vector", device)

    def apply_to(self, cross_encoder_path, scaling_coef=1.0, device="cuda"):
        """Load Θ_T as a CrossEncoder, inject α·τ into its encoder weights."""
        with torch.no_grad():
            ce_model = CrossEncoder(cross_encoder_path, max_length=512)
            log_gpu("After loading Θ_T cross-encoder", device)

            ce_state_dict = ce_model.model.model.state_dict()
            new_state_dict = {}

            matched, skipped = 0, 0
            for key in ce_state_dict:
                # CrossEncoder keys: 'roberta.encoder...' or 'roberta.embeddings...'
                # Task vector keys: 'encoder...' or 'embeddings...'
                stripped = key.replace('roberta.', '', 1)
                if stripped in self.vector:
                    new_state_dict[key] = (
                        ce_state_dict[key].to(device)
                        + scaling_coef * self.vector[stripped].to(device)
                    )
                    matched += 1
                else:
                    # Classification head or pooler — keep original weights
                    new_state_dict[key] = ce_state_dict[key].to(device)
                    skipped += 1

            logger.info(f"Task vector applied: {matched} keys merged, {skipped} keys kept as-is (α={scaling_coef})")
            ce_model.model.model.load_state_dict(new_state_dict, strict=False)
            log_gpu(f"After applying task vector (α={scaling_coef})", device)

        return ce_model


# ---------------------------------------------------------------------------
# Reranker (cross-encoder scoring on BM25 top-k)
# ---------------------------------------------------------------------------
class RerankCrossEncoder:
    """Re-rank BM25 results using a beir CrossEncoder model."""

    def __init__(self, model, batch_size=128):
        self.cross_encoder = model
        self.batch_size = batch_size

    def rerank(self, corpus, queries, results, top_k=100):
        sentence_pairs, pair_ids = [], []

        for query_id in results:
            doc_scores = results[query_id]
            if len(doc_scores) > top_k:
                top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            else:
                top_docs = doc_scores.items()

            for doc_id, _ in top_docs:
                pair_ids.append([query_id, doc_id])
                corpus_text = (
                    corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")
                ).strip()
                sentence_pairs.append([queries[query_id], corpus_text])

        logger.info(f"Re-ranking {len(sentence_pairs)} query-doc pairs (top-{top_k})")
        start = time.time()
        raw_scores = self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)
        elapsed = time.time() - start
        logger.info(f"Re-ranking took {elapsed:.1f}s ({len(sentence_pairs)/elapsed:.0f} pairs/sec)")

        rerank_scores = []
        for r in raw_scores:
            try:
                s = r[1]
            except (TypeError, IndexError):
                s = r
            rerank_scores.append(float(s))

        rerank_results = {qid: {} for qid in results}
        for (qid, did), score in zip(pair_ids, rerank_scores):
            rerank_results[qid][did] = score

        return rerank_results


# ---------------------------------------------------------------------------
# GPU memory logging
# ---------------------------------------------------------------------------
def log_gpu(label, device="cuda"):
    if not torch.cuda.is_available():
        return
    try:
        dev_idx = int(str(device).replace("cuda:", "").replace("cuda", "0"))
        alloc = torch.cuda.memory_allocated(dev_idx) / (1024**3)
        reserved = torch.cuda.memory_reserved(dev_idx) / (1024**3)
        logger.info(f"[GPU {dev_idx}] {label}: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dataset downloading
# ---------------------------------------------------------------------------
def download_dataset(name):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    logger.info(f"Dataset '{name}' ready at {data_path}")
    return data_path


# ---------------------------------------------------------------------------
# BM25 first-stage retrieval
# ---------------------------------------------------------------------------
def run_bm25(dataset_name, corpus, queries):
    logger.info(f"Running BM25 on {dataset_name} ({len(corpus)} docs, {len(queries)} queries)")
    start = time.time()
    model = BM25(
        index_name=dataset_name,
        hostname="localhost",
        initialize=True,
        number_of_shards=1,
    )
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)
    logger.info(f"BM25 done in {time.time()-start:.1f}s, retrieved for {len(results)} queries")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@click.command()
@click.option("--mode", type=click.Choice(["dev_sweep", "test", "baselines", "download_only"]),
              required=True, help="What to run")
@click.option("--device", type=str, default="cuda:0")
@click.option("--alfa", type=float, default=1.0,
              help="Scaling factor α (used in 'test' mode; ignored in 'dev_sweep')")
@click.option("--seed", type=int, default=42)
@click.option("--batch_size", type=int, default=128)
@click.option("--model_base_path", type=str, default="cross-encoder/stsb-roberta-base",
              help="Θ_T: IR cross-encoder (see docstring for discussion)")
@click.option("--model_pretrained", type=str, default="roberta-base",
              help="Θ₀: Base pretrained model")
@click.option("--model_domain", type=str, default="allenai/biomed_roberta_base",
              help="Θ_D: Domain-specific model")
@click.option("--output_dir", type=str, default="./results",
              help="Directory for output CSVs and logs")
def main(mode, device, alfa, seed, batch_size, model_base_path, model_pretrained,
         model_domain, output_dir):
    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f"reproduce_roberta_{mode}.log")
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("=" * 70)
    logger.info(f"Mode: {mode}")
    logger.info(f"Θ₀ (pretrained):  {model_pretrained}")
    logger.info(f"Θ_D (domain):      {model_domain}")
    logger.info(f"Θ_T (IR):          {model_base_path}")
    logger.info(f"Device: {device}, Seed: {seed}, Batch size: {batch_size}")
    logger.info("=" * 70)

    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        log_gpu("Session start", device)

    # Download datasets
    scifact_path = download_dataset("scifact")
    nfcorpus_path = download_dataset("nfcorpus")

    if mode == "download_only":
        logger.info("Datasets downloaded. Exiting.")
        return

    # Load SciFact test set (always needed)
    test_corpus, test_queries, test_qrels = GenericDataLoader(scifact_path).load(split="test")
    logger.info(f"SciFact test: {len(test_corpus)} docs, {len(test_queries)} queries")

    # -----------------------------------------------------------------------
    # MODE: dev_sweep — sweep α on SciFact train (20%) + NFCorpus dev
    # -----------------------------------------------------------------------
    if mode == "dev_sweep":
        # SciFact: use 20% of train queries as dev (following base paper)
        sf_corpus, sf_queries, sf_qrels = GenericDataLoader(scifact_path).load(split="train")
        n_dev = max(1, int(0.2 * len(sf_queries)))
        dev_query_ids = list(sf_queries.keys())[:n_dev]
        sf_dev_queries = {q: sf_queries[q] for q in dev_query_ids}
        sf_dev_qrels = {q: sf_qrels[q] for q in dev_query_ids if q in sf_qrels}
        logger.info(f"SciFact dev: using {len(sf_dev_queries)} queries (20% of {len(sf_queries)} train)")

        # NFCorpus dev
        nf_corpus, nf_queries, nf_qrels = GenericDataLoader(nfcorpus_path).load(split="dev")
        logger.info(f"NFCorpus dev: {len(nf_corpus)} docs, {len(nf_queries)} queries")

        # BM25 on both dev sets
        bm25_sf = run_bm25("scifact_dev", sf_corpus, sf_dev_queries)
        bm25_nf = run_bm25("nfcorpus_dev", nf_corpus, nf_queries)

        # Compute task vector once
        logger.info("Computing task vector τ_D = Θ_D − Θ₀ ...")
        tv = TaskVectorRoBERTa(model_pretrained, model_domain, device=device)

        alphas = [round(a * 0.1, 1) for a in range(11)]  # 0.0, 0.1, ..., 1.0
        results_rows = []

        for alpha in alphas:
            logger.info(f"\n{'='*50}\nα = {alpha}\n{'='*50}")
            ce = tv.apply_to(model_base_path, scaling_coef=alpha, device=device)
            reranker = RerankCrossEncoder(ce, batch_size=batch_size)

            for ds_name, corpus, queries, qrels, bm25_results in [
                ("scifact_dev", sf_corpus, sf_dev_queries, sf_dev_qrels, bm25_sf),
                ("nfcorpus_dev", nf_corpus, nf_queries, nf_qrels, bm25_nf),
            ]:
                rerank_res = reranker.rerank(corpus, queries, bm25_results, top_k=100)

                bm25_run = Run(bm25_results, name="BM25")
                rerank_run = Run(rerank_res, name=f"rerank_a{alpha}")

                fused_run = fuse(
                    runs=[bm25_run, rerank_run],
                    norm="min-max",
                    method="wsum",
                    params={"weights": (0.5, 0.5)},
                )
                fused_run.name = f"BM25+rerank_a{alpha}"

                qrels_filtered = {q: qrels[q] for q in bm25_results if q in qrels}
                ranx_qrels = Qrels(qrels_filtered)

                report = compare(
                    qrels=ranx_qrels,
                    runs=[fused_run],
                    metrics=["precision@10", "ndcg@3", "ndcg@10", "map@100"],
                    max_p=0.01,
                )
                logger.info(f"\n{ds_name} α={alpha}:\n{report}")

                from ranx import evaluate
                scores = evaluate(ranx_qrels, fused_run,
                                  ["precision@10", "ndcg@3", "ndcg@10", "map@100"])
                row = {
                    "dataset": ds_name,
                    "alpha": alpha,
                    "P@10": round(scores["precision@10"], 4),
                    "NDCG@3": round(scores["ndcg@3"], 4),
                    "NDCG@10": round(scores["ndcg@10"], 4),
                    "MAP@100": round(scores["map@100"], 4),
                }
                results_rows.append(row)
                logger.info(f"  → {row}")

        # Save CSV
        csv_path = os.path.join(output_dir, "dev_sweep_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "alpha", "P@10", "NDCG@3", "NDCG@10", "MAP@100"])
            writer.writeheader()
            writer.writerows(results_rows)
        logger.info(f"\nDev sweep results saved to {csv_path}")

    # -----------------------------------------------------------------------
    # MODE: test — run on SciFact test with a single α
    # -----------------------------------------------------------------------
    elif mode == "test":
        bm25_test = run_bm25("scifact", test_corpus, test_queries)

        logger.info("Computing task vector τ_D = Θ_D − Θ₀ ...")
        tv = TaskVectorRoBERTa(model_pretrained, model_domain, device=device)

        logger.info(f"Applying task vector with α = {alfa}")
        ce = tv.apply_to(model_base_path, scaling_coef=alfa, device=device)
        reranker = RerankCrossEncoder(ce, batch_size=batch_size)

        rerank_res = reranker.rerank(test_corpus, test_queries, bm25_test, top_k=100)

        bm25_run = Run(bm25_test, name="BM25")
        rerank_run = Run(rerank_res, name=f"TA_a{alfa}")

        fused_run = fuse(
            runs=[bm25_run, rerank_run],
            norm="min-max",
            method="wsum",
            params={"weights": (0.5, 0.5)},
        )
        fused_run.name = f"BM25+TA_a{alfa}"

        qrels_filtered = {q: test_qrels[q] for q in bm25_test if q in test_qrels}
        ranx_qrels = Qrels(qrels_filtered)

        report = compare(
            qrels=ranx_qrels,
            runs=[bm25_run, fused_run],
            metrics=["precision@10", "ndcg@3", "ndcg@10", "map@100"],
            max_p=0.01,
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL RESULTS — SciFact test, α = {alfa}")
        logger.info(f"{'='*60}")
        logger.info(f"\n{report}")
        print(report)

        from ranx import evaluate
        scores = evaluate(ranx_qrels, fused_run,
                          ["precision@10", "ndcg@3", "ndcg@10", "map@100"])
        result = {
            "dataset": "scifact_test",
            "alpha": alfa,
            "P@10": round(scores["precision@10"], 4),
            "NDCG@3": round(scores["ndcg@3"], 4),
            "NDCG@10": round(scores["ndcg@10"], 4),
            "MAP@100": round(scores["map@100"], 4),
        }
        csv_path = os.path.join(output_dir, "test_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            writer.writeheader()
            writer.writerow(result)
        logger.info(f"Results saved to {csv_path}")
        logger.info(f"Result: {result}")

        logger.info("\nExpected (from Table 1):")
        logger.info("  BM25:       P@10=.091, NDCG@3=.637, NDCG@10=.691, MAP@100=.649")
        logger.info(f"  Θ′ (α={alfa}): compare your numbers above against the paper")

    # -----------------------------------------------------------------------
    # MODE: baselines — BM25 only, Θ_T reranker
    # -----------------------------------------------------------------------
    elif mode == "baselines":
        bm25_test = run_bm25("scifact", test_corpus, test_queries)

        qrels_filtered = {q: test_qrels[q] for q in bm25_test if q in test_qrels}
        ranx_qrels = Qrels(qrels_filtered)

        runs_to_compare = []

        bm25_run = Run(bm25_test, name="BM25")
        runs_to_compare.append(bm25_run)

        baselines = [
            ("Theta_T", model_base_path),
        ]

        for name, path in baselines:
            logger.info(f"\nRunning baseline: {name} ({path})")
            ce = CrossEncoder(path, max_length=512)
            reranker = RerankCrossEncoder(ce, batch_size=batch_size)
            rerank_res = reranker.rerank(test_corpus, test_queries, bm25_test, top_k=100)

            rerank_run = Run(rerank_res, name=name)
            fused = fuse(
                runs=[bm25_run, rerank_run],
                norm="min-max",
                method="wsum",
                params={"weights": (0.5, 0.5)},
            )
            fused.name = f"BM25+{name}"
            runs_to_compare.append(fused)

        report = compare(
            qrels=ranx_qrels,
            runs=runs_to_compare,
            metrics=["precision@10", "ndcg@3", "ndcg@10", "map@100"],
            max_p=0.01,
        )
        logger.info(f"\nBaseline results:\n{report}")
        print(report)

    log_gpu("Session end", device)
    logger.info("Done.")


if __name__ == "__main__":
    main()
