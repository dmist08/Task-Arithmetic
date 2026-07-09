"""
Offline reproduction of Table 1 (RoBERTa-base on SciFact) from Braga et al. SIGIR 2025.

This script runs ENTIRELY OFFLINE on gpunode1 (no internet required).
All models and datasets must be pre-downloaded using download_all.py.

BM25 requires Elasticsearch running on localhost:9200.
If Elasticsearch is not available on gpunode1, use --skip_bm25 to load
pre-saved BM25 results (you'll need to run BM25 once on a node that has ES).

Pipeline:
  1. Load SciFact/NFCorpus from local disk
  2. BM25 first-stage retrieval (top 100) OR load cached BM25 results
  3. Compute task vector: τ_D = Θ_D − Θ₀ (all from local model directories)
  4. Merge: Θ′ = Θ_T + α·τ_D
  5. Cross-encoder re-ranking of BM25 top-100
  6. Fuse BM25 + rerank scores → evaluate P@10, NDCG@3, NDCG@10, MAP@100

Usage (on gpunode1):
    # Step 1: α-sweep on dev sets
    python reproduce_roberta_offline.py --cache_dir ../offline_cache --mode dev_sweep --device cuda:0

    # Step 2: Test with best α
    python reproduce_roberta_offline.py --cache_dir ../offline_cache --mode test --alfa 0.3 --device cuda:0

    # Step 3: Baselines
    python reproduce_roberta_offline.py --cache_dir ../offline_cache --mode baselines --device cuda:0

    # If no Elasticsearch on gpunode1 — save BM25 results on login node first:
    python reproduce_roberta_offline.py --cache_dir ../offline_cache --mode save_bm25
    # Then on gpunode1:
    python reproduce_roberta_offline.py --cache_dir ../offline_cache --mode dev_sweep --skip_bm25 --device cuda:0
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

from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer, util
from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from transformers import AutoModel

from ranx import Qrels, Run, compare, fuse, evaluate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import seed_everything

logger = logging.getLogger(__name__)

# Tell HuggingFace to NOT try to reach the internet
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def model_path(cache_dir, model_id):
    """Convert a HF model ID to its local path in offline_cache/models/."""
    safe = model_id.replace("/", "__")
    p = os.path.join(cache_dir, "models", safe)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Model not found at {p}. Run download_all.py first on a node with internet."
        )
    return p


def dataset_path(cache_dir, dataset_name):
    """Get local dataset path."""
    p = os.path.join(cache_dir, "datasets", dataset_name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Dataset not found at {p}. Run download_all.py first on a node with internet."
        )
    return p


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
# Task Vector for RoBERTa (using Bi-Encoder weights)
# ---------------------------------------------------------------------------
class TaskVectorRoBERTa:
    def __init__(self, pretrained_path, finetuned_path, device="cuda"):
        with torch.no_grad():
            log_gpu("Before loading Θ₀", device)
            pretrained_sd = AutoModel.from_pretrained(pretrained_path, local_files_only=True).state_dict()
            log_gpu("After loading Θ₀", device)

            finetuned_sd = AutoModel.from_pretrained(finetuned_path, local_files_only=True).state_dict()
            log_gpu("After loading Θ_D", device)

            self.vector = {}
            for key in pretrained_sd:
                self.vector[key] = finetuned_sd[key].to(device) - pretrained_sd[key].to(device)

            del pretrained_sd, finetuned_sd
            torch.cuda.empty_cache()
            log_gpu("After computing task vector (freed base models)", device)

    def apply_to(self, bi_encoder_path, scaling_coef=1.0, device="cuda"):
        with torch.no_grad():
            # Load the SentenceTransformer bi-encoder model
            model = SentenceTransformer(bi_encoder_path, device=device)
            log_gpu("After loading Θ_T bi-encoder", device)

            model_sd = model.state_dict()
            logger.info(f"First 5 model keys: {list(model_sd.keys())[:5]}")
            logger.info(f"First 5 vector keys: {list(self.vector.keys())[:5]}")
            new_state_dict = {}

            matched, skipped = 0, 0
            for key in model_sd:
                matched_key = None
                if key in self.vector:
                    matched_key = key
                else:
                    for prefix in ["0.auto_model.", "roberta."]:
                        if key.startswith(prefix):
                            stripped = key[len(prefix):]
                            if stripped in self.vector:
                                matched_key = stripped
                                break

                if matched_key is not None:
                    new_state_dict[key] = (
                        model_sd[key].to(device)
                        + scaling_coef * self.vector[matched_key].to(device)
                    )
                    matched += 1
                else:
                    new_state_dict[key] = model_sd[key].to(device)
                    skipped += 1

            logger.info(f"Task vector: {matched} keys merged, {skipped} kept (α={scaling_coef})")
            model.load_state_dict(new_state_dict, strict=False)
            log_gpu(f"After applying task vector (α={scaling_coef})", device)

        return model


# ---------------------------------------------------------------------------
# Bi-encoder reranker
# ---------------------------------------------------------------------------
class RerankCrossEncoder:
    def __init__(self, model, batch_size=128):
        self.model = model
        self.batch_size = batch_size

    def rerank(self, corpus, queries, results, top_k=100):
        """
        Rerank BM25 top-k results using bi-encoder cosine similarity.
        """
        logger.info(f"Re-ranking using bi-encoder ({len(results)} queries)")
        start = time.time()

        rerank_results = {}
        for query_id in results:
            query_text = queries[query_id]
            doc_scores = results[query_id]
            if len(doc_scores) > top_k:
                top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            else:
                top_docs = list(doc_scores.items())

            if not top_docs:
                rerank_results[query_id] = {}
                continue

            doc_ids = [d for d, _ in top_docs]
            doc_texts = [
                (corpus[did].get("title", "") + " " + corpus[did].get("text", "")).strip()
                for did in doc_ids
            ]

            # Encode query and documents
            q_emb = self.model.encode(query_text, convert_to_tensor=True, show_progress_bar=False)
            d_embs = self.model.encode(doc_texts, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False)

            # Compute cosine similarities
            scores = util.cos_sim(q_emb, d_embs)[0].tolist()

            rerank_results[query_id] = {
                doc_ids[i]: float(scores[i])
                for i in range(len(doc_ids))
            }

        elapsed = time.time() - start
        logger.info(f"Re-ranking took {elapsed:.1f}s")
        return rerank_results


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
def run_bm25(index_name, corpus, queries, hostname="localhost", port=9200, top_k=100):
    """
    BM25 retrieval via Elasticsearch — matches the BEIR/paper pipeline.
    Requires Elasticsearch 7.x running on localhost:9200 (start with setup_elasticsearch.sh).
    """
    logger.info(f"Running BM25 via Elasticsearch: {index_name} ({len(corpus)} docs, {len(queries)} queries)")
    start = time.time()

    es = Elasticsearch(f"http://{hostname}:{port}", timeout=120)
    if not es.ping():
        raise ConnectionError(
            f"Cannot connect to Elasticsearch at {hostname}:{port}. "
            "Run Scripts/setup_elasticsearch.sh first."
        )

    # Re-create index fresh
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(
        index=index_name,
        body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {"default": {"type": "BM25"}},
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "text":  {"type": "text"},
                }
            },
        },
    )

    # Bulk index corpus
    actions = [
        {
            "_index": index_name,
            "_id": doc_id,
            "_source": {
                "title": corpus[doc_id].get("title", ""),
                "text":  corpus[doc_id].get("text",  ""),
            },
        }
        for doc_id in corpus
    ]
    es_helpers.bulk(es, actions, chunk_size=500, request_timeout=120)
    es.indices.refresh(index=index_name)
    logger.info(f"Indexed {len(corpus)} docs in {time.time()-start:.1f}s")

    # Query
    results = {}
    for query_id, query_text in queries.items():
        resp = es.search(
            index=index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["title", "text"],
                        "type": "best_fields",
                    }
                },
                "size": top_k,
            },
        )
        results[query_id] = {
            hit["_id"]: float(hit["_score"])
            for hit in resp["hits"]["hits"]
        }

    logger.info(f"BM25 done in {time.time()-start:.1f}s")
    return results


def save_bm25_results(results, path):
    with open(path, 'w') as f:
        json.dump(results, f)
    logger.info(f"BM25 results saved to {path}")


def load_bm25_results(path):
    with open(path, 'r') as f:
        results = json.load(f)
    logger.info(f"BM25 results loaded from {path} ({len(results)} queries)")
    return results


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_run(qrels_dict, bm25_results, rerank_results, run_name, alpha):
    bm25_run = Run(bm25_results, name="BM25")
    rerank_run = Run(rerank_results, name=run_name)

    fused_run = fuse(
        runs=[bm25_run, rerank_run],
        norm="min-max",
        method="wsum",
        params={"weights": (0.5, 0.5)},
    )
    fused_run.name = f"BM25+{run_name}"

    qrels_filtered = {q: qrels_dict[q] for q in bm25_results if q in qrels_dict}
    ranx_qrels = Qrels(qrels_filtered)

    scores = evaluate(ranx_qrels, fused_run,
                      ["precision@10", "ndcg@3", "ndcg@10", "map@100"])

    return {
        "alpha": alpha,
        "P@10": round(scores["precision@10"], 4),
        "NDCG@3": round(scores["ndcg@3"], 4),
        "NDCG@10": round(scores["ndcg@10"], 4),
        "MAP@100": round(scores["map@100"], 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@click.command()
@click.option("--cache_dir", type=str, required=True,
              help="Path to offline_cache/ created by download_all.py")
@click.option("--mode", type=click.Choice(["dev_sweep", "test", "baselines", "save_bm25"]),
              required=True)
@click.option("--device", type=str, default="cuda:0")
@click.option("--alfa", type=float, default=1.0,
              help="α for 'test' mode")
@click.option("--seed", type=int, default=42)
@click.option("--batch_size", type=int, default=128)
@click.option("--skip_bm25", is_flag=True, default=False,
              help="Load pre-cached BM25 results instead of running Elasticsearch")
@click.option("--output_dir", type=str, default="./results")
@click.option("--theta_t", type=str, default="sentence-transformers/msmarco-roberta-base-v3",
              help="HF model ID for Θ_T (must match a downloaded model)")
@click.option("--theta_0", type=str, default="roberta-base",
              help="HF model ID for Θ₀")
@click.option("--theta_d", type=str, default="allenai/biomed_roberta_base",
              help="HF model ID for Θ_D")
def main(cache_dir, mode, device, alfa, seed, batch_size, skip_bm25,
         output_dir, theta_t, theta_0, theta_d):
    seed_everything(seed)
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(output_dir, exist_ok=True)

    bm25_cache_dir = os.path.join(cache_dir, "bm25_cache")
    os.makedirs(bm25_cache_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f"offline_{mode}.log")
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Resolve local paths
    theta_0_path = model_path(cache_dir, theta_0)
    theta_d_path = model_path(cache_dir, theta_d)
    theta_t_path = model_path(cache_dir, theta_t)
    scifact_path = dataset_path(cache_dir, "scifact")
    nfcorpus_path = dataset_path(cache_dir, "nfcorpus")

    logger.info("=" * 60)
    logger.info(f"OFFLINE REPRODUCTION — Mode: {mode}")
    logger.info(f"Cache dir:  {cache_dir}")
    logger.info(f"Θ₀: {theta_0_path}")
    logger.info(f"Θ_D: {theta_d_path}")
    logger.info(f"Θ_T: {theta_t_path}")
    logger.info(f"Device: {device}, Seed: {seed}")
    logger.info(f"Skip BM25 (use cache): {skip_bm25}")
    logger.info("=" * 60)

    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        logger.info(f"GPU: {props.name}, VRAM: {props.total_memory / (1024**3):.1f} GB")
        log_gpu("Session start", device)

    # Helper to get or run BM25
    def get_bm25(index_name, corpus, queries):
        cache_file = os.path.join(bm25_cache_dir, f"bm25_{index_name}.json")
        if skip_bm25:
            if not os.path.exists(cache_file):
                raise FileNotFoundError(
                    f"--skip_bm25 set but no cache at {cache_file}. "
                    f"Run with --mode save_bm25 first on a node with Elasticsearch."
                )
            return load_bm25_results(cache_file)
        else:
            results = run_bm25(index_name, corpus, queries)
            save_bm25_results(results, cache_file)
            return results

    # -----------------------------------------------------------------------
    # MODE: save_bm25 — just run BM25 and save results (for nodes with ES)
    # -----------------------------------------------------------------------
    if mode == "save_bm25":
        logger.info("Saving BM25 results for all splits...")

        # SciFact test
        corpus, queries, _ = GenericDataLoader(scifact_path).load(split="test")
        results = run_bm25("scifact_test", corpus, queries)
        save_bm25_results(results, os.path.join(bm25_cache_dir, "bm25_scifact_test.json"))

        # SciFact dev (20% of train)
        corpus, queries, _ = GenericDataLoader(scifact_path).load(split="train")
        n_dev = max(1, int(0.2 * len(queries)))
        dev_qids = list(queries.keys())[:n_dev]
        dev_queries = {q: queries[q] for q in dev_qids}
        results = run_bm25("scifact_dev", corpus, dev_queries)
        save_bm25_results(results, os.path.join(bm25_cache_dir, "bm25_scifact_dev.json"))

        # NFCorpus dev
        corpus, queries, _ = GenericDataLoader(nfcorpus_path).load(split="dev")
        results = run_bm25("nfcorpus_dev", corpus, queries)
        save_bm25_results(results, os.path.join(bm25_cache_dir, "bm25_nfcorpus_dev.json"))

        logger.info("All BM25 results saved. Transfer offline_cache/ to gpunode1.")
        return

    # -----------------------------------------------------------------------
    # MODE: dev_sweep
    # -----------------------------------------------------------------------
    if mode == "dev_sweep":
        # SciFact dev
        sf_corpus, sf_queries, sf_qrels = GenericDataLoader(scifact_path).load(split="train")
        n_dev = max(1, int(0.2 * len(sf_queries)))
        dev_qids = list(sf_queries.keys())[:n_dev]
        sf_dev_queries = {q: sf_queries[q] for q in dev_qids}
        sf_dev_qrels = {q: sf_qrels[q] for q in dev_qids if q in sf_qrels}
        logger.info(f"SciFact dev: {len(sf_dev_queries)} queries")

        # NFCorpus dev
        nf_corpus, nf_queries, nf_qrels = GenericDataLoader(nfcorpus_path).load(split="dev")
        logger.info(f"NFCorpus dev: {len(nf_corpus)} docs, {len(nf_queries)} queries")

        bm25_sf = get_bm25("scifact_dev", sf_corpus, sf_dev_queries)
        bm25_nf = get_bm25("nfcorpus_dev", nf_corpus, nf_queries)

        logger.info("Computing task vector τ_D = Θ_D − Θ₀ ...")
        tv = TaskVectorRoBERTa(theta_0_path, theta_d_path, device=device)

        alphas = [round(a * 0.1, 1) for a in range(11)]
        all_results = []

        for alpha in alphas:
            logger.info(f"\n{'='*50}\nα = {alpha}\n{'='*50}")
            ce = tv.apply_to(theta_t_path, scaling_coef=alpha, device=device)
            reranker = RerankCrossEncoder(ce, batch_size=batch_size)

            for ds_label, corpus, queries, qrels, bm25_res in [
                ("scifact_dev", sf_corpus, sf_dev_queries, sf_dev_qrels, bm25_sf),
                ("nfcorpus_dev", nf_corpus, nf_queries, nf_qrels, bm25_nf),
            ]:
                rerank_res = reranker.rerank(corpus, queries, bm25_res, top_k=100)
                row = evaluate_run(qrels, bm25_res, rerank_res, f"TA_a{alpha}", alpha)
                row["dataset"] = ds_label
                all_results.append(row)
                logger.info(f"  {ds_label} α={alpha}: {row}")

        csv_path = os.path.join(output_dir, "dev_sweep_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "alpha", "P@10", "NDCG@3", "NDCG@10", "MAP@100"])
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"\nResults saved to {csv_path}")

        # Print summary table
        logger.info("\n" + "=" * 70)
        logger.info("DEV SWEEP SUMMARY")
        logger.info(f"{'Dataset':<15} {'α':>5} {'P@10':>8} {'NDCG@3':>8} {'NDCG@10':>8} {'MAP@100':>8}")
        logger.info("-" * 70)
        for r in all_results:
            logger.info(f"{r['dataset']:<15} {r['alpha']:>5.1f} {r['P@10']:>8.4f} {r['NDCG@3']:>8.4f} {r['NDCG@10']:>8.4f} {r['MAP@100']:>8.4f}")

    # -----------------------------------------------------------------------
    # MODE: test
    # -----------------------------------------------------------------------
    elif mode == "test":
        test_corpus, test_queries, test_qrels = GenericDataLoader(scifact_path).load(split="test")
        logger.info(f"SciFact test: {len(test_corpus)} docs, {len(test_queries)} queries")

        bm25_test = get_bm25("scifact_test", test_corpus, test_queries)

        logger.info("Computing task vector...")
        tv = TaskVectorRoBERTa(theta_0_path, theta_d_path, device=device)

        logger.info(f"Applying with α = {alfa}")
        ce = tv.apply_to(theta_t_path, scaling_coef=alfa, device=device)
        reranker = RerankCrossEncoder(ce, batch_size=batch_size)

        rerank_res = reranker.rerank(test_corpus, test_queries, bm25_test, top_k=100)
        result = evaluate_run(test_qrels, bm25_test, rerank_res, f"TA_a{alfa}", alfa)
        result["dataset"] = "scifact_test"

        # Also evaluate BM25 alone
        bm25_qrels = {q: test_qrels[q] for q in bm25_test if q in test_qrels}
        ranx_qrels = Qrels(bm25_qrels)
        bm25_run = Run(bm25_test, name="BM25")
        bm25_scores = evaluate(ranx_qrels, bm25_run,
                               ["precision@10", "ndcg@3", "ndcg@10", "map@100"])

        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS — SciFact test")
        logger.info("=" * 60)
        logger.info(f"{'Variant':<20} {'P@10':>8} {'NDCG@3':>8} {'NDCG@10':>8} {'MAP@100':>8}")
        logger.info("-" * 60)
        logger.info(f"{'BM25':<20} {bm25_scores['precision@10']:>8.4f} {bm25_scores['ndcg@3']:>8.4f} {bm25_scores['ndcg@10']:>8.4f} {bm25_scores['map@100']:>8.4f}")
        logger.info(f"{'Θ′ (α='+str(alfa)+')':<20} {result['P@10']:>8.4f} {result['NDCG@3']:>8.4f} {result['NDCG@10']:>8.4f} {result['MAP@100']:>8.4f}")
        logger.info("")
        logger.info("Expected from Table 1:")
        logger.info(f"{'BM25 (paper)':<20} {'0.0910':>8} {'0.6370':>8} {'0.6910':>8} {'0.6490':>8}")
        logger.info(f"{'Θ′ α=0.3 (paper)':<20} {'0.0960':>8} {'0.6690':>8} {'0.7200':>8} {'0.6760':>8}")

        csv_path = os.path.join(output_dir, "test_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "alpha", "P@10", "NDCG@3", "NDCG@10", "MAP@100"])
            writer.writeheader()
            writer.writerow(result)
        logger.info(f"\nSaved to {csv_path}")

    # -----------------------------------------------------------------------
    # MODE: baselines
    # -----------------------------------------------------------------------
    elif mode == "baselines":
        test_corpus, test_queries, test_qrels = GenericDataLoader(scifact_path).load(split="test")
        bm25_test = get_bm25("scifact_test", test_corpus, test_queries)

        bm25_qrels = {q: test_qrels[q] for q in bm25_test if q in test_qrels}
        ranx_qrels = Qrels(bm25_qrels)

        # BM25
        bm25_run = Run(bm25_test, name="BM25")
        bm25_scores = evaluate(ranx_qrels, bm25_run,
                               ["precision@10", "ndcg@3", "ndcg@10", "map@100"])

        # Θ_T alone (α=0 → just the IR model)
        ce = SentenceTransformer(theta_t_path, device=device)
        reranker = RerankCrossEncoder(ce, batch_size=batch_size)
        rerank_res = reranker.rerank(test_corpus, test_queries, bm25_test, top_k=100)
        theta_t_result = evaluate_run(test_qrels, bm25_test, rerank_res, "Theta_T", 0.0)

        logger.info("\n" + "=" * 60)
        logger.info("BASELINE RESULTS — SciFact test")
        logger.info("=" * 60)
        logger.info(f"{'Variant':<20} {'P@10':>8} {'NDCG@3':>8} {'NDCG@10':>8} {'MAP@100':>8}")
        logger.info("-" * 60)
        logger.info(f"{'BM25':<20} {bm25_scores['precision@10']:>8.4f} {bm25_scores['ndcg@3']:>8.4f} {bm25_scores['ndcg@10']:>8.4f} {bm25_scores['map@100']:>8.4f}")
        logger.info(f"{'BM25+Θ_T':<20} {theta_t_result['P@10']:>8.4f} {theta_t_result['NDCG@3']:>8.4f} {theta_t_result['NDCG@10']:>8.4f} {theta_t_result['MAP@100']:>8.4f}")

    log_gpu("Session end", device)
    logger.info("Done.")


if __name__ == "__main__":
    main()
