"""
Offline reproduction of Table 2 (MT5-base multilingual) from Braga et al. SIGIR 2025.

Matches the paper's pipeline exactly:
  - TaskVectorT5 via AutoModelForSeq2SeqLM (same as paper's utils.py)
  - MonoT5 scoring with '▁yes'/'▁no' tokens (same as paper's MIRACL script)
  - BM25 via Elasticsearch with language-specific analyzers
  - Score fusion via ranx (min-max norm, weighted sum λ=0.5/0.5)

Usage (on gpunode1):
    # GermanQuAD with task arithmetic (α=1.0, fully zero-shot)
    python Scripts/reproduce_mt5_offline.py \\
        --cache_dir ../offline_cache \\
        --dataset germanquad \\
        --mode test --alfa 1.0 --device cuda:0

    # Baselines only (BM25, Θ_T alone)
    python Scripts/reproduce_mt5_offline.py \\
        --cache_dir ../offline_cache \\
        --dataset germanquad \\
        --mode baselines --device cuda:0

    # α-sweep (if you want to find optimal α)
    python Scripts/reproduce_mt5_offline.py \\
        --cache_dir ../offline_cache \\
        --dataset germanquad \\
        --mode dev_sweep --device cuda:0

    # Save BM25 results (on node with Elasticsearch)
    python Scripts/reproduce_mt5_offline.py \\
        --cache_dir ../offline_cache \\
        --dataset germanquad \\
        --mode save_bm25
"""

import json
import logging
import os
import sys
import time
import csv
import random

import click
import torch
import numpy as np

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from beir.datasets.data_loader import GenericDataLoader
from ranx import Qrels, Run, fuse, evaluate

logger = logging.getLogger(__name__)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def seed_everything(seed: int):
    """Matches paper's utils.py seed_everything."""
    logger.info(f"Setting global random seed to {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Constants — model and dataset mappings (matches paper Section 3.2)
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "germanquad": {
        "theta_d": "airKlizz/mt5-base-wikinewssum-german",
        "language": "german",
        "split": "test",
    },
    "miracl_fr": {
        "theta_d": "airKlizz/mt5-base-wikinewssum-french",
        "language": "french",
        "split": "dev",
    },
    "miracl_es": {
        "theta_d": "airKlizz/mt5-base-wikinewssum-spanish",
        "language": "spanish",
        "split": "dev",
    },
    "miracl_en": {
        "theta_d": "airKlizz/mt5-base-wikinewssum-english",
        "language": "english",
        "split": "dev",
    },
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def model_path(cache_dir, model_id):
    safe = model_id.replace("/", "__")
    p = os.path.join(cache_dir, "models", safe)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model not found at {p}. Run download_mt5.py first.")
    return p


def dataset_path(cache_dir, dataset_name):
    p = os.path.join(cache_dir, "datasets", dataset_name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Dataset not found at {p}. Run download_mt5.py first.")
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
# Task Vector for T5/MT5 — matches paper's TaskVectorT5 in utils.py
# ---------------------------------------------------------------------------
class TaskVectorT5:
    """
    Computes τ = Θ_D - Θ₀ using AutoModelForSeq2SeqLM state dicts.
    Applies to Θ_T: Θ' = Θ_T + α·τ
    """

    def __init__(self, pretrained_path, finetuned_path, device="cuda"):
        with torch.no_grad():
            log_gpu("Before loading Θ₀", device)
            pretrained_sd = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_path, local_files_only=True
            ).state_dict()
            log_gpu("After loading Θ₀", device)

            finetuned_sd = AutoModelForSeq2SeqLM.from_pretrained(
                finetuned_path, local_files_only=True
            ).state_dict()
            log_gpu("After loading Θ_D", device)

            self.vector = {}
            for key in pretrained_sd:
                if key in finetuned_sd:
                    self.vector[key] = (
                        finetuned_sd[key].to(device) - pretrained_sd[key].to(device)
                    )

            del pretrained_sd, finetuned_sd
            torch.cuda.empty_cache()
            log_gpu("After computing task vector", device)

    def apply_to(self, theta_t_path, scaling_coef=1.0, device="cuda"):
        """
        Load Θ_T as AutoModelForSeq2SeqLM, apply task vector, return (model, state_dict).
        This matches the paper's TaskVectorT5.apply_to() exactly.
        """
        with torch.no_grad():
            sum_model = AutoModelForSeq2SeqLM.from_pretrained(
                theta_t_path, local_files_only=True
            )
            log_gpu("After loading Θ_T", device)

            pretrained_sd = sum_model.state_dict()
            new_state_dict = {}
            matched, skipped = 0, 0

            for key in pretrained_sd:
                if key in self.vector:
                    new_state_dict[key] = (
                        pretrained_sd[key].to(device)
                        + scaling_coef * self.vector[key].to(device)
                    )
                    matched += 1
                else:
                    new_state_dict[key] = pretrained_sd[key].to(device)
                    skipped += 1

            logger.info(f"Task vector: {matched} keys merged, {skipped} kept (α={scaling_coef})")
            sum_model.load_state_dict(new_state_dict, strict=False)
            sum_model.to(device)
            log_gpu(f"After applying task vector (α={scaling_coef})", device)

        return sum_model, new_state_dict


# ---------------------------------------------------------------------------
# MonoT5 Reranker — matches paper's usage of monot5.MonoT5 + RerankT5
# ---------------------------------------------------------------------------
class MonoT5Reranker:
    """
    MonoT5 cross-encoder scoring:
      Input:  "Query: {q} Document: {d} Relevant:"
      Output: log P("▁yes") - log P("▁no")

    Uses the monot5 package if available, otherwise falls back to
    manual implementation via transformers.
    """

    def __init__(self, model, tokenizer, device="cuda",
                 token_true="▁yes", token_false="▁no", batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

        # Get token IDs for true/false tokens
        self.true_id = self.tokenizer.convert_tokens_to_ids(token_true)
        self.false_id = self.tokenizer.convert_tokens_to_ids(token_false)
        logger.info(f"MonoT5 tokens: '{token_true}'={self.true_id}, '{token_false}'={self.false_id}")

        if self.true_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Token '{token_true}' not in vocabulary!")
        if self.false_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Token '{token_false}' not in vocabulary!")

    def score_pairs(self, pairs):
        """Score a list of (query, document) pairs. Returns list of float scores."""
        scores = []
        self.model.eval()

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            # Format: "Query: {q} Document: {d} Relevant:"
            inputs_text = [
                f"Query: {q} Document: {d} Relevant:" for q, d in batch
            ]

            inputs = self.tokenizer(
                inputs_text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                # Decode with a single step: generate first token logits
                decoder_input_ids = torch.zeros(
                    (len(batch), 1), dtype=torch.long, device=self.device
                )
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                logits = outputs.logits[:, 0, :]  # First token logits

                # Score = log P(true) - log P(false)
                true_logits = logits[:, self.true_id]
                false_logits = logits[:, self.false_id]
                batch_scores = (true_logits - false_logits).cpu().tolist()
                scores.extend(batch_scores)

        return scores

    def rerank(self, corpus, queries, results, top_k=100):
        """Rerank BM25 results using MonoT5 scoring."""
        sentence_pairs = []
        pair_ids = []

        for query_id in results:
            doc_scores = results[query_id]
            if len(doc_scores) > top_k:
                top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            else:
                top_docs = list(doc_scores.items())

            for doc_id, _ in top_docs:
                pair_ids.append((query_id, doc_id))
                corpus_text = (
                    corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")
                ).strip()
                sentence_pairs.append((queries[query_id], corpus_text))

        logger.info(f"Re-ranking {len(sentence_pairs)} pairs (top-{top_k})")
        start = time.time()
        scores = self.score_pairs(sentence_pairs)
        elapsed = time.time() - start
        logger.info(f"Re-ranking took {elapsed:.1f}s ({len(sentence_pairs)/max(elapsed,0.01):.0f} pairs/sec)")

        rerank_results = {qid: {} for qid in results}
        for (qid, did), score in zip(pair_ids, scores):
            rerank_results[qid][did] = score

        return rerank_results


# ---------------------------------------------------------------------------
# BM25 via Elasticsearch (with language-specific analyzers)
# ---------------------------------------------------------------------------
def run_bm25(index_name, corpus, queries, language=None, hostname="localhost", port=9200, top_k=100):
    """
    BM25 retrieval via Elasticsearch.
    Uses language-specific analyzer for German/French/Spanish.
    """
    logger.info(f"Running BM25 via Elasticsearch: {index_name} ({len(corpus)} docs, {len(queries)} queries, lang={language})")
    start = time.time()

    es = Elasticsearch(f"http://{hostname}:{port}", timeout=120)
    if not es.ping():
        raise ConnectionError(
            f"Cannot connect to Elasticsearch at {hostname}:{port}. "
            "Run Scripts/setup_elasticsearch.sh first."
        )

    # Re-create index
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # Language-specific analyzer settings
    index_body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "similarity": {"default": {"type": "BM25"}},
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "text": {"type": "text"},
            }
        },
    }

    # Add language analyzer if specified
    if language and language != "english":
        index_body["settings"]["analysis"] = {
            "analyzer": {
                "lang_analyzer": {
                    "type": language,  # ES built-in: german, french, spanish
                }
            }
        }
        index_body["mappings"]["properties"]["title"]["analyzer"] = "lang_analyzer"
        index_body["mappings"]["properties"]["text"]["analyzer"] = "lang_analyzer"

    es.indices.create(index=index_name, body=index_body)

    # Bulk index
    actions = [
        {
            "_index": index_name,
            "_id": doc_id,
            "_source": {
                "title": corpus[doc_id].get("title", ""),
                "text": corpus[doc_id].get("text", ""),
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
            hit["_id"]: float(hit["_score"]) for hit in resp["hits"]["hits"]
        }

    logger.info(f"BM25 done in {time.time()-start:.1f}s")
    return results


def save_bm25_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f)
    logger.info(f"BM25 results saved to {path}")


def load_bm25_results(path):
    with open(path, "r") as f:
        results = json.load(f)
    logger.info(f"BM25 results loaded from {path} ({len(results)} queries)")
    return results


# ---------------------------------------------------------------------------
# Evaluation
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

    scores = evaluate(
        ranx_qrels, fused_run, ["precision@10", "ndcg@3", "ndcg@10", "map@100"]
    )

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
              help="Path to offline_cache/ created by download_mt5.py")
@click.option("--dataset", type=click.Choice(list(DATASET_CONFIG.keys())), required=True,
              help="Dataset to evaluate on")
@click.option("--mode", type=click.Choice(["test", "baselines", "dev_sweep", "save_bm25"]),
              required=True)
@click.option("--device", type=str, default="cuda:0")
@click.option("--alfa", type=float, default=1.0,
              help="α for 'test' mode (paper uses α=1.0 for multilingual)")
@click.option("--seed", type=int, default=42)
@click.option("--batch_size", type=int, default=32,
              help="MonoT5 batch size (lower than CE due to seq2seq memory)")
@click.option("--skip_bm25", is_flag=True, default=False,
              help="Load pre-cached BM25 results")
@click.option("--output_dir", type=str, default="./results")
def main(cache_dir, dataset, mode, device, alfa, seed, batch_size, skip_bm25, output_dir):
    seed_everything(seed)
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(output_dir, exist_ok=True)

    bm25_cache_dir = os.path.join(cache_dir, "bm25_cache")
    os.makedirs(bm25_cache_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f"mt5_{dataset}_{mode}.log")
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # Suppress verbose HTTP request logs from elasticsearch and urllib3
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Config for this dataset
    ds_cfg = DATASET_CONFIG[dataset]
    theta_0_id = "google/mt5-base"
    theta_t_id = "unicamp-dl/mt5-base-mmarco-v2"
    theta_d_id = ds_cfg["theta_d"]
    language = ds_cfg["language"]
    split = ds_cfg["split"]

    theta_0_path = model_path(cache_dir, theta_0_id)
    theta_t_path = model_path(cache_dir, theta_t_id)
    theta_d_path = model_path(cache_dir, theta_d_id)
    ds_path = dataset_path(cache_dir, dataset)

    logger.info("=" * 60)
    logger.info(f"MT5-base REPRODUCTION — {dataset} / {mode}")
    logger.info(f"Θ₀: {theta_0_id}")
    logger.info(f"Θ_T: {theta_t_id}")
    logger.info(f"Θ_D: {theta_d_id}")
    logger.info(f"Language: {language}, Split: {split}")
    logger.info(f"Device: {device}, Seed: {seed}")
    logger.info("=" * 60)

    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        logger.info(f"GPU: {props.name}, VRAM: {props.total_memory / (1024**3):.1f} GB")
        log_gpu("Session start", device)

    # Load dataset
    corpus, queries, qrels = GenericDataLoader(ds_path).load(split=split)
    logger.info(f"Dataset: {len(corpus)} docs, {len(queries)} queries")

    # BM25 helper
    def get_bm25():
        cache_file = os.path.join(bm25_cache_dir, f"bm25_{dataset}_{split}.json")
        if skip_bm25:
            if not os.path.exists(cache_file):
                raise FileNotFoundError(
                    f"--skip_bm25 set but no cache at {cache_file}. "
                    f"Run --mode save_bm25 first."
                )
            return load_bm25_results(cache_file)
        else:
            results = run_bm25(dataset, corpus, queries, language=language)
            save_bm25_results(results, cache_file)
            return results

    # MonoT5 helper
    def create_reranker(model_path_or_model, device=device):
        """Create MonoT5 reranker from model path or pre-loaded model."""
        if isinstance(model_path_or_model, str):
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path_or_model, local_files_only=True
            )
            model.to(device)
        else:
            model = model_path_or_model

        tokenizer = AutoTokenizer.from_pretrained(theta_t_path, local_files_only=True)
        return MonoT5Reranker(
            model, tokenizer, device=device,
            token_true="▁yes", token_false="▁no",
            batch_size=batch_size,
        )

    # -------------------------------------------------------------------
    # MODE: save_bm25
    # -------------------------------------------------------------------
    if mode == "save_bm25":
        logger.info(f"Saving BM25 results for {dataset} ({split})...")
        bm25_results = run_bm25(dataset, corpus, queries, language=language)
        save_bm25_results(
            bm25_results,
            os.path.join(bm25_cache_dir, f"bm25_{dataset}_{split}.json"),
        )
        logger.info("Done. Transfer offline_cache/ to gpunode1.")
        return

    # -------------------------------------------------------------------
    # MODE: baselines
    # -------------------------------------------------------------------
    if mode == "baselines":
        bm25_results = get_bm25()

        # BM25 alone
        bm25_qrels = {q: qrels[q] for q in bm25_results if q in qrels}
        ranx_qrels = Qrels(bm25_qrels)
        bm25_run = Run(bm25_results, name="BM25")
        bm25_scores = evaluate(
            ranx_qrels, bm25_run, ["precision@10", "ndcg@3", "ndcg@10", "map@100"]
        )

        # Θ_T alone (MonoT5 without task arithmetic)
        logger.info("Loading Θ_T (MonoT5 baseline)...")
        reranker = create_reranker(theta_t_path)
        rerank_res = reranker.rerank(corpus, queries, bm25_results, top_k=100)
        theta_t_result = evaluate_run(qrels, bm25_results, rerank_res, "Theta_T", 0.0)

        logger.info("\n" + "=" * 60)
        logger.info(f"BASELINES — {dataset}")
        logger.info("=" * 60)
        logger.info(f"{'Variant':<20} {'P@10':>8} {'NDCG@3':>8} {'NDCG@10':>8} {'MAP@100':>8}")
        logger.info("-" * 60)
        logger.info(f"{'BM25':<20} {bm25_scores['precision@10']:>8.4f} {bm25_scores['ndcg@3']:>8.4f} {bm25_scores['ndcg@10']:>8.4f} {bm25_scores['map@100']:>8.4f}")
        logger.info(f"{'BM25+Θ_T':<20} {theta_t_result['P@10']:>8.4f} {theta_t_result['NDCG@3']:>8.4f} {theta_t_result['NDCG@10']:>8.4f} {theta_t_result['MAP@100']:>8.4f}")

        if dataset == "germanquad":
            logger.info("")
            logger.info("Expected from Table 2:")
            logger.info(f"{'BM25 (paper)':>20} {'0.0463':>8} {'0.2613':>8} {'0.2760':>8} {'0.2050':>8}")
            logger.info(f"{'Θ_T (paper)':>20} {'0.0496':>8} {'0.2999':>8} {'0.3040':>8} {'0.2410':>8}")

    # -------------------------------------------------------------------
    # MODE: test
    # -------------------------------------------------------------------
    elif mode == "test":
        bm25_results = get_bm25()

        logger.info("Computing task vector τ = Θ_D − Θ₀ ...")
        tv = TaskVectorT5(theta_0_path, theta_d_path, device=device)

        logger.info(f"Applying task vector with α = {alfa}")
        sum_model, new_state_dict = tv.apply_to(theta_t_path, scaling_coef=alfa, device=device)

        # Create MonoT5 reranker with the modified model
        reranker = create_reranker(sum_model, device=device)
        rerank_res = reranker.rerank(corpus, queries, bm25_results, top_k=100)
        result = evaluate_run(qrels, bm25_results, rerank_res, f"TA_a{alfa}", alfa)
        result["dataset"] = dataset

        # BM25 baseline
        bm25_qrels = {q: qrels[q] for q in bm25_results if q in qrels}
        ranx_qrels = Qrels(bm25_qrels)
        bm25_run = Run(bm25_results, name="BM25")
        bm25_scores = evaluate(
            ranx_qrels, bm25_run, ["precision@10", "ndcg@3", "ndcg@10", "map@100"]
        )

        logger.info("\n" + "=" * 60)
        logger.info(f"RESULTS — {dataset} (α={alfa})")
        logger.info("=" * 60)
        logger.info(f"{'Variant':<20} {'P@10':>8} {'NDCG@3':>8} {'NDCG@10':>8} {'MAP@100':>8}")
        logger.info("-" * 60)
        logger.info(f"{'BM25':<20} {bm25_scores['precision@10']:>8.4f} {bm25_scores['ndcg@3']:>8.4f} {bm25_scores['ndcg@10']:>8.4f} {bm25_scores['map@100']:>8.4f}")
        alpha_label = f"Θ′ (α={alfa})"
        logger.info(f"{alpha_label:<20} {result['P@10']:>8.4f} {result['NDCG@3']:>8.4f} {result['NDCG@10']:>8.4f} {result['MAP@100']:>8.4f}")

        if dataset == "germanquad":
            logger.info("")
            logger.info("Expected from Table 2:")
            logger.info(f"{'BM25 (paper)':<20} {'0.0463':>8} {'0.2613':>8} {'0.2760':>8} {'0.2050':>8}")
            logger.info(f"{'Θ′ α=1.0 (paper)':<20} {'0.0570':>8} {'0.3561':>8} {'0.3590':>8} {'0.2960':>8}")

        csv_path = os.path.join(output_dir, f"mt5_{dataset}_test.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "alpha", "P@10", "NDCG@3", "NDCG@10", "MAP@100"])
            writer.writeheader()
            writer.writerow(result)
        logger.info(f"\nSaved to {csv_path}")

    # -------------------------------------------------------------------
    # MODE: dev_sweep
    # -------------------------------------------------------------------
    elif mode == "dev_sweep":
        bm25_results = get_bm25()

        logger.info("Computing task vector τ = Θ_D − Θ₀ ...")
        tv = TaskVectorT5(theta_0_path, theta_d_path, device=device)

        alphas = [round(a * 0.1, 1) for a in range(11)]
        all_results = []

        for alpha in alphas:
            logger.info(f"\n{'='*50}\nα = {alpha}\n{'='*50}")
            sum_model, new_state_dict = tv.apply_to(theta_t_path, scaling_coef=alpha, device=device)
            reranker = create_reranker(sum_model, device=device)

            rerank_res = reranker.rerank(corpus, queries, bm25_results, top_k=100)
            row = evaluate_run(qrels, bm25_results, rerank_res, f"TA_a{alpha}", alpha)
            row["dataset"] = dataset
            all_results.append(row)
            logger.info(f"  {dataset} α={alpha}: {row}")

            # Free model memory
            del sum_model, reranker
            torch.cuda.empty_cache()

        csv_path = os.path.join(output_dir, f"mt5_{dataset}_sweep.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "alpha", "P@10", "NDCG@3", "NDCG@10", "MAP@100"])
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"\nResults saved to {csv_path}")

        # Summary table
        logger.info("\n" + "=" * 70)
        logger.info(f"SWEEP SUMMARY — {dataset}")
        logger.info(f"{'α':>5} {'P@10':>8} {'NDCG@3':>8} {'NDCG@10':>8} {'MAP@100':>8}")
        logger.info("-" * 70)
        for r in all_results:
            logger.info(f"{r['alpha']:>5.1f} {r['P@10']:>8.4f} {r['NDCG@3']:>8.4f} {r['NDCG@10']:>8.4f} {r['MAP@100']:>8.4f}")

    log_gpu("Session end", device)
    logger.info("Done.")


if __name__ == "__main__":
    main()
