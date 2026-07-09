"""
Download all models and datasets for MT5-base reproduction (Table 2).

Run on a node with internet access:
    python Scripts/download_mt5.py --cache_dir ./offline_cache

Then rsync offline_cache/ to gpunode1 for offline execution.

Models downloaded (~13 GB total):
  - google/mt5-base              (Θ₀)
  - unicamp-dl/mt5-base-mmarco-v2 (Θ_T — IR model)
  - airKlizz/mt5-base-wikinewssum-german  (Θ_D German)
  - airKlizz/mt5-base-wikinewssum-french  (Θ_D French)
  - airKlizz/mt5-base-wikinewssum-spanish (Θ_D Spanish)
  - airKlizz/mt5-base-wikinewssum-english (Θ_D English)

Datasets:
  - GermanQuAD (converted to BEIR format)
  - MIRACL (French, Spanish, English dev splits)
"""

import os
import json
import argparse
from pathlib import Path


def download_model(model_id, cache_dir):
    """Download a HuggingFace model to offline_cache/models/<safe_name>/."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    safe = model_id.replace("/", "__")
    out_dir = os.path.join(cache_dir, "models", safe)

    if os.path.exists(out_dir) and any(
        f.endswith((".bin", ".safetensors")) for f in os.listdir(out_dir)
    ):
        print(f"  [SKIP] {model_id} already at {out_dir}")
        return out_dir

    print(f"  [DOWNLOAD] {model_id} → {out_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print(f"  [OK] {model_id} saved ({sum(f.stat().st_size for f in Path(out_dir).rglob('*') if f.is_file()) / 1e9:.1f} GB)")
    return out_dir


def download_germanquad(cache_dir):
    """Download GermanQuAD and convert to BEIR format."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    out_dir = os.path.join(cache_dir, "datasets", "germanquad")
    qrels_dir = os.path.join(out_dir, "qrels")

    if os.path.exists(os.path.join(out_dir, "corpus.jsonl")):
        print("  [SKIP] GermanQuAD already converted")
        return out_dir

    print("  [DOWNLOAD] deepset/germanquad (via parquet) ...")
    # Download the raw parquet file directly — avoids broken dataset script
    parquet_path = hf_hub_download(
        repo_id="deepset/germanquad",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    ds = pd.read_parquet(parquet_path)

    os.makedirs(qrels_dir, exist_ok=True)

    # Build corpus from unique contexts and queries from questions
    corpus = {}   # doc_id → {"title": "", "text": context}
    queries = {}  # q_id → question
    qrels = {}    # q_id → {doc_id: relevance}

    context_to_id = {}  # dedup contexts
    doc_counter = 0

    for i, row in ds.iterrows():
        context = str(row["context"]).strip()
        question = str(row["question"]).strip()

        # Assign doc_id for this context (dedup)
        if context not in context_to_id:
            doc_id = f"doc_{doc_counter}"
            context_to_id[context] = doc_id
            corpus[doc_id] = {"title": "", "text": context}
            doc_counter += 1
        else:
            doc_id = context_to_id[context]

        q_id = f"q_{i}"
        queries[q_id] = question
        qrels[q_id] = {doc_id: 1}

    # Write BEIR format
    with open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for doc_id, doc in corpus.items():
            f.write(json.dumps({"_id": doc_id, **doc}, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
        for q_id, text in queries.items():
            f.write(json.dumps({"_id": q_id, "text": text}, ensure_ascii=False) + "\n")

    with open(os.path.join(qrels_dir, "test.tsv"), "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q_id, rels in qrels.items():
            for doc_id, score in rels.items():
                f.write(f"{q_id}\t{doc_id}\t{score}\n")

    print(f"  [OK] GermanQuAD: {len(corpus)} docs, {len(queries)} queries → {out_dir}")
    return out_dir


def download_miracl(cache_dir, languages=("fr", "es", "en")):
    """Download MIRACL dev splits and convert to BEIR format."""
    from datasets import load_dataset

    for lang in languages:
        out_dir = os.path.join(cache_dir, "datasets", f"miracl_{lang}")
        qrels_dir = os.path.join(out_dir, "qrels")

        if os.path.exists(os.path.join(out_dir, "corpus.jsonl")):
            print(f"  [SKIP] MIRACL-{lang} already converted")
            continue

        print(f"  [DOWNLOAD] miracl/miracl ({lang}) ...")
        try:
            # Load corpus
            corpus_ds = load_dataset("miracl/miracl-corpus", lang, split="train")
            # Load dev queries + qrels
            dev_ds = load_dataset("miracl/miracl", lang, split="dev")
        except Exception as e:
            print(f"  [WARN] Could not download MIRACL-{lang}: {e}")
            print(f"         Try: pip install datasets[streaming]")
            continue

        os.makedirs(qrels_dir, exist_ok=True)

        # Write corpus
        corpus = {}
        with open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
            for row in corpus_ds:
                doc_id = str(row["docid"])
                title = row.get("title", "")
                text = row.get("text", "")
                corpus[doc_id] = True
                f.write(json.dumps({"_id": doc_id, "title": title, "text": text}, ensure_ascii=False) + "\n")

        # Write queries and qrels
        queries = {}
        qrels = {}
        for row in dev_ds:
            q_id = str(row["query_id"])
            queries[q_id] = row["query"]
            qrels[q_id] = {}
            for pos in row.get("positive_passages", []):
                qrels[q_id][str(pos["docid"])] = 1
            for neg in row.get("negative_passages", []):
                qrels[q_id][str(neg["docid"])] = 0

        with open(os.path.join(out_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
            for q_id, text in queries.items():
                f.write(json.dumps({"_id": q_id, "text": text}, ensure_ascii=False) + "\n")

        with open(os.path.join(qrels_dir, "dev.tsv"), "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for q_id, rels in qrels.items():
                for doc_id, score in rels.items():
                    if score > 0:
                        f.write(f"{q_id}\t{doc_id}\t{score}\n")

        print(f"  [OK] MIRACL-{lang}: {len(corpus)} docs, {len(queries)} queries → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download MT5-base models and datasets")
    parser.add_argument("--cache_dir", type=str, default="./offline_cache",
                        help="Root directory for downloaded models and datasets")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Phase 1: GermanQuAD only (3 models). Phase 2: + MIRACL (6 models)")
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(os.path.join(cache_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "datasets"), exist_ok=True)

    print("=" * 60)
    print(f"MT5-base Model & Dataset Downloader (Phase {args.phase})")
    print(f"Cache dir: {cache_dir}")
    print("=" * 60)

    # === Models ===
    print("\n--- Models ---")

    # Always needed
    models_phase1 = [
        "google/mt5-base",                            # Θ₀
        "unicamp-dl/mt5-base-mmarco-v2",              # Θ_T (IR model)
        "airKlizz/mt5-base-wikinewssum-german",       # Θ_D German
    ]

    models_phase2 = [
        "airKlizz/mt5-base-wikinewssum-french",       # Θ_D French
        "airKlizz/mt5-base-wikinewssum-spanish",      # Θ_D Spanish
        "airKlizz/mt5-base-wikinewssum-english",      # Θ_D English
    ]

    models_to_download = models_phase1 + (models_phase2 if args.phase >= 2 else [])

    for model_id in models_to_download:
        download_model(model_id, cache_dir)

    # === Datasets ===
    print("\n--- Datasets ---")

    download_germanquad(cache_dir)

    if args.phase >= 2:
        download_miracl(cache_dir)

    print("\n" + "=" * 60)
    print("Done! Transfer offline_cache/ to gpunode1:")
    print(f"  rsync -avP {cache_dir}/ gpunode1:~/dharmik_workspace/Task-Arithmetic/offline_cache/")
    print("=" * 60)


if __name__ == "__main__":
    main()
