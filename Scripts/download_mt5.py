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
    """Download GermanQuAD (official MTEB version) and convert to BEIR format."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    out_dir = os.path.join(cache_dir, "datasets", "germanquad")
    qrels_dir = os.path.join(out_dir, "qrels")

    if os.path.exists(os.path.join(out_dir, "corpus.jsonl")):
        print("  [SKIP] GermanQuAD already converted")
        return out_dir

    print("  [DOWNLOAD] mteb/germanquad-retrieval (via parquet) ...")

    # 1. Download Corpus
    corpus_path = hf_hub_download(
        repo_id="mteb/germanquad-retrieval",
        filename="corpus/corpus/0000.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    df_corpus = pd.read_parquet(corpus_path)

    # 2. Download Queries
    queries_path = hf_hub_download(
        repo_id="mteb/germanquad-retrieval",
        filename="queries/queries/0000.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    df_queries = pd.read_parquet(queries_path)

    # 3. Download Qrels
    qrels_path = hf_hub_download(
        repo_id="mteb/germanquad-retrieval-qrels",
        filename="default/test/0000.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    df_qrels = pd.read_parquet(qrels_path)

    os.makedirs(qrels_dir, exist_ok=True)

    # Write corpus
    with open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for _, row in df_corpus.iterrows():
            # MTEB corpus format: _id, title, text
            doc_id = str(row["_id"])
            title = str(row.get("title", ""))
            text = str(row.get("text", ""))
            f.write(json.dumps({"_id": doc_id, "title": title, "text": text}, ensure_ascii=False) + "\n")

    # Write queries
    with open(os.path.join(out_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
        for _, row in df_queries.iterrows():
            # MTEB queries format: _id, text
            q_id = str(row["_id"])
            text = str(row.get("text", ""))
            f.write(json.dumps({"_id": q_id, "text": text}, ensure_ascii=False) + "\n")

    # Write qrels
    with open(os.path.join(qrels_dir, "test.tsv"), "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for _, row in df_qrels.iterrows():
            # MTEB qrels format: query-id, corpus-id, score
            qid = str(row["query-id"])
            cid = str(row["corpus-id"])
            score = int(row["score"])
            f.write(f"{qid}\t{cid}\t{score}\n")

    print(f"  [OK] GermanQuAD: {len(df_corpus)} docs, {len(df_queries)} queries → {out_dir}")
    return out_dir


def download_miracl(cache_dir, languages=("fr", "es", "en")):
    """Download MIRACL dev splits and convert to BEIR format directly from files."""
    import gzip
    from huggingface_hub import list_repo_files, hf_hub_download

    # Get lists of files once
    print("  [INFO] Fetching file list from miracl/miracl and miracl/miracl-corpus...")
    corpus_repo_files = list_repo_files(repo_id="miracl/miracl-corpus", repo_type="dataset")
    topics_repo_files = list_repo_files(repo_id="miracl/miracl", repo_type="dataset")

    for lang in languages:
        out_dir = os.path.join(cache_dir, "datasets", f"miracl_{lang}")
        qrels_dir = os.path.join(out_dir, "qrels")

        if os.path.exists(os.path.join(out_dir, "corpus.jsonl")):
            print(f"  [SKIP] MIRACL-{lang} already converted")
            continue

        print(f"  [DOWNLOAD] miracl/miracl ({lang}) ...")

        # 1. Find and download corpus files (.jsonl.gz)
        prefix_corpus = f"miracl-corpus-v1.0-{lang}/"
        corpus_files = [f for f in corpus_repo_files if f.startswith(prefix_corpus) and f.endswith(".jsonl.gz")]

        if not corpus_files:
            print(f"  [WARN] No corpus files found for language: {lang}. Skipping.")
            continue

        os.makedirs(qrels_dir, exist_ok=True)

        print(f"    Downloading {len(corpus_files)} corpus files...")
        corpus = {}
        with open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8") as out_f:
            for c_file in corpus_files:
                local_path = hf_hub_download(
                    repo_id="miracl/miracl-corpus",
                    filename=c_file,
                    repo_type="dataset"
                )
                # Read gzip and write to corpus.jsonl
                with gzip.open(local_path, "rt", encoding="utf-8") as gz_f:
                    for line in gz_f:
                        row = json.loads(line)
                        doc_id = str(row["docid"])
                        title = row.get("title", "")
                        text = row.get("text", "")
                        corpus[doc_id] = True
                        out_f.write(json.dumps({"_id": doc_id, "title": title, "text": text}, ensure_ascii=False) + "\n")

        # 2. Find and download topics and qrels files (.tsv)
        prefix_topics = f"miracl-v1.0-{lang}/"
        try:
            topics_file_name = [f for f in topics_repo_files if f.startswith(prefix_topics) and "topics" in f and "dev.tsv" in f][0]
            qrels_file_name = [f for f in topics_repo_files if f.startswith(prefix_topics) and "qrels" in f and "dev.tsv" in f][0]
        except IndexError:
            print(f"  [WARN] Topics/qrels not found for language: {lang} in dev split. Skipping.")
            continue

        local_topics_path = hf_hub_download(
            repo_id="miracl/miracl",
            filename=topics_file_name,
            repo_type="dataset"
        )
        local_qrels_path = hf_hub_download(
            repo_id="miracl/miracl",
            filename=qrels_file_name,
            repo_type="dataset"
        )

        # 3. Read and write queries
        queries = {}
        with open(local_topics_path, "r", encoding="utf-8") as f:
            # Format: query_id\tquery
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    queries[parts[0]] = parts[1]

        with open(os.path.join(out_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
            for q_id, text in queries.items():
                f.write(json.dumps({"_id": q_id, "text": text}, ensure_ascii=False) + "\n")

        # 4. Read and write qrels
        qrels_count = 0
        with open(os.path.join(qrels_dir, "dev.tsv"), "w", encoding="utf-8") as out_qrels_f:
            out_qrels_f.write("query-id\tcorpus-id\tscore\n")
            with open(local_qrels_path, "r", encoding="utf-8") as f:
                # Format: query_id\t0\tdoc_id\trelevance
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        qid, _, did, rel = parts
                        if int(rel) > 0:
                            out_qrels_f.write(f"{qid}\t{did}\t{rel}\n")
                            qrels_count += 1

        print(f"  [OK] MIRACL-{lang}: {len(corpus)} docs, {len(queries)} queries, {qrels_count} qrels → {out_dir}")


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
