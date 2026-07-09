"""
Download all models and datasets needed for reproduction.
Run this on the LOGIN NODE (with internet access).

Everything gets saved under a single directory (default: ../offline_cache/)
that gpunode1 can access via the shared filesystem.

Usage:
    python Scripts/download_all.py

    # Or specify a custom cache directory:
    python Scripts/download_all.py --cache_dir /path/to/shared/storage/ta_cache
"""

import os
import sys
import argparse
import zipfile
import urllib.request
import shutil


def download_file(url, dest_path):
    """Download a file with progress."""
    if os.path.exists(dest_path):
        print(f"  [SKIP] Already exists: {dest_path}")
        return
    print(f"  Downloading: {url}")
    print(f"  To: {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"  Done: {os.path.getsize(dest_path) / (1024*1024):.1f} MB")


def download_and_extract_beir(dataset_name, datasets_dir):
    """Download a BEIR dataset zip and extract it."""
    dest_dir = os.path.join(datasets_dir, dataset_name)
    if os.path.exists(dest_dir) and os.listdir(dest_dir):
        print(f"  [SKIP] Already extracted: {dest_dir}")
        return dest_dir

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    zip_path = os.path.join(datasets_dir, f"{dataset_name}.zip")
    download_file(url, zip_path)

    print(f"  Extracting to {datasets_dir} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(datasets_dir)
    os.remove(zip_path)
    print(f"  Extracted: {dest_dir}")
    return dest_dir


def download_hf_model(model_id, models_dir):
    """Download a HuggingFace model to a local directory."""
    safe_name = model_id.replace("/", "__")
    dest_dir = os.path.join(models_dir, safe_name)

    if os.path.exists(dest_dir) and any(f.endswith(('.bin', '.safetensors')) for f in os.listdir(dest_dir)):
        print(f"  [SKIP] Already downloaded: {dest_dir}")
        return dest_dir

    print(f"  Downloading HF model: {model_id}")
    print(f"  To: {dest_dir}")

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=model_id,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
    )
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(dest_dir)
        for f in filenames
    ) / (1024 * 1024)
    print(f"  Done: {size_mb:.1f} MB total")
    return dest_dir


def main():
    parser = argparse.ArgumentParser(description="Download all models and datasets for offline reproduction")
    parser.add_argument("--cache_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "offline_cache"),
                        help="Root directory for all downloads")
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    models_dir = os.path.join(cache_dir, "models")
    datasets_dir = os.path.join(cache_dir, "datasets")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    print("=" * 60)
    print("Task Arithmetic Reproduction — Offline Download Script")
    print(f"Cache directory: {cache_dir}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. BEIR Datasets
    # ------------------------------------------------------------------
    print("\n[1/2] Downloading BEIR datasets...")
    for ds in ["scifact", "nfcorpus"]:
        print(f"\n  --- {ds} ---")
        download_and_extract_beir(ds, datasets_dir)

    # ------------------------------------------------------------------
    # 2. HuggingFace Models
    # ------------------------------------------------------------------
    print("\n[2/2] Downloading HuggingFace models...")

    hf_models = [
        ("roberta-base", "Θ₀ — base pretrained"),
        ("allenai/biomed_roberta_base", "Θ_D — BioMed-RoBERTa (domain)"),
        ("cross-encoder/stsb-roberta-base", "Θ_T — cross-encoder (IR)"),
    ]

    for model_id, description in hf_models:
        print(f"\n  --- {description}: {model_id} ---")
        download_hf_model(model_id, models_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL DOWNLOADS COMPLETE")
    print("=" * 60)
    print(f"\nDirectory structure:")
    print(f"  {cache_dir}/")
    print(f"  ├── datasets/")
    print(f"  │   ├── scifact/")
    print(f"  │   └── nfcorpus/")
    print(f"  └── models/")
    for model_id, desc in hf_models:
        safe = model_id.replace("/", "__")
        print(f"      ├── {safe}/  ({desc})")

    print(f"\nNext step: on gpunode1, run:")
    print(f"  cd Scripts")
    print(f"  python reproduce_roberta_offline.py \\")
    print(f"    --cache_dir {cache_dir} \\")
    print(f"    --mode dev_sweep \\")
    print(f"    --device cuda:0")
    print()


if __name__ == "__main__":
    main()
