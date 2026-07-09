"""
Print a neat side-by-side comparison table of our reproduction results vs. the paper's Table 2.
"""

import os
import csv
import json

# Exact MT5-base Table 2 scores extracted directly from Braga et al. SIGIR 2025
PAPER_RESULTS = {
    "germanquad": {
        "BM25": {"P@10": 0.0590, "NDCG@3": 0.3810, "NDCG@10": 0.4370, "MAP@100": 0.3970},
        "BM25+Theta_T": {"P@10": 0.0690, "NDCG@3": 0.4510, "NDCG@10": 0.5130, "MAP@100": 0.4630},
        "BM25+TA_a1.0": {"P@10": 0.0710, "NDCG@3": 0.4770, "NDCG@10": 0.5370, "MAP@100": 0.4870},
    },
    "miracl_fr": {
        "BM25": {"P@10": 0.0520, "NDCG@3": 0.1250, "NDCG@10": 0.1740, "MAP@100": 0.1390},
        "BM25+Theta_T": {"P@10": 0.0710, "NDCG@3": 0.1750, "NDCG@10": 0.2340, "MAP@100": 0.1860},
        "BM25+TA_a1.0": {"P@10": 0.0810, "NDCG@3": 0.2150, "NDCG@10": 0.2780, "MAP@100": 0.2200},
    },
    "miracl_es": {
        "BM25": {"P@10": 0.1350, "NDCG@3": 0.2480, "NDCG@10": 0.2700, "MAP@100": 0.2150},
        "BM25+Theta_T": {"P@10": 0.1900, "NDCG@3": 0.3420, "NDCG@10": 0.3790, "MAP@100": 0.3010},
        "BM25+TA_a1.0": {"P@10": 0.2000, "NDCG@3": 0.3730, "NDCG@10": 0.4050, "MAP@100": 0.3250},
    },
    "miracl_en": {
        "BM25": {"P@10": 0.1070, "NDCG@3": 0.2510, "NDCG@10": 0.3020, "MAP@100": 0.2470},
        "BM25+Theta_T": {"P@10": 0.1400, "NDCG@3": 0.3300, "NDCG@10": 0.3980, "MAP@100": 0.3250},
        "BM25+TA_a1.0": {"P@10": 0.1510, "NDCG@3": 0.3660, "NDCG@10": 0.4350, "MAP@100": 0.3580},
    }
}

DATASET_DISPLAY_NAMES = {
    "germanquad": "GermanQuAD (German)",
    "miracl_fr": "MIRACL French (fr)",
    "miracl_es": "MIRACL Spanish (es)",
    "miracl_en": "MIRACL English (en)"
}


def load_baselines_from_log(log_path):
    """Parse baselines output from log file, tracking dataset context."""
    results = {}
    if not os.path.exists(log_path):
        return results

    current_dataset = None
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        # Detect dataset context
        line_upper = line.upper()
        if "LOADING DATASET" in line_upper or "RESULTS —" in line_upper or "BASELINES —" in line_upper:
            for ds in PAPER_RESULTS.keys():
                if ds.upper() in line_upper:
                    current_dataset = ds
                    break

        # Discard log prefix (e.g., date, time, logger name, log level)
        if "INFO" in line:
            line = line.split("INFO")[-1]
            
        parts = line.strip().split()
        if len(parts) >= 5:
            variant = parts[0]
            # Normalize names
            if variant == "BM25+Theta_T" or "Theta_T" in variant or "Θ_T" in variant:
                variant = "BM25+Theta_T"
            elif variant == "BM25":
                variant = "BM25"
            else:
                continue

            try:
                scores = [float(p) for p in parts[1:5]]
                ds_key = current_dataset if current_dataset else "germanquad"
                if ds_key not in results:
                    results[ds_key] = {}
                results[ds_key][variant] = {
                    "P@10": scores[0],
                    "NDCG@3": scores[1],
                    "NDCG@10": scores[2],
                    "MAP@100": scores[3],
                }
            except ValueError:
                continue
    return results


def main():
    results_dir = "./Scripts/results"
    print("=" * 90)
    print("MT5 REPRODUCTION RESULT COMPARISON — TABLE 2 (MULTILINGUAL)")
    print("=" * 90)

    datasets = ["germanquad", "miracl_fr", "miracl_es", "miracl_en"]
    our_scores = {ds: {} for ds in datasets}

    # Find and parse log files from newest to oldest to aggregate baselines
    log_files = [f for f in os.listdir(results_dir) if f.startswith("mt5_gpunode_") and f.endswith(".log")]
    log_files = sorted(log_files, reverse=True)
    
    for lf in log_files:
        log_path = os.path.join(results_dir, lf)
        log_data = load_baselines_from_log(log_path)
        for ds, ds_scores in log_data.items():
            if ds in our_scores:
                for k, v in ds_scores.items():
                    if k not in our_scores[ds]:
                        our_scores[ds][k] = v

    # Load test/TA scores from CSV files
    for ds in datasets:
        test_csv = os.path.join(results_dir, f"mt5_{ds}_test.csv")
        if os.path.exists(test_csv):
            with open(test_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    alpha = row["alpha"]
                    our_scores[ds][f"BM25+TA_a{alpha}"] = {
                        "P@10": float(row["P@10"]),
                        "NDCG@3": float(row["NDCG@3"]),
                        "NDCG@10": float(row["NDCG@10"]),
                        "MAP@100": float(row["MAP@100"]),
                    }

    # Print tables side-by-side with paper values
    for ds in datasets:
        print(f"\n==================== {DATASET_DISPLAY_NAMES[ds].upper()} ====================")
        print("-" * 90)
        print(f"{'Metric & Variant':<22} | {'Paper Score':<12} | {'Our Score':<12} | {'Gap (Our - Paper)':<18}")
        print("-" * 90)

        paper = PAPER_RESULTS[ds]
        ds_our = our_scores[ds]

        variants = [
            ("BM25", "BM25"),
            ("BM25+Theta_T", "BM25+Θ_T (mMARCO)"),
            ("BM25+TA_a1.0", "BM25+Θ′ (α=1.0)"),
        ]

        for paper_var, display_name in variants:
            print(f"\n--- {display_name} ---")
            p_scores = paper.get(paper_var, {})
            o_scores = ds_our.get(paper_var, {})

            for metric in ["NDCG@10", "NDCG@3", "P@10", "MAP@100"]:
                p_val = p_scores.get(metric, 0.0)
                o_val = o_scores.get(metric, None)

                if o_val is not None:
                    gap = o_val - p_val
                    print(f"  {metric:<18} | {p_val:<12.4f} | {o_val:<12.4f} | {gap:<+18.4f}")
                else:
                    print(f"  {metric:<18} | {p_val:<12.4f} | {'N/A':<12} | {'N/A':<18}")

        print("-" * 90)

    print("\nDone. Copy this output and paste it to the chat!")


if __name__ == "__main__":
    main()
