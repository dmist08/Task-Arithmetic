"""
Print a neat side-by-side comparison table of our reproduction results vs. the paper's Table 2.
"""

import os
import csv
import json

PAPER_RESULTS = {
    "germanquad": {
        "BM25": {"P@10": 0.0463, "NDCG@3": 0.2613, "NDCG@10": 0.2760, "MAP@100": 0.2050},
        "BM25+Theta_T": {"P@10": 0.0496, "NDCG@3": 0.2999, "NDCG@10": 0.3040, "MAP@100": 0.2410},
        "BM25+TA_a1.0": {"P@10": 0.0570, "NDCG@3": 0.3561, "NDCG@10": 0.3590, "MAP@100": 0.2960},
    }
}


def load_baselines_from_log(log_path):
    """Parse baselines output from log file if exists."""
    results = {}
    if not os.path.exists(log_path):
        return results

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if "BASELINES" in line or "RESULTS" in line:
            continue
        # Format: BM25 / BM25+Theta_T followed by floats
        parts = line.strip().split()
        if len(parts) >= 5:
            variant = parts[0]
            try:
                scores = [float(p) for p in parts[1:5]]
                results[variant] = {
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
    print("=" * 70)
    print("MT5 REPRODUCTION RESULT COMPARISON")
    print("=" * 70)

    # Let's find files
    test_csv = os.path.join(results_dir, "mt5_germanquad_test.csv")
    log_file = os.path.join(results_dir, "mt5_germanquad_baselines.log")

    if not os.path.exists(test_csv):
        # Check for any log file to parse as fallback
        log_files = [f for f in os.listdir(results_dir) if f.startswith("mt5_gpunode_") and f.endswith(".log")]
        if log_files:
            log_file = os.path.join(results_dir, sorted(log_files)[-1])
        else:
            print(f"No results files found in {results_dir} yet.")
            print("Please run `bash Scripts/run_mt5_gpunode.sh` first.")
            return

    # Load baseline scores
    our_scores = load_baselines_from_log(log_file)

    # Load test/TA scores
    if os.path.exists(test_csv):
        with open(test_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                alpha = row["alpha"]
                our_scores[f"BM25+TA_a{alpha}"] = {
                    "P@10": float(row["P@10"]),
                    "NDCG@3": float(row["NDCG@3"]),
                    "NDCG@10": float(row["NDCG@10"]),
                    "MAP@100": float(row["MAP@100"]),
                }

    paper = PAPER_RESULTS["germanquad"]

    print("\nGermanQuAD Table 2 Comparison:")
    print("-" * 90)
    print(f"{'Metric & Variant':<22} | {'Paper Score':<12} | {'Our Score':<12} | {'Gap (Our - Paper)':<18}")
    print("-" * 90)

    variants = [
        ("BM25", "BM25"),
        ("BM25+Theta_T", "BM25+Θ_T (mMARCO)"),
        ("BM25+TA_a1.0", "BM25+Θ′ (α=1.0)"),
    ]

    for paper_var, display_name in variants:
        print(f"\n--- {display_name} ---")
        p_scores = paper.get(paper_var, {})
        o_scores = our_scores.get(paper_var, {})

        for metric in ["NDCG@10", "NDCG@3", "P@10", "MAP@100"]:
            p_val = p_scores.get(metric, 0.0)
            o_val = o_scores.get(metric, None)

            if o_val is not None:
                gap = o_val - p_val
                print(f"  {metric:<18} | {p_val:<12.4f} | {o_val:<12.4f} | {gap:<+18.4f}")
            else:
                print(f"  {metric:<18} | {p_val:<12.4f} | {'N/A':<12} | {'N/A':<18}")

    print("-" * 90)
    print("Done. Copy this output and paste it to the chat!")


if __name__ == "__main__":
    main()
