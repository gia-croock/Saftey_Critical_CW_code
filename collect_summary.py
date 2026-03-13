"""
Collects key metrics from all experiment results into a single summary.json.

Usage:
    python collect_summary.py
    python collect_summary.py --results_dir results --output results/summary.json
"""

import os
import json
import argparse


def extract_metrics(results_path):
    """Extract summary metrics from a results.json file."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        return {}

    # ASR_e: fraction of items where harmbench classifier said "Yes"
    harmbench_verdicts = [
        item.get("harmbench_evaluation", {}).get("is_jailbreak")
        for item in data
        if "harmbench_evaluation" in item
    ]
    asr_e = None
    if harmbench_verdicts:
        asr_e = round(sum(1 for v in harmbench_verdicts if v) / len(harmbench_verdicts) * 100, 2)

    # ASR_k: look for asr_k_evaluation field
    asr_k_verdicts = [
        item.get("asr_k_evaluation", {}).get("is_jailbreak")
        for item in data
        if "asr_k_evaluation" in item
    ]
    asr_k = None
    if asr_k_verdicts:
        asr_k = round(sum(1 for v in asr_k_verdicts if v) / len(asr_k_verdicts) * 100, 2)

    # PPL: average perplexity across items
    ppls = [
        item.get("perplexity")
        for item in data
        if item.get("perplexity") is not None
    ]
    avg_ppl = round(sum(ppls) / len(ppls), 2) if ppls else None

    # SAR: look for sar_evaluation field
    sar_verdicts = [
        item.get("sar_evaluation", {}).get("adherent")
        for item in data
        if "sar_evaluation" in item
    ]
    sar = None
    if sar_verdicts:
        sar = round(sum(1 for v in sar_verdicts if v) / len(sar_verdicts) * 100, 2)

    return {
        "ASR_e": asr_e,
        "ASR_k": asr_k,
        "PPL": avg_ppl,
        "SAR": sar,
        "num_items": len(data),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect experiment results into summary.json")
    parser.add_argument("--results_dir", default="results", help="Root results directory")
    parser.add_argument("--output", default="results/summary.json", help="Output summary path")
    args = parser.parse_args()

    summary = []

    for root, dirs, files in os.walk(args.results_dir):
        if "results.json" not in files:
            continue

        # Extract attack type and experiment name from path
        # e.g. results/context_nesting/fully_random_block_audit/results.json
        rel_path = os.path.relpath(root, args.results_dir)
        parts = rel_path.split(os.sep)

        if len(parts) == 2:
            attack, experiment = parts
        elif len(parts) == 3:
            # e.g. harmless_context_nesting has no extra nesting but just in case
            attack = os.path.join(parts[0], parts[1])
            experiment = parts[2]
        else:
            continue

        results_path = os.path.join(root, "results.json")
        try:
            metrics = extract_metrics(results_path)
        except Exception as e:
            print(f"WARNING: Failed to read {results_path}: {e}")
            continue

        entry = {
            "attack": attack,
            "experiment": experiment,
            **metrics,
        }
        summary.append(entry)
        print(f"  {attack}/{experiment}: ASR_e={metrics.get('ASR_e')} ASR_k={metrics.get('ASR_k')} PPL={metrics.get('PPL')} SAR={metrics.get('SAR')}")

    # Sort for consistent output
    summary.sort(key=lambda x: (x["attack"], x["experiment"]))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved summary ({len(summary)} experiments) to {args.output}")


if __name__ == "__main__":
    main()
