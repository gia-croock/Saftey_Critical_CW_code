#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -N download_raw_prompts

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/DiffuGuard/venv/bin/activate

cd ~/DiffuGuard
mkdir -p data/pre_refined_prompt

echo "=============================================="
echo "Downloading raw prompt datasets"
echo "=============================================="

python3 - <<'PYEOF'
import os
import urllib.request
import pandas as pd
from datasets import load_dataset

out_dir = "data/pre_refined_prompt"

#  HarmBench
# HuggingFace mirror: huihui-ai/harmbench_behaviors
# Already has a Behavior column (harmbench_behaviors_text_test.csv, 320 rows)
# -------------------------------------------------------
print("\n--- HarmBench ---")
hb_url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
urllib.request.urlretrieve(hb_url, f"{out_dir}/harmbench_raw.csv")
df = pd.read_csv(f"{out_dir}/harmbench_raw.csv")
excluded = {"copyright", "misinformation_disinformation"}
if "SemanticCategory" in df.columns:
    before = len(df)
    df = df[~df["SemanticCategory"].isin(excluded)].reset_index(drop=True)
    print(f"Filtered {before - len(df)} rows (copyright/misinformation). {len(df)} remaining.")
df[["Behavior"]].to_csv(f"{out_dir}/harmbench.csv", index=False)
print(f"Saved {len(df)} HarmBench behaviors to {out_dir}/harmbench.csv")


print("\nAll datasets downloaded to data/pre_refined_prompt/")
PYEOF

echo ""
echo "=============================================="
echo "Done! Files in data/pre_refined_prompt/:"
ls data/pre_refined_prompt/
echo "=============================================="
