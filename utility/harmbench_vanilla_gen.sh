#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N final_harmbench_vanilla_gen
#PBS -o eval_logs/final_harmbench_vanilla_gen.log
#PBS -j oe

# Generates LLaDA responses for vanilla HarmBench prompts (no defense baseline).

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/DiffuGuard/venv/bin/activate

cd ~/DiffuGuard

PYTHON=$(which python)
MODEL_PATH="$HOME/DiffuGuard/hf_models/LLaDA-8B-Instruct"
RAW_CSV="$HOME/DiffuGuard/data/pre_refined_prompt/harmbench.csv"
HARMBENCH_JSON="$HOME/DiffuGuard/data/pre_refined_prompt/harmbench.json"

mkdir -p results

echo "=============================================="
echo "HarmBench - Generation"
echo "=============================================="

# Convert harmbench.csv to JSON expected by jailbreakbench_llada.py
echo "Converting harmbench.csv to JSON..."
$PYTHON - <<PYEOF
import pandas as pd
import json

df = pd.read_csv("${RAW_CSV}")
records = [{"Behavior": row["Behavior"]} for _, row in df.iterrows()]
with open("${HARMBENCH_JSON}", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
print(f"Saved {len(records)} behaviors to ${HARMBENCH_JSON}")
PYEOF

echo ""
echo "--- Running: vanilla ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${HARMBENCH_JSON}" \
    --output_json "results/out_harmbench_vanilla.json" \
    --sp_mode off \
    --mask_counts 0 \
    --debug_print

echo ""
echo "=============================================="
echo "Generation done. Run final_harmbench_eval.sh next."
echo "=============================================="
