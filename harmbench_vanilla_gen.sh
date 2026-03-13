#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N final_harmbench_vanilla_gen
#PBS -o eval_logs/vanilla_baseline_gen.log
#PBS -j oe

# Generates LLaDA responses for vanilla HarmBench prompts (no defense baseline).

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/Saftey_Critical_CW_code/venv/bin/activate

cd ~/Saftey_Critical_CW_code

PYTHON=$(which python)
MODEL_PATH="$HOME/Saftey_Critical_CW_code/hf_models/LLaDA-8B-Instruct"
RAW_CSV="$HOME/Saftey_Critical_CW_code/data/pre_refined_prompt/harmbench.csv"
HARMBENCH_JSON="$HOME/Saftey_Critical_CW_code/data/pre_refined_prompt/harmbench.json"

OUTDIR="results/vanilla/baseline"
mkdir -p "$OUTDIR"

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
# baseline: direct prompt, no attack, no defense; code defaults for temp/steps
# DiffuGuard Table 5 uses temp=0.5, steps=64; CN paper uses steps=32.
# Vanilla baseline uses code defaults to represent unmodified model behaviour.
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${HARMBENCH_JSON}" \
    --output_json "${OUTDIR}/generation.json" \
    --sp_mode off \
    --mask_counts 0 \
    --debug_print

echo ""
echo "=============================================="
echo "Generation done. Run harmbench_vanilla_eval.sh next."
echo "=============================================="
