#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N dream_harmbench_vanilla_gen
#PBS -o eval_logs/dream_vanilla_gen.log
#PBS -j oe

# Generates Dream responses for vanilla HarmBench prompts (no attack, no defense).
# Mirrors harmbench_vanilla_gen.sh (LLaDA). Provides baseline ASR for Dream.
# Run dream_harmbench_vanilla_eval.sh after this.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/Saftey_Critical_CW_code/venv/bin/activate

cd ~/Saftey_Critical_CW_code

PYTHON=$(which python)
MODEL_PATH="$HOME/Saftey_Critical_CW_code/hf_models/Dream-v0-Instruct-7B"
RAW_CSV="$HOME/Saftey_Critical_CW_code/data/pre_refined_prompt/harmbench.csv"
HARMBENCH_JSON="$HOME/Saftey_Critical_CW_code/data/pre_refined_prompt/harmbench.json"

OUTDIR="dream_results/vanilla/baseline"
mkdir -p "$OUTDIR"

echo "=============================================="
echo "HarmBench Vanilla Baseline (Dream) - Generation"
echo "=============================================="

# Convert harmbench.csv to JSON if not already done
if [ ! -f "$HARMBENCH_JSON" ]; then
    echo "Converting harmbench.csv to JSON..."
    $PYTHON - <<PYEOF
import pandas as pd, json
df = pd.read_csv("${RAW_CSV}")
records = [{"Behavior": row["Behavior"]} for _, row in df.iterrows()]
with open("${HARMBENCH_JSON}", "w") as f:
    json.dump(records, f, indent=2)
print(f"Saved {len(records)} behaviors to ${HARMBENCH_JSON}")
PYEOF
fi

echo ""
echo "--- Running: vanilla baseline ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${HARMBENCH_JSON}" \
    --output_json "${OUTDIR}/generation.json" \
    --remasking off \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --debug_print

echo ""
echo "=============================================="
echo "Generation done. Run dream_harmbench_vanilla_eval.sh next."
echo "=============================================="
