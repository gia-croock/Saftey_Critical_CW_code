#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N dream_harmless_plain_gen
#PBS -o eval_logs/dream_harmless_plain_gen.log
#PBS -j oe

# Runs zeroshot (no attack) on BENIGN prompts across all defenses.
# Purpose: measure PPL and SAR on harmless inputs as a utility baseline
# with no attack wrapper (cf. dream_harmless_context_nesting_gen.sh which uses CN).
# No ASR evaluation (these are not harmful prompts).
# Mirrors harmless_plain_gen.sh (LLaDA).
# Requires: data/pre_refined_prompt/benign_prompts.csv
# Run dream_harmless_plain_eval.sh after this.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/Saftey_Critical_CW_code/venv/bin/activate

cd ~/Saftey_Critical_CW_code

PYTHON=$(which python)
MODEL_PATH="$HOME/Saftey_Critical_CW_code/hf_models/Dream-v0-Instruct-7B"
BENIGN_CSV="$HOME/Saftey_Critical_CW_code/data/pre_refined_prompt/benign_prompts.csv"
BENIGN_JSON="$HOME/Saftey_Critical_CW_code/data/pre_refined_prompt/benign_prompts.json"

BASEDIR="dream_results/harmless_plain"

mkdir -p eval_logs

# ==============================================================
# Step 0: Convert benign CSV to JSON
# ==============================================================
if [ ! -f "$BENIGN_JSON" ]; then
    echo "=============================================="
    echo "Step 0: Converting benign_prompts.csv to JSON"
    echo "=============================================="
    $PYTHON - <<PYEOF
import pandas as pd
import json

df = pd.read_csv("${BENIGN_CSV}")
records = [{"Behavior": row["Behavior"]} for _, row in df.iterrows()]
with open("${BENIGN_JSON}", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
print(f"Saved {len(records)} behaviors to ${BENIGN_JSON}")
PYEOF
else
    echo "Reusing existing: ${BENIGN_JSON}"
fi

if [ ! -f "$BENIGN_JSON" ]; then
    echo "ERROR: JSON conversion failed. Exiting."
    exit 1
fi

echo ""
echo "=============================================="
echo "HarmBench Harmless Plain (Dream) - Generation"
echo "=============================================="

# ==============================================================
# no_defense
# ==============================================================
EXPDIR="${BASEDIR}/no_defense"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: no_defense ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_JSON}" \
    --output_json "${EXPDIR}/generation.json" \
    --remasking off \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# fully_random
# ==============================================================
EXPDIR="${BASEDIR}/fully_random"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: fully_random ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_JSON}" \
    --output_json "${EXPDIR}/generation.json" \
    --remasking random \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# stochastic_annealing
# ==============================================================
EXPDIR="${BASEDIR}/stochastic_annealing"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: stochastic_annealing ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_JSON}" \
    --output_json "${EXPDIR}/generation.json" \
    --remasking adaptive \
    --alpha0 0.3 \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# stochastic_annealing_exp
# ==============================================================
EXPDIR="${BASEDIR}/stochastic_annealing_exp"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: stochastic_annealing_exp ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_JSON}" \
    --output_json "${EXPDIR}/generation.json" \
    --remasking adaptive_step_exp \
    --alpha0 0.9 \
    --c 0.12 \
    --m 3 \
    --ratio 3.0 \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# fully_random (block level audit)
# ==============================================================
EXPDIR="${BASEDIR}/fully_random_block_audit"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: fully_random (block level audit) ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_JSON}" \
    --output_json "${EXPDIR}/generation.json" \
    --remasking random \
    --sp_mode hidden \
    --sp_threshold 0.1 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# stochastic_annealing (block level audit)
# ==============================================================
EXPDIR="${BASEDIR}/stochastic_annealing_block_audit"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: stochastic_annealing (block level audit) ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_JSON}" \
    --output_json "${EXPDIR}/generation.json" \
    --remasking adaptive \
    --alpha0 0.3 \
    --sp_mode hidden \
    --sp_threshold 0.1 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# greedy (block level audit)
# ==============================================================
EXPDIR="${BASEDIR}/greedy_block_audit"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: greedy (block level audit) ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_JSON}" \
    --output_json "${EXPDIR}/generation.json" \
    --remasking low_confidence \
    --sp_mode hidden \
    --sp_threshold 0.1 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# stochastic_annealing_exp (block level audit)
# ==============================================================
EXPDIR="${BASEDIR}/stochastic_annealing_exp_block_audit"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: stochastic_annealing_exp (block level audit) ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_JSON}" \
    --output_json "${EXPDIR}/generation.json" \
    --remasking adaptive_step_exp \
    --alpha0 0.9 \
    --c 0.12 \
    --m 3 \
    --ratio 3.0 \
    --sp_mode hidden \
    --sp_threshold 0.1 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# spd (no block audit)
# ==============================================================
EXPDIR="${BASEDIR}/spd"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: spd ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_JSON}" \
    --output_json "${EXPDIR}/generation.json" \
    --remasking off \
    --spd_k 5 \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# spd (with block audit)
# ==============================================================
EXPDIR="${BASEDIR}/spd_block_audit"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: spd (block level audit) ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_JSON}" \
    --output_json "${EXPDIR}/generation.json" \
    --remasking off \
    --spd_k 5 \
    --sp_mode hidden \
    --sp_threshold 0.1 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# spd + self-reminder (no block audit)
# ==============================================================
EXPDIR="${BASEDIR}/spd_reminder"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: spd + self-reminder ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_JSON}" \
    --output_json "${EXPDIR}/generation.json" \
    --remasking off \
    --spd_k 5 \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --defense_method self-reminder \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# spd + self-reminder (with block audit)
# ==============================================================
EXPDIR="${BASEDIR}/spd_reminder_block_audit"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: spd + self-reminder (block level audit) ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_JSON}" \
    --output_json "${EXPDIR}/generation.json" \
    --remasking off \
    --spd_k 5 \
    --sp_mode hidden \
    --sp_threshold 0.1 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 64 \
    --gen_length 128 \
    --defense_method self-reminder \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

echo ""
echo "=============================================="
echo "Generation done. Run dream_harmless_plain_eval.sh next."
echo "=============================================="
