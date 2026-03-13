#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N final_harmbench_context_nesting_gen
#PBS -o eval_logs/context_nesting_gen.log
#PBS -j oe

# Runs Context Nesting attack on HarmBench prompts across strategies.
# Context Nesting wraps each behavior in a structural template (Python, LaTeX,
# JSON, Markdown, YAML, or Story) — no inline masks, no --fill_all_masks.
# Requires: data/refined_prompt_data/harmbench_context_nesting.json
#   (generate with run_refinement_all.sh if not already done)
# Evaluation is in harmbench_context_nesting_eval.sh.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/DiffuGuard/venv/bin/activate

cd ~/DiffuGuard

PYTHON=$(which python)
MODEL_PATH="$HOME/DiffuGuard/hf_models/LLaDA-8B-Instruct"
HARMBENCH_JSON="$HOME/DiffuGuard/data/pre_refined_prompt/harmbench.json"
CONTEXT_NESTING_PROMPTS="$HOME/DiffuGuard/data/refined_prompt_data/harmbench_context_nesting_refined.json"

BASEDIR="results/context_nesting"

if [ ! -f "$CONTEXT_NESTING_PROMPTS" ]; then
    echo "ERROR: $CONTEXT_NESTING_PROMPTS not found. Run run_refinement_all.sh first."
    exit 1
fi

echo "=============================================="
echo "HarmBench Context Nesting Attack - Generation"
echo "=============================================="

# ==============================================================
# no_defense
# ==============================================================
EXPDIR="${BASEDIR}/no_defense"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: no_defense ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# fully_random
# ==============================================================
EXPDIR="${BASEDIR}/fully_random"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: fully_random ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --remasking random \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# stochastic_annealing
# ==============================================================
EXPDIR="${BASEDIR}/stochastic_annealing"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: stochastic_annealing ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --remasking adaptive \
    --alpha0 0.6 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# stochastic_annealing_exp
# ==============================================================
EXPDIR="${BASEDIR}/stochastic_annealing_exp"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: stochastic_annealing_exp ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --remasking adaptive_step_exp \
    --alpha0 0.9 \
    --c 0.12 \
    --m 3 \
    --ratio 3.0 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# fully_random (block level audit active)
# ==============================================================
EXPDIR="${BASEDIR}/fully_random_block_audit"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: fully_random (block level audit) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --remasking random \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# stochastic_annealing (block level audit active)
# ==============================================================
EXPDIR="${BASEDIR}/stochastic_annealing_block_audit"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: stochastic_annealing (block level audit) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --remasking adaptive \
    --alpha0 0.6 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# greedy (block level audit active)
# ==============================================================
EXPDIR="${BASEDIR}/greedy_block_audit"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: greedy (block level audit) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# stochastic_annealing_exp (block level audit active)
# ==============================================================
EXPDIR="${BASEDIR}/stochastic_annealing_exp_block_audit"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: stochastic_annealing_exp (block level audit) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --remasking adaptive_step_exp \
    --alpha0 0.9 \
    --c 0.12 \
    --m 3 \
    --ratio 3.0 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# SPD (no block audit)
# ==============================================================
EXPDIR="${BASEDIR}/spd"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: spd ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 32 \
    --cfg_scale 0.0 \
    --spd_k 5 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# SPD (with block audit)
# ==============================================================
EXPDIR="${BASEDIR}/spd_block_audit"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: spd (block level audit) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 32 \
    --cfg_scale 0.0 \
    --spd_k 5 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# SPD + self-reminder (no block audit)
# ==============================================================
EXPDIR="${BASEDIR}/spd_reminder"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: spd + self-reminder ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 32 \
    --cfg_scale 0.0 \
    --spd_k 5 \
    --defense_method self-reminder \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

# ==============================================================
# SPD + self-reminder (with block audit)
# ==============================================================
EXPDIR="${BASEDIR}/spd_reminder_block_audit"
mkdir -p "$EXPDIR"
echo ""
echo "--- Strategy: spd + self-reminder (block level audit) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 32 \
    --cfg_scale 0.0 \
    --spd_k 5 \
    --defense_method self-reminder \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

echo ""
echo "=============================================="
echo "Generation done. Run harmbench_context_nesting_eval.sh next."
echo "=============================================="
