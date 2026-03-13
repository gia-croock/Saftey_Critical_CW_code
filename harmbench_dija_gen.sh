#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N final_harmbench_dija_gen
#PBS -o eval_logs/dija_gen.log
#PBS -j oe

# Runs DIJA attack on HarmBench refined prompts across remasking strategies.
# Requires: data/refined_prompt_data/harmbench_refined.json
#   (generate with run_refinement_all.sh if not already done)
# Evaluation is in harmbench_dija_eval.sh.
#
# ── Parameter justification key ──────────────────────────────────────────
# Sources: [DIJA] = DIJA paper, [DG] = DiffuGuard paper, [CN] = Context Nesting paper
#
# DIJA-specific params:
#   --attack_method DIJA          : [DIJA] the DIJA interleaved mask-text attack (Sec 3.2)
#   --fill_all_masks              : [DIJA] fills all [MASK] positions — core to DIJA design;
#                                   forces model to complete all masked spans
#
# Shared generation params:
#   (no --temperature, --steps)   : uses code defaults in most experiments;
#                                   [DIJA] paper does not specify these for LLaDA;
#                                   [DG] Table 5 uses temp=0.5, steps=64;
#                                   [CN] Sec 5.1 uses steps=32
#   --temperature 0.0 (greedy_block_audit only): explicit greedy decoding for that config
#
# Remasking strategies: same justifications as context_nesting_gen.sh
#   (no --remasking flag)         : default — greedy low-confidence remasking
#   --remasking random            : [DG] fully random (Eq. 6)
#   --remasking adaptive          : [DG] stochastic annealing (Eq. 7-8)
#   --remasking adaptive_step_exp : not in papers — custom exponential decay variant
#
# Stochastic annealing params:
#   --alpha0 0.6 (adaptive)       : differs from [DG] default (α₀=0.3, Table 6);
#                                   0.6 tested in ablation (Table 7)
#   --alpha0 0.9, --c 0.12, --m 3, --ratio 3.0 : not in papers — custom exp schedule
#
# Block-level audit params (sp_mode=hidden):
#   --sp_threshold 0.2            : differs from [DG] default (λ=0.1, Table 6);
#                                   0.2 tested in ablation (Table 8)
#   --refinement_steps 8          : [DG] same as Table 6 (extra_steps=8)
#   --remask_ratio 0.9            : [DG] same as Table 6 (γ=0.9)
# ─────────────────────────────────────────────────────────────────────────

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/Saftey_Critical_CW_code/venv/bin/activate

cd ~/Saftey_Critical_CW_code

PYTHON=$(which python)
MODEL_PATH="$HOME/Saftey_Critical_CW_code/hf_models/LLaDA-8B-Instruct"
REFINED_PROMPTS="$HOME/Saftey_Critical_CW_code/data/refined_prompt_data/harmbench_refined.json"

BASEDIR="results/dija"

if [ ! -f "$REFINED_PROMPTS" ]; then
    echo "ERROR: $REFINED_PROMPTS not found. Run run_refinement_all.sh first."
    exit 1
fi

echo "=============================================="
echo "HarmBench DIJA Attack - Generation"
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
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --fill_all_masks \
    --sp_mode off \
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
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode off \
    --fill_all_masks \
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
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode off \
    --fill_all_masks \
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
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode off \
    --fill_all_masks \
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
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --fill_all_masks \
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
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --fill_all_masks \
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
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --fill_all_masks \
    --temperature 0.0 \
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
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "${EXPDIR}/generation.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --fill_all_masks \
    --remasking adaptive_step_exp \
    --alpha0 0.9 \
    --c 0.12 \
    --m 3 \
    --ratio 3.0 \
    --debug_print
echo "Done: ${EXPDIR}/generation.json"

echo ""
echo "=============================================="
echo "Generation done. Run harmbench_dija_eval.sh next."
echo "=============================================="
