#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N final_harmbench_context_nesting_gen
#PBS -o eval_logs/final_harmbench_context_nesting_gen.log
#PBS -j oe

# Runs Context Nesting attack on HarmBench prompts across strategies.
# Context Nesting wraps each behavior in a structural template (Python, LaTeX,
# JSON, Markdown, YAML, or Story) — no inline masks, no --fill_all_masks.
# Requires: data/refined_prompt_data/harmbench_context_nesting.json
#   (generate with run_refinement_all.sh if not already done)
# Evaluation is in final_harmbench_context_nesting_eval.sh.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/DiffuGuard/venv/bin/activate

cd ~/DiffuGuard

PYTHON=$(which python)
MODEL_PATH="$HOME/DiffuGuard/hf_models/LLaDA-8B-Instruct"
HARMBENCH_JSON="$HOME/DiffuGuard/data/pre_refined_prompt/harmbench.json"
CONTEXT_NESTING_PROMPTS="$HOME/DiffuGuard/data/refined_prompt_data/harmbench_context_nesting_refined.json"

mkdir -p results

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
echo ""
echo "--- Strategy: no_defense ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "results/out_harmbench_context_nesting_no_defense.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --debug_print
echo "Done: results/out_harmbench_context_nesting_no_defense.json"

# ==============================================================
# fully_random
# ==============================================================
echo ""
echo "--- Strategy: fully_random ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "results/out_harmbench_context_nesting_fully_random.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --remasking random \
    --debug_print
echo "Done: results/out_harmbench_context_nesting_fully_random.json"

# ==============================================================
# stochastic_anealing
# ==============================================================
echo ""
echo "--- Strategy: stochastic_anealing ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "results/out_harmbench_context_nesting_stochastic_anealing.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --remasking adaptive \
    --alpha0 0.6 \
    --debug_print
echo "Done: results/out_harmbench_context_nesting_stochastic_anealing.json"

# ==============================================================
# stochastic_anealing_exp
# ==============================================================
echo ""
echo "--- Strategy: stochastic_anealing_exp ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "results/out_harmbench_context_nesting_stochastic_anealing_exp.json" \
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
echo "Done: results/out_harmbench_context_nesting_stochastic_anealing_exp.json"

# ==============================================================
# fully_random (block level audit active)
# ==============================================================
echo ""
echo "--- Strategy: fully_random (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "results/out_harmbench_context_nesting_fully_random_defense.json" \
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
echo "Done: results/out_harmbench_context_nesting_fully_random_defense.json"

# ==============================================================
# stochastic_anealing (block level audit active)
# ==============================================================
echo ""
echo "--- Strategy: stochastic_anealing (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "results/out_harmbench_context_nesting_stochastic_anealing_defense.json" \
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
echo "Done: results/out_harmbench_context_nesting_stochastic_anealing_defense.json"

# ==============================================================
# greedy (block level audit active)
# ==============================================================
echo ""
echo "--- Strategy: greedy (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "results/out_harmbench_context_nesting_greedy_defense.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --debug_print
echo "Done: results/out_harmbench_context_nesting_greedy_defense.json"

# ==============================================================
# stochastic_anealing_exp (block level audit active)
# ==============================================================
echo ""
echo "--- Strategy: stochastic_anealing_exp (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "results/out_harmbench_context_nesting_stochastic_anealing_exp_defense.json" \
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
echo "Done: results/out_harmbench_context_nesting_stochastic_anealing_exp_defense.json"

# ==============================================================
# SPD (no block audit)
# ==============================================================
echo ""
echo "--- Strategy: spd ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "results/out_harmbench_context_nesting_spd.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 32 \
    --cfg_scale 0.0 \
    --spd_k 5 \
    --debug_print
echo "Done: results/out_harmbench_context_nesting_spd.json"

# ==============================================================
# SPD (with block audit)
# ==============================================================
echo ""
echo "--- Strategy: spd (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "results/out_harmbench_context_nesting_spd_defense.json" \
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
echo "Done: results/out_harmbench_context_nesting_spd_defense.json"

# ==============================================================
# SPD + self-reminder (no block audit)
# ==============================================================
echo ""
echo "--- Strategy: spd + self-reminder ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "results/out_harmbench_context_nesting_spd_reminder.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 32 \
    --cfg_scale 0.0 \
    --spd_k 5 \
    --defense_method self-reminder \
    --debug_print
echo "Done: results/out_harmbench_context_nesting_spd_reminder.json"

# ==============================================================
# SPD + self-reminder (with block audit)
# ==============================================================
echo ""
echo "--- Strategy: spd + self-reminder (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${CONTEXT_NESTING_PROMPTS}" \
    --output_json "results/out_harmbench_context_nesting_spd_reminder_defense.json" \
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
echo "Done: results/out_harmbench_context_nesting_spd_reminder_defense.json"

echo ""
echo "=============================================="
echo "Generation done. Run final_harmbench_context_nesting_eval.sh next."
echo "=============================================="
