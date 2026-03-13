#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N final_harmless_context_nesting_gen
#PBS -o eval_logs/final_harmless_context_nesting_gen.log
#PBS -j oe

# Runs Context Nesting on BENIGN prompts across all strategies + defenses.
# Purpose: measure SAR and PPL on harmless inputs as a utility baseline.
# No ASR evaluation (these are not harmful prompts).
# Requires: data/pre_refined_prompt/benign_prompts.csv

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/DiffuGuard/venv/bin/activate

cd ~/DiffuGuard

PYTHON=$(which python)
MODEL_PATH="$HOME/DiffuGuard/hf_models/LLaDA-8B-Instruct"
PPL_MODEL="$HOME/hf_models/Llama-3.1-8B"
BENIGN_CSV="$HOME/DiffuGuard/data/pre_refined_prompt/benign_prompts.csv"
BENIGN_REFINED="$HOME/DiffuGuard/data/refined_prompt_data/harmless_context_nesting_refined.json"

mkdir -p results eval_logs

# ==============================================================
# Step 0: Run context nesting refiner on benign prompts
# ==============================================================
echo "=============================================="
echo "Step 0: Generating context-nested benign prompts"
echo "=============================================="
$PYTHON utility/context_nesting_prompt_refiner.py \
    --attack-prompt "${BENIGN_CSV}" \
    --output-json "${BENIGN_REFINED}" \
    --seed 42
echo "Done: ${BENIGN_REFINED}"

if [ ! -f "$BENIGN_REFINED" ]; then
    echo "ERROR: Refinement failed. Exiting."
    exit 1
fi

# ==============================================================
# no_defense
# ==============================================================
echo ""
echo "--- Strategy: no_defense ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_REFINED}" \
    --output_json "results/out_harmless_context_nesting_no_defense.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --debug_print
echo "Done: results/out_harmless_context_nesting_no_defense.json"

# ==============================================================
# fully_random
# ==============================================================
echo ""
echo "--- Strategy: fully_random ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_REFINED}" \
    --output_json "results/out_harmless_context_nesting_fully_random.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --remasking random \
    --debug_print
echo "Done: results/out_harmless_context_nesting_fully_random.json"

# ==============================================================
# stochastic_annealing
# ==============================================================
echo ""
echo "--- Strategy: stochastic_annealing ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_REFINED}" \
    --output_json "results/out_harmless_context_nesting_stochastic_anealing.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --remasking adaptive \
    --alpha0 0.6 \
    --debug_print
echo "Done: results/out_harmless_context_nesting_stochastic_anealing.json"

# ==============================================================
# stochastic_annealing_exp
# ==============================================================
echo ""
echo "--- Strategy: stochastic_annealing_exp ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_REFINED}" \
    --output_json "results/out_harmless_context_nesting_stochastic_anealing_exp.json" \
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
echo "Done: results/out_harmless_context_nesting_stochastic_anealing_exp.json"

# ==============================================================
# fully_random (defense)
# ==============================================================
echo ""
echo "--- Strategy: fully_random (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_REFINED}" \
    --output_json "results/out_harmless_context_nesting_fully_random_defense.json" \
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
echo "Done: results/out_harmless_context_nesting_fully_random_defense.json"

# ==============================================================
# stochastic_annealing (defense)
# ==============================================================
echo ""
echo "--- Strategy: stochastic_annealing (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_REFINED}" \
    --output_json "results/out_harmless_context_nesting_stochastic_anealing_defense.json" \
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
echo "Done: results/out_harmless_context_nesting_stochastic_anealing_defense.json"

# ==============================================================
# greedy (defense)
# ==============================================================
echo ""
echo "--- Strategy: greedy (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_REFINED}" \
    --output_json "results/out_harmless_context_nesting_greedy_defense.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --mask_counts 0 \
    --temperature 0.0 \
    --steps 32 \
    --cfg_scale 0.0 \
    --debug_print
echo "Done: results/out_harmless_context_nesting_greedy_defense.json"

# ==============================================================
# stochastic_annealing_exp (defense)
# ==============================================================
echo ""
echo "--- Strategy: stochastic_annealing_exp (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_REFINED}" \
    --output_json "results/out_harmless_context_nesting_stochastic_anealing_exp_defense.json" \
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
echo "Done: results/out_harmless_context_nesting_stochastic_anealing_exp_defense.json"

# ==============================================================
# spd (no block audit)
# ==============================================================
echo ""
echo "--- Strategy: spd ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_REFINED}" \
    --output_json "results/out_harmless_context_nesting_spd.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 32 \
    --cfg_scale 0.0 \
    --spd_k 5 \
    --debug_print
echo "Done: results/out_harmless_context_nesting_spd.json"

# ==============================================================
# spd (with block audit)
# ==============================================================
echo ""
echo "--- Strategy: spd (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_REFINED}" \
    --output_json "results/out_harmless_context_nesting_spd_defense.json" \
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
echo "Done: results/out_harmless_context_nesting_spd_defense.json"

# ==============================================================
# spd + self-reminder (no block audit)
# ==============================================================
echo ""
echo "--- Strategy: spd + self-reminder ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_REFINED}" \
    --output_json "results/out_harmless_context_nesting_spd_reminder.json" \
    --sp_mode off \
    --mask_counts 0 \
    --temperature 0.5 \
    --steps 32 \
    --cfg_scale 0.0 \
    --spd_k 5 \
    --defense_method self-reminder \
    --debug_print
echo "Done: results/out_harmless_context_nesting_spd_reminder.json"

# ==============================================================
# spd + self-reminder (with block audit)
# ==============================================================
echo ""
echo "--- Strategy: spd + self-reminder (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${BENIGN_REFINED}" \
    --output_json "results/out_harmless_context_nesting_spd_reminder_defense.json" \
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
echo "Done: results/out_harmless_context_nesting_spd_reminder_defense.json"

echo ""
echo "=============================================="
echo "Generation done. Run final_harmless_context_nesting_eval.sh next."
echo "=============================================="
