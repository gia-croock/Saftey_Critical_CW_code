#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N final_harmbench_dija_gen
#PBS -o eval_logs/final_harmbench_dija_gen.log
#PBS -j oe

# Runs DIJA attack on HarmBench refined prompts across 5 remasking strategies.
# Requires: data/refined_prompt_data/harmbench_refined.json
#   (generate with run_refinement_all.sh if not already done)
# Evaluation is in final_harmbench_dija_eval.sh.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/DiffuGuard/venv/bin/activate

cd ~/DiffuGuard

PYTHON=$(which python)
MODEL_PATH="$HOME/DiffuGuard/hf_models/LLaDA-8B-Instruct"
REFINED_PROMPTS="$HOME/DiffuGuard/data/refined_prompt_data/harmbench_refined.json"

mkdir -p results

if [ ! -f "$REFINED_PROMPTS" ]; then
    echo "ERROR: $REFINED_PROMPTS not found. Run run_refinement_all.sh first."
    exit 1
fi

echo "=============================================="
echo "HarmBench DIJA Attack - Generation"
echo "5 strategies = 5 runs"
echo "=============================================="

# ==============================================================
# no_defense
# ==============================================================
echo ""
echo "--- Strategy: no_defense ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "results/out_harmbench_dija_no_defense.json" \
    --fill_all_masks \
    --sp_mode off \
    --debug_print
echo "Done: results/out_harmbench_dija_no_defense.json"


# ==============================================================
# fully_random
# ==============================================================
echo ""
echo "--- Strategy: fully_random ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "results/out_harmbench_dija_fully_random.json" \
    --sp_mode off \
    --fill_all_masks \
    --remasking random \
    --debug_print
echo "Done: results/out_harmbench_dija_fully_random.json"

# ==============================================================
# stochastic_anealing
# ==============================================================
echo ""
echo "--- Strategy: stochastic_anealing ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "results/out_harmbench_dija_stochastic_anealing.json" \
    --sp_mode off \
    --fill_all_masks \
    --remasking adaptive \
    --alpha0 0.6 \
    --debug_print
echo "Done: results/out_harmbench_dija_stochastic_anealing.json"

# ==============================================================
# stochastic_anealing_exp
# ==============================================================
echo ""
echo "--- Strategy: stochastic_anealing_exp ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "results/out_harmbench_dija_stochastic_anealing_exp.json" \
    --sp_mode off \
    --fill_all_masks \
    --remasking adaptive_step_exp \
    --alpha0 0.9 \
    --c 0.12 \
    --m 3 \
    --ratio 3.0 \
    --debug_print
echo "Done: results/out_harmbench_dija_stochastic_anealing_exp.json"

# ==============================================================
# fully_random (block level audit active)
# ==============================================================
echo ""
echo "--- Strategy: fully_random (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "results/out_harmbench_dija_fully_random_defense.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --fill_all_masks \
    --remasking random \
    --debug_print
echo "Done: results/out_harmbench_dija_fully_random_defense.json"

# ==============================================================
# stochastic_anealing (block level audit active)
# ==============================================================
echo ""
echo "--- Strategy: stochastic_anealing (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "results/out_harmbench_dija_stochastic_anealing_defense.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --fill_all_masks \
    --remasking adaptive \
    --alpha0 0.6 \
    --debug_print
echo "Done: results/out_harmbench_dija_stochastic_anealing_defense.json"

# ==============================================================
# greedy (block level audit active)
# ==============================================================
echo ""
echo "--- Strategy: greedy (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "results/out_harmbench_dija_greedy_defense.json" \
    --sp_mode hidden \
    --sp_threshold 0.2 \
    --refinement_steps 8 \
    --remask_ratio 0.9 \
    --fill_all_masks \
    --temperature 0.0 \
    --debug_print
echo "Done: results/out_harmbench_dija_greedy_defense.json"

# ==============================================================
# stochastic_anealing_exp (block level audit active)
# ==============================================================
echo ""
echo "--- Strategy: stochastic_anealing_exp (defense) ---"
$PYTHON models/jailbreakbench_llada.py \
    --model_path "${MODEL_PATH}" \
    --attack_method DIJA \
    --attack_prompt "${REFINED_PROMPTS}" \
    --output_json "results/out_harmbench_dija_stochastic_anealing_exp_defense.json" \
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
echo "Done: results/out_harmbench_dija_stochastic_anealing_exp_defense.json"

echo ""
echo "=============================================="
echo "Generation done. Run final_harmbench_dija_eval.sh next."
echo "=============================================="
