#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N final_harmbench_context_nesting_eval
#PBS -o eval_logs/final_harmbench_context_nesting_eval.log
#PBS -j oe

# Evaluates HarmBench Context Nesting generation outputs (ASR_e + ASR_k + PPL + SAR).
# Each strategy produces one consolidated results file.
# Run final_harmbench_context_nesting_gen.sh first.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/DiffuGuard/venv/bin/activate

cd ~/DiffuGuard

PYTHON=$(which python)
HARMBENCH_MODEL="$HOME/hf_models/HarmBench-Llama-2-13b-cls"
PPL_MODEL="$HOME/hf_models/Llama-3.1-8B"

# ==============================================================
# no_defense
# ==============================================================
echo "=============================================="
echo "Evaluating: no_defense"
echo "=============================================="
GEN_JSON="results/out_harmbench_context_nesting_no_defense.json"
HARMBENCH_EVAL_JSON="results/out_harmbench_context_nesting_no_defense_harmbench_eval.json"
RESULTS_JSON="results/harmbench_context_nesting_no_defense_results.json"

if [ -f "$GEN_JSON" ]; then
    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py --input "$GEN_JSON" --model_path "${HARMBENCH_MODEL}" --batch_size 8
    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py --input "$HARMBENCH_EVAL_JSON" --output "$RESULTS_JSON"
    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py --model_id "${PPL_MODEL}" --json_file "$RESULTS_JSON" --text_key "response" --stride 512 --output_json "$RESULTS_JSON"
    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py --input "$RESULTS_JSON" --output "$RESULTS_JSON"
    echo "Done: $RESULTS_JSON"
else
    echo "WARNING: $GEN_JSON not found, skipping."
fi

# ==============================================================
# fully_random
# ==============================================================
echo ""
echo "=============================================="
echo "Evaluating: fully_random"
echo "=============================================="
GEN_JSON="results/out_harmbench_context_nesting_fully_random.json"
HARMBENCH_EVAL_JSON="results/out_harmbench_context_nesting_fully_random_harmbench_eval.json"
RESULTS_JSON="results/harmbench_context_nesting_fully_random_results.json"

if [ -f "$GEN_JSON" ]; then
    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py --input "$GEN_JSON" --model_path "${HARMBENCH_MODEL}" --batch_size 8
    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py --input "$HARMBENCH_EVAL_JSON" --output "$RESULTS_JSON"
    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py --model_id "${PPL_MODEL}" --json_file "$RESULTS_JSON" --text_key "response" --stride 512 --output_json "$RESULTS_JSON"
    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py --input "$RESULTS_JSON" --output "$RESULTS_JSON"
    echo "Done: $RESULTS_JSON"
else
    echo "WARNING: $GEN_JSON not found, skipping."
fi

# ==============================================================
# stochastic_anealing
# ==============================================================
echo ""
echo "=============================================="
echo "Evaluating: stochastic_anealing"
echo "=============================================="
GEN_JSON="results/out_harmbench_context_nesting_stochastic_anealing.json"
HARMBENCH_EVAL_JSON="results/out_harmbench_context_nesting_stochastic_anealing_harmbench_eval.json"
RESULTS_JSON="results/harmbench_context_nesting_stochastic_anealing_results.json"

if [ -f "$GEN_JSON" ]; then
    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py --input "$GEN_JSON" --model_path "${HARMBENCH_MODEL}" --batch_size 8
    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py --input "$HARMBENCH_EVAL_JSON" --output "$RESULTS_JSON"
    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py --model_id "${PPL_MODEL}" --json_file "$RESULTS_JSON" --text_key "response" --stride 512 --output_json "$RESULTS_JSON"
    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py --input "$RESULTS_JSON" --output "$RESULTS_JSON"
    echo "Done: $RESULTS_JSON"
else
    echo "WARNING: $GEN_JSON not found, skipping."
fi

# ==============================================================
# stochastic_anealing_exp
# ==============================================================
echo ""
echo "=============================================="
echo "Evaluating: stochastic_anealing_exp"
echo "=============================================="
GEN_JSON="results/out_harmbench_context_nesting_stochastic_anealing_exp.json"
HARMBENCH_EVAL_JSON="results/out_harmbench_context_nesting_stochastic_anealing_exp_harmbench_eval.json"
RESULTS_JSON="results/harmbench_context_nesting_stochastic_anealing_exp_results.json"

if [ -f "$GEN_JSON" ]; then
    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py --input "$GEN_JSON" --model_path "${HARMBENCH_MODEL}" --batch_size 8
    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py --input "$HARMBENCH_EVAL_JSON" --output "$RESULTS_JSON"
    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py --model_id "${PPL_MODEL}" --json_file "$RESULTS_JSON" --text_key "response" --stride 512 --output_json "$RESULTS_JSON"
    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py --input "$RESULTS_JSON" --output "$RESULTS_JSON"
    echo "Done: $RESULTS_JSON"
else
    echo "WARNING: $GEN_JSON not found, skipping."
fi

# ==============================================================
# fully_random (block level audit active)
# ==============================================================
echo ""
echo "=============================================="
echo "Evaluating: fully_random (defense)"
echo "=============================================="
GEN_JSON="results/out_harmbench_context_nesting_fully_random_defense.json"
HARMBENCH_EVAL_JSON="results/out_harmbench_context_nesting_fully_random_defense_harmbench_eval.json"
RESULTS_JSON="results/harmbench_context_nesting_fully_random_defense_results.json"

if [ -f "$GEN_JSON" ]; then
    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py --input "$GEN_JSON" --model_path "${HARMBENCH_MODEL}" --batch_size 8
    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py --input "$HARMBENCH_EVAL_JSON" --output "$RESULTS_JSON"
    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py --model_id "${PPL_MODEL}" --json_file "$RESULTS_JSON" --text_key "response" --stride 512 --output_json "$RESULTS_JSON"
    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py --input "$RESULTS_JSON" --output "$RESULTS_JSON"
    echo "Done: $RESULTS_JSON"
else
    echo "WARNING: $GEN_JSON not found, skipping."
fi

# ==============================================================
# stochastic_anealing (block level audit active)
# ==============================================================
echo ""
echo "=============================================="
echo "Evaluating: stochastic_anealing (defense)"
echo "=============================================="
GEN_JSON="results/out_harmbench_context_nesting_stochastic_anealing_defense.json"
HARMBENCH_EVAL_JSON="results/out_harmbench_context_nesting_stochastic_anealing_defense_harmbench_eval.json"
RESULTS_JSON="results/harmbench_context_nesting_stochastic_anealing_defense_results.json"

if [ -f "$GEN_JSON" ]; then
    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py --input "$GEN_JSON" --model_path "${HARMBENCH_MODEL}" --batch_size 8
    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py --input "$HARMBENCH_EVAL_JSON" --output "$RESULTS_JSON"
    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py --model_id "${PPL_MODEL}" --json_file "$RESULTS_JSON" --text_key "response" --stride 512 --output_json "$RESULTS_JSON"
    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py --input "$RESULTS_JSON" --output "$RESULTS_JSON"
    echo "Done: $RESULTS_JSON"
else
    echo "WARNING: $GEN_JSON not found, skipping."
fi

# ==============================================================
# greedy (block level audit active)
# ==============================================================
echo ""
echo "=============================================="
echo "Evaluating: greedy (defense)"
echo "=============================================="
GEN_JSON="results/out_harmbench_context_nesting_greedy_defense.json"
HARMBENCH_EVAL_JSON="results/out_harmbench_context_nesting_greedy_defense_harmbench_eval.json"
RESULTS_JSON="results/harmbench_context_nesting_greedy_defense_results.json"

if [ -f "$GEN_JSON" ]; then
    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py --input "$GEN_JSON" --model_path "${HARMBENCH_MODEL}" --batch_size 8
    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py --input "$HARMBENCH_EVAL_JSON" --output "$RESULTS_JSON"
    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py --model_id "${PPL_MODEL}" --json_file "$RESULTS_JSON" --text_key "response" --stride 512 --output_json "$RESULTS_JSON"
    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py --input "$RESULTS_JSON" --output "$RESULTS_JSON"
    echo "Done: $RESULTS_JSON"
else
    echo "WARNING: $GEN_JSON not found, skipping."
fi

# ==============================================================
# stochastic_anealing_exp (block level audit active)
# ==============================================================
echo ""
echo "=============================================="
echo "Evaluating: stochastic_anealing_exp (defense)"
echo "=============================================="
GEN_JSON="results/out_harmbench_context_nesting_stochastic_anealing_exp_defense.json"
HARMBENCH_EVAL_JSON="results/out_harmbench_context_nesting_stochastic_anealing_exp_defense_harmbench_eval.json"
RESULTS_JSON="results/harmbench_context_nesting_stochastic_anealing_exp_defense_results.json"

if [ -f "$GEN_JSON" ]; then
    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py --input "$GEN_JSON" --model_path "${HARMBENCH_MODEL}" --batch_size 8
    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py --input "$HARMBENCH_EVAL_JSON" --output "$RESULTS_JSON"
    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py --model_id "${PPL_MODEL}" --json_file "$RESULTS_JSON" --text_key "response" --stride 512 --output_json "$RESULTS_JSON"
    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py --input "$RESULTS_JSON" --output "$RESULTS_JSON"
    echo "Done: $RESULTS_JSON"
else
    echo "WARNING: $GEN_JSON not found, skipping."
fi

# ==============================================================
# spd (no block audit)
# ==============================================================
echo ""
echo "=============================================="
echo "Evaluating: spd"
echo "=============================================="
GEN_JSON="results/out_harmbench_context_nesting_spd.json"
HARMBENCH_EVAL_JSON="results/out_harmbench_context_nesting_spd_harmbench_eval.json"
RESULTS_JSON="results/harmbench_context_nesting_spd_results.json"

if [ -f "$GEN_JSON" ]; then
    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py --input "$GEN_JSON" --model_path "${HARMBENCH_MODEL}" --batch_size 8
    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py --input "$HARMBENCH_EVAL_JSON" --output "$RESULTS_JSON"
    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py --model_id "${PPL_MODEL}" --json_file "$RESULTS_JSON" --text_key "response" --stride 512 --output_json "$RESULTS_JSON"
    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py --input "$RESULTS_JSON" --output "$RESULTS_JSON"
    echo "Done: $RESULTS_JSON"
else
    echo "WARNING: $GEN_JSON not found, skipping."
fi

# ==============================================================
# spd (with block audit)
# ==============================================================
echo ""
echo "=============================================="
echo "Evaluating: spd (defense)"
echo "=============================================="
GEN_JSON="results/out_harmbench_context_nesting_spd_defense.json"
HARMBENCH_EVAL_JSON="results/out_harmbench_context_nesting_spd_defense_harmbench_eval.json"
RESULTS_JSON="results/harmbench_context_nesting_spd_defense_results.json"

if [ -f "$GEN_JSON" ]; then
    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py --input "$GEN_JSON" --model_path "${HARMBENCH_MODEL}" --batch_size 8
    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py --input "$HARMBENCH_EVAL_JSON" --output "$RESULTS_JSON"
    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py --model_id "${PPL_MODEL}" --json_file "$RESULTS_JSON" --text_key "response" --stride 512 --output_json "$RESULTS_JSON"
    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py --input "$RESULTS_JSON" --output "$RESULTS_JSON"
    echo "Done: $RESULTS_JSON"
else
    echo "WARNING: $GEN_JSON not found, skipping."
fi

# ==============================================================
# spd + self-reminder (no block audit)
# ==============================================================
echo ""
echo "=============================================="
echo "Evaluating: spd + self-reminder"
echo "=============================================="
GEN_JSON="results/out_harmbench_context_nesting_spd_reminder.json"
HARMBENCH_EVAL_JSON="results/out_harmbench_context_nesting_spd_reminder_harmbench_eval.json"
RESULTS_JSON="results/harmbench_context_nesting_spd_reminder_results.json"

if [ -f "$GEN_JSON" ]; then
    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py --input "$GEN_JSON" --model_path "${HARMBENCH_MODEL}" --batch_size 8
    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py --input "$HARMBENCH_EVAL_JSON" --output "$RESULTS_JSON"
    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py --model_id "${PPL_MODEL}" --json_file "$RESULTS_JSON" --text_key "response" --stride 512 --output_json "$RESULTS_JSON"
    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py --input "$RESULTS_JSON" --output "$RESULTS_JSON"
    echo "Done: $RESULTS_JSON"
else
    echo "WARNING: $GEN_JSON not found, skipping."
fi

# ==============================================================
# spd + self-reminder (with block audit)
# ==============================================================
echo ""
echo "=============================================="
echo "Evaluating: spd + self-reminder (defense)"
echo "=============================================="
GEN_JSON="results/out_harmbench_context_nesting_spd_reminder_defense.json"
HARMBENCH_EVAL_JSON="results/out_harmbench_context_nesting_spd_reminder_defense_harmbench_eval.json"
RESULTS_JSON="results/harmbench_context_nesting_spd_reminder_defense_results.json"

if [ -f "$GEN_JSON" ]; then
    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py --input "$GEN_JSON" --model_path "${HARMBENCH_MODEL}" --batch_size 8
    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py --input "$HARMBENCH_EVAL_JSON" --output "$RESULTS_JSON"
    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py --model_id "${PPL_MODEL}" --json_file "$RESULTS_JSON" --text_key "response" --stride 512 --output_json "$RESULTS_JSON"
    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py --input "$RESULTS_JSON" --output "$RESULTS_JSON"
    echo "Done: $RESULTS_JSON"
else
    echo "WARNING: $GEN_JSON not found, skipping."
fi

echo ""
echo "=============================================="
echo "All evaluations complete. Results files:"
echo "  results/harmbench_context_nesting_no_defense_results.json"
echo "  results/harmbench_context_nesting_fully_random_results.json"
echo "  results/harmbench_context_nesting_stochastic_anealing_results.json"
echo "  results/harmbench_context_nesting_stochastic_anealing_exp_results.json"
echo "  results/harmbench_context_nesting_fully_random_defense_results.json"
echo "  results/harmbench_context_nesting_stochastic_anealing_defense_results.json"
echo "  results/harmbench_context_nesting_greedy_defense_results.json"
echo "  results/harmbench_context_nesting_stochastic_anealing_exp_defense_results.json"
echo "  results/harmbench_context_nesting_spd_results.json"
echo "  results/harmbench_context_nesting_spd_defense_results.json"
echo "  results/harmbench_context_nesting_spd_reminder_results.json"
echo "  results/harmbench_context_nesting_spd_reminder_defense_results.json"
echo "=============================================="
