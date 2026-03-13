#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N final_harmbench_context_nesting_eval
#PBS -o eval_logs/context_nesting_eval.log
#PBS -j oe

# Evaluates HarmBench Context Nesting generation outputs (ASR_e + ASR_k + PPL + SAR).
# Each strategy produces one consolidated results file.
# Run harmbench_context_nesting_gen.sh first.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/Saftey_Critical_CW_code/venv/bin/activate

cd ~/Saftey_Critical_CW_code

PYTHON=$(which python)
HARMBENCH_MODEL="$HOME/hf_models/HarmBench-Llama-2-13b-cls"
PPL_MODEL="$HOME/hf_models/Llama-3.1-8B"

BASEDIR="results/context_nesting"

for STRATEGY in \
    "no_defense" \
    "fully_random" \
    "stochastic_annealing" \
    "stochastic_annealing_exp" \
    "fully_random_block_audit" \
    "stochastic_annealing_block_audit" \
    "greedy_block_audit" \
    "stochastic_annealing_exp_block_audit" \
    "spd" \
    "spd_block_audit" \
    "spd_reminder" \
    "spd_reminder_block_audit"
do
    EXPDIR="${BASEDIR}/${STRATEGY}"
    GEN_JSON="${EXPDIR}/generation.json"
    HARMBENCH_EVAL_JSON="${EXPDIR}/harmbench_eval.json"
    RESULTS_JSON="${EXPDIR}/results.json"

    echo ""
    echo "=============================================="
    echo "Evaluating: ${STRATEGY}"
    echo "=============================================="

    if [ ! -f "$GEN_JSON" ]; then
        echo "WARNING: $GEN_JSON not found, skipping."
        continue
    fi

    echo "Step 1: ASR_e"
    $PYTHON evaluate_harmbench_local.py \
        --input "$GEN_JSON" \
        --model_path "${HARMBENCH_MODEL}" \
        --output "$HARMBENCH_EVAL_JSON" \
        --batch_size 8

    echo "Step 2: ASR_k"
    $PYTHON evaluate_asr_k.py \
        --input "$HARMBENCH_EVAL_JSON" \
        --output "$RESULTS_JSON"

    echo "Step 3: PPL"
    $PYTHON analysis/perplexity.py \
        --model_id "${PPL_MODEL}" \
        --json_file "$RESULTS_JSON" \
        --text_key "response" \
        --stride 512 \
        --output_json "$RESULTS_JSON"

    echo "Step 4: SAR"
    $PYTHON evaluate_sar.py \
        --input "$RESULTS_JSON" \
        --output "$RESULTS_JSON"

    echo "Done: $RESULTS_JSON"
done

echo ""
echo "=============================================="
echo "All evaluations complete."
echo "=============================================="
