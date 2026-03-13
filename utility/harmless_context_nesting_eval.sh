#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N final_harmless_context_nesting_eval
#PBS -o eval_logs/final_harmless_context_nesting_eval.log
#PBS -j oe

# Evaluates harmless context nesting outputs (PPL + SAR only).
# No ASR evaluation — these are benign prompts.
# Run final_harmless_context_nesting_gen.sh first.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/DiffuGuard/venv/bin/activate

cd ~/DiffuGuard

PYTHON=$(which python)
PPL_MODEL="$HOME/hf_models/Llama-3.1-8B"

mkdir -p eval_logs

for STRATEGY in \
    "no_defense" \
    "fully_random" \
    "stochastic_anealing" \
    "stochastic_anealing_exp" \
    "fully_random_defense" \
    "stochastic_anealing_defense" \
    "greedy_defense" \
    "stochastic_anealing_exp_defense" \
    "spd" \
    "spd_defense" \
    "spd_reminder" \
    "spd_reminder_defense"
do
    GEN_JSON="results/out_harmless_context_nesting_${STRATEGY}.json"
    RESULTS_JSON="results/harmless_context_nesting_${STRATEGY}_results.json"

    echo ""
    echo "=============================================="
    echo "Evaluating: ${STRATEGY}"
    echo "=============================================="

    if [ ! -f "$GEN_JSON" ]; then
        echo "WARNING: $GEN_JSON not found, skipping."
        continue
    fi

    cp "$GEN_JSON" "$RESULTS_JSON"

    echo "Step 1: PPL"
    $PYTHON analysis/perplexity.py \
        --model_id "${PPL_MODEL}" \
        --json_file "$RESULTS_JSON" \
        --text_key "response" \
        --stride 512 \
        --output_json "$RESULTS_JSON"

    echo "Step 2: SAR"
    $PYTHON evaluate_sar.py \
        --input "$RESULTS_JSON" \
        --output "$RESULTS_JSON"

    echo "Done: $RESULTS_JSON"
done

echo ""
echo "=============================================="
echo "All evaluations complete."
echo "=============================================="