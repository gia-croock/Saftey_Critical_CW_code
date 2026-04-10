#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N dream_harmless_plain_eval
#PBS -o eval_logs/dream_harmless_plain_eval.log
#PBS -j oe

# Evaluates Dream harmless plain (no attack) outputs (PPL only).
# No ASR or SAR evaluation — these are benign prompts.
# Mirrors harmless_plain_eval.sh (LLaDA).
# Run dream_harmless_plain_gen.sh first.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/Saftey_Critical_CW_code/venv/bin/activate

cd ~/Saftey_Critical_CW_code

PYTHON=$(which python)
PPL_MODEL="$HOME/hf_models/Llama-3.1-8B"

BASEDIR="dream_results/harmless_plain"

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
    RESULTS_JSON="${EXPDIR}/results.json"

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

    echo "Done: $RESULTS_JSON"
done

echo ""
echo "=============================================="
echo "All Dream harmless plain evaluations complete."
echo "=============================================="
