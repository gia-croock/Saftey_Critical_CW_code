#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N dream_harmbench_vanilla_eval
#PBS -o eval_logs/dream_vanilla_eval.log
#PBS -j oe

# Evaluates Dream vanilla HarmBench generation outputs (ASR_e + ASR_k + PPL).
# Mirrors harmbench_vanilla_eval.sh (LLaDA).
# Run dream_harmbench_vanilla_gen.sh first.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/Saftey_Critical_CW_code/venv/bin/activate

cd ~/Saftey_Critical_CW_code

PYTHON=$(which python)
HARMBENCH_MODEL="$HOME/hf_models/HarmBench-Llama-2-13b-cls"
PPL_MODEL="$HOME/hf_models/Llama-3.1-8B"

EXPDIR="dream_results/vanilla/baseline"
GEN_JSON="${EXPDIR}/generation.json"
HARMBENCH_EVAL_JSON="${EXPDIR}/harmbench_eval.json"
RESULTS_JSON="${EXPDIR}/results.json"

if [ ! -f "$GEN_JSON" ]; then
    echo "ERROR: $GEN_JSON not found. Run dream_harmbench_vanilla_gen.sh first."
    exit 1
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

echo ""
echo "=============================================="
echo "Done. Final results: $RESULTS_JSON"
echo "=============================================="
