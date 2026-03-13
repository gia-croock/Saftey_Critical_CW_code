#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N final_harmbench_vanilla_eval
#PBS -o eval_logs/final_harmbench_vanilla_eval.log
#PBS -j oe

# Evaluates HarmBench generation outputs (ASR_e + ASR_k + PPL).
# All results are written to a single file: results/harmbench_results.json
# Run final_harmbench_gen.sh first.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/DiffuGuard/venv/bin/activate

cd ~/DiffuGuard

PYTHON=$(which python)
HARMBENCH_MODEL="$HOME/hf_models/HarmBench-Llama-2-13b-cls"
PPL_MODEL="$HOME/hf_models/Llama-3.1-8B"

GEN_JSON="results/out_harmbench_vanilla.json" # Output of llada / input of evaluation
HARMBENCH_EVAL_JSON="results/out_harmbench_vanilla_harmbench_eval.json" # qualitative harmbench_evaluation output 
RESULTS_JSON="results/harmbench_results.json" # Final output containing ASR_e, ASR_k, and PPL results

if [ ! -f "$GEN_JSON" ]; then
    echo "ERROR: $GEN_JSON not found. Run final_harmbench_gen.sh first."
    exit 1
fi

# ==============================================================
# STEP 1: ASR_e — HarmBench classifier
# ==============================================================
echo "=============================================="
echo "Step 1: ASR_e — HarmBench Classifier"
echo "=============================================="

$PYTHON evaluate_harmbench_local.py \
    --input "$GEN_JSON" \
    --model_path "${HARMBENCH_MODEL}" \
    --batch_size 8

# ==============================================================
# STEP 2: ASR_k — keyword-based, reads harmbench_eval, writes harmbench_results
# ==============================================================
echo ""
echo "=============================================="
echo "Step 2: ASR_k — Keyword Evaluation"
echo "=============================================="

if [ -f "$HARMBENCH_EVAL_JSON" ]; then
    $PYTHON evaluate_asr_k.py \
        --input "$HARMBENCH_EVAL_JSON" \
        --output "$RESULTS_JSON"
else
    echo "WARNING: $HARMBENCH_EVAL_JSON not found, skipping ASR_k."
fi

# ==============================================================
# STEP 3: PPL — writes per-item scores into harmbench_results
# ==============================================================
echo ""
echo "=============================================="
echo "Step 3: Perplexity (PPL)"
echo "=============================================="

if [ -f "$RESULTS_JSON" ]; then
    $PYTHON analysis/perplexity.py \
        --model_id "${PPL_MODEL}" \
        --json_file "$RESULTS_JSON" \
        --text_key "response" \
        --stride 512 \
        --output_json "$RESULTS_JSON"
else
    echo "WARNING: $RESULTS_JSON not found, skipping PPL."
fi

# ==============================================================
echo ""
echo "=============================================="
echo "Done. Final results: $RESULTS_JSON"
echo "  Contains: ASR_e (harmbench_evaluation)"
echo "            ASR_k (asr_k_evaluation)"
echo "            PPL   (perplexity)"
echo "=============================================="
