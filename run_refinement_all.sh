#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N run_refinement_all

# Refines raw prompt datasets using Qwen2.5-7B-Instruct.
# Input:  data/pre_refined_prompt/{harmbench}.csv
# Output: data/refined_prompt_data/{harmbench}_refined.json

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/DiffuGuard/venv/bin/activate

cd ~/DiffuGuard

PYTHON=$(which python)
QWEN_MODEL="$HOME/DiffuGuard/hf_models/Qwen2.5-7B-Instruct"
TEMPLATE="refine_prompt/redteam_prompt_template.txt"

mkdir -p data/refined_prompt_data

datasets=( "harmbench" )

echo "=============================================="
echo "Running prompt refinement with Qwen2.5-7B"
echo "=============================================="

for dataset in "${datasets[@]}"; do
    echo ""
    echo "--- Refining: $dataset ---"
    INPUT_CSV="data/pre_refined_prompt/${dataset}.csv"
    OUTPUT_JSON="data/refined_prompt_data/${dataset}_refined.json"

    if [ -f "$INPUT_CSV" ]; then
        $PYTHON utility/refiner.py hf \
            --hf-model-path "${QWEN_MODEL}" \
            --prompt-template-path "${TEMPLATE}" \
            --attack-prompt "${INPUT_CSV}" \
            --output-json "${OUTPUT_JSON}" \
            --max-new-tokens 200
        echo "Saved: $OUTPUT_JSON"
    else
        echo "Warning: $INPUT_CSV not found, skipping"
    fi
done

echo ""
echo "=============================================="
echo "Running context nesting prompt refinement (no LLM)"
echo "=============================================="

for dataset in "${datasets[@]}"; do
    echo ""
    echo "--- Context nesting: $dataset ---"
    INPUT_CSV="data/pre_refined_prompt/${dataset}.csv"
    OUTPUT_JSON="data/refined_prompt_data/${dataset}_context_nesting_refined.json"

    if [ -f "$INPUT_CSV" ]; then
        $PYTHON utility/context_nesting_prompt_refiner.py \
            --attack-prompt "${INPUT_CSV}" \
            --output-json "${OUTPUT_JSON}"
        echo "Saved: $OUTPUT_JSON"
    else
        echo "Warning: $INPUT_CSV not found, skipping"
    fi
done

echo ""
echo "=============================================="
echo "Done! Refined files in data/refined_prompt_data/:"
ls data/refined_prompt_data/
echo "=============================================="
