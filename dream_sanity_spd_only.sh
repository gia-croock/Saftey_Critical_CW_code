#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N dream_sanity_spd_only
#PBS -o eval_logs/dream_sanity_spd_only.log
#PBS -j oe

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/Saftey_Critical_CW_code/venv/bin/activate

cd ~/Saftey_Critical_CW_code

PYTHON=$(which python)
MODEL_PATH="$HOME/Saftey_Critical_CW_code/hf_models/Dream-v0-Instruct-7B"
OUTDIR="dream_results/sanity_check"
PROMPTDIR="dream_results/sanity_check/prompts"

echo "=============================================="
echo "4) Context Nesting + SPD defense (debug)"
echo "=============================================="
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${PROMPTDIR}/context_nesting_prompt.json" \
    --output_json "${OUTDIR}/context_nesting_spd_out.json" \
    --remasking off \
    --spd_k 5 \
    --sp_mode off \
    --steps 64 \
    --gen_length 128 \
    --temperature 0.5 \
    --mask_counts 0 \
    --debug_print

echo "Done. Check eval_logs/dream_sanity_spd_only.log"
