#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -N dream_sanity_check_all
#PBS -o eval_logs/dream_sanity_check_all.log
#PBS -j oe

# Runs 1 prompt through each attack method (no_defense only) to verify
# Dream is generating correctly before committing to a full run.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source ~/Saftey_Critical_CW_code/venv/bin/activate

cd ~/Saftey_Critical_CW_code

PYTHON=$(which python)
MODEL_PATH="$HOME/Saftey_Critical_CW_code/hf_models/Dream-v0-Instruct-7B"
OUTDIR="dream_results/sanity_check"
PROMPTDIR="dream_results/sanity_check/prompts"
mkdir -p "$OUTDIR" "$PROMPTDIR"

# Extract 1 prompt from each dataset
$PYTHON -c "
import json

outdir = 'dream_results/sanity_check/prompts'
datasets = {
    'dija':            'data/refined_prompt_data/harmbench_refined.json',
    'context_nesting': 'data/refined_prompt_data/harmbench_context_nesting_refined.json',
    'harmless':        'data/refined_prompt_data/harmless_context_nesting_refined.json',
}
for name, path in datasets.items():
    data = json.load(open(path))
    json.dump(data[4:5], open(f'{outdir}/{name}_prompt.json', 'w'), indent=2)
    item = data[4]
    vanilla = item.get('vanilla prompt') or item.get('goal') or ''
    print(f'[{name}] prompt: {vanilla[:100]}')
"

echo ""
echo "=============================================="
echo "1) DIJA (no_defense)"
echo "=============================================="
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method DIJA \
    --attack_prompt "${PROMPTDIR}/dija_prompt.json" \
    --output_json "${OUTDIR}/dija_out.json" \
    --remasking off \
    --sp_mode off \
    --steps 64 \
    --gen_length 128 \
    --temperature 0.5

echo ""
echo "=============================================="
echo "2) Context Nesting (no_defense)"
echo "=============================================="
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${PROMPTDIR}/context_nesting_prompt.json" \
    --output_json "${OUTDIR}/context_nesting_out.json" \
    --remasking off \
    --sp_mode off \
    --steps 64 \
    --gen_length 128 \
    --temperature 0.5 \
    --mask_counts 0

echo ""
echo "=============================================="
echo "3) Harmless / context nesting (no_defense)"
echo "=============================================="
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method zeroshot \
    --attack_prompt "${PROMPTDIR}/harmless_prompt.json" \
    --output_json "${OUTDIR}/harmless_out.json" \
    --remasking off \
    --sp_mode off \
    --steps 64 \
    --gen_length 128 \
    --temperature 0.5 \
    --mask_counts 0

echo ""
echo "=============================================="
echo "4) Context Nesting + SPD defense"
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

echo ""
echo "=============================================="
echo "Results"
echo "=============================================="
$PYTHON - <<'EOF'
import json, os

cases = {
    "DIJA":                    "dream_results/sanity_check/dija_out.json",
    "Context Nesting":         "dream_results/sanity_check/context_nesting_out.json",
    "Harmless":                "dream_results/sanity_check/harmless_out.json",
    "Context Nesting + SPD":   "dream_results/sanity_check/context_nesting_spd_out.json",
}

for name, path in cases.items():
    if not os.path.exists(path):
        print(f"[{name}] NO OUTPUT FILE - generation failed")
        continue
    item = json.load(open(path))[0]
    response = item.get("response", "[empty]")
    print(f"[{name}]")
    print(f"  Prompt:   {item.get('vanilla prompt', '')[:100]}")
    print(f"  Response: {response[:300]}")
    print()
EOF
