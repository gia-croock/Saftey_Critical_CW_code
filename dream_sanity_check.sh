#!/bin/bash
# Quick sanity check for Dream DIJA generation.
# Runs a single prompt through no_defense and prints the response,
# then compares against the equivalent LLaDA result.
# Run this interactively on the cluster before submitting the full PBS job.

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0

cd ~/Saftey_Critical_CW_code
source venv/bin/activate

PYTHON=$(which python)
MODEL_PATH="$HOME/Saftey_Critical_CW_code/hf_models/Dream-v0-Instruct-7B"
REFINED_PROMPTS="data/refined_prompt_data/harmbench_refined.json"
TEST_PROMPT="/tmp/dream_test_prompt.json"
TEST_OUTPUT="/tmp/dream_test_out.json"

echo "=============================================="
echo "Dream Sanity Check - DIJA no_defense (1 prompt)"
echo "=============================================="

# Extract first prompt
$PYTHON -c "
import json
data = json.load(open('${REFINED_PROMPTS}'))
json.dump(data[:1], open('${TEST_PROMPT}', 'w'), indent=2)
print('Test prompt:', data[0].get('vanilla prompt', data[0].get('goal', ''))[:120])
"

echo ""
echo "--- Running Dream generation ---"
$PYTHON models/jailbreakbench_dream.py \
    --model_path "${MODEL_PATH}" \
    --attack_method DIJA \
    --attack_prompt "${TEST_PROMPT}" \
    --output_json "${TEST_OUTPUT}" \
    --remasking off \
    --sp_mode off \
    --steps 64 \
    --gen_length 128 \
    --temperature 0.5 \
    --debug_print

echo ""
echo "=============================================="
echo "Results comparison"
echo "=============================================="
$PYTHON - <<'EOF'
import json

dream = json.load(open("/tmp/dream_test_out.json"))[0]
llada = json.load(open("results/dija/no_defense/generation.json"))[0]

print("Prompt (vanilla):")
print(" ", dream.get("vanilla prompt", "")[:150])
print()
print("Dream response:")
print(" ", dream.get("response", "[empty]")[:400])
print()
print("LLaDA response (same prompt index 0):")
print(" ", llada.get("response", "[empty]")[:400])
EOF
