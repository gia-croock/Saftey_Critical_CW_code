#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -N setup_diffuguard

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0


# Create venv if not already created
if [ ! -d ~/Saftey_Critical_CW_code/venv ]; then
  cd ~/Saftey_Critical_CW_code
  python -m venv venv
fi

source ~/Saftey_Critical_CW_code/venv/bin/activate

# Filter out packages that break on cluster:
#   av, decord        - need ffmpeg/system libs, not needed for core experiments
#   nvidia-*          - cluster provides CUDA via modules
#   triton            - cluster provides this or not needed
#   en_core_web_sm    - URL-based install, do separately
#   yt-dlp            - not needed
grep -v "^av==" ~/Saftey_Critical_CW_code/requirements.txt \
  | grep -v "^decord==" \
  | grep -v "^nvidia-" \
  | grep -v "^triton==" \
  | grep -v "^en_core_web_sm" \
  | grep -v "^yt-dlp==" \
  > /tmp/requirements_filtered.txt

pip install --upgrade pip
pip install -r /tmp/requirements_filtered.txt

# Install spacy model separately
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Create results directory structure
echo ""
echo "Creating results directory structure..."
mkdir -p results/vanilla/baseline

for exp in \
    no_defense \
    fully_random \
    stochastic_annealing \
    stochastic_annealing_exp \
    spd \
    spd_reminder \
    fully_random_block_audit \
    stochastic_annealing_block_audit \
    greedy_block_audit \
    stochastic_annealing_exp_block_audit \
    spd_block_audit \
    spd_reminder_block_audit
do
    mkdir -p "results/context_nesting/$exp"
    mkdir -p "results/harmless_context_nesting/$exp"
done

for exp in \
    no_defense \
    fully_random \
    stochastic_annealing \
    stochastic_annealing_exp \
    fully_random_block_audit \
    stochastic_annealing_block_audit \
    greedy_block_audit \
    stochastic_annealing_exp_block_audit
do
    mkdir -p "results/dija/$exp"
done

echo "Results directories created."

echo ""
echo "========================================="
echo "  DiffuGuard setup complete!"
echo "========================================="