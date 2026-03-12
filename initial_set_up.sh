#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -N setup_diffuguard

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0


# Create venv if not already created
if [ ! -d ~/DiffuGuard/venv ]; then
  cd ~/DiffuGuard
  python -m venv venv
fi

source ~/DiffuGuard/venv/bin/activate

# Filter out packages that break on cluster:
#   av, decord        - need ffmpeg/system libs, not needed for core experiments
#   nvidia-*          - cluster provides CUDA via modules
#   triton            - cluster provides this or not needed
#   en_core_web_sm    - URL-based install, do separately
#   yt-dlp            - not needed
grep -v "^av==" ~/DiffuGuard/requirements.txt \
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

echo ""
echo "========================================="
echo "  DiffuGuard setup complete!"
echo "========================================="