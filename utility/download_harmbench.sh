#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -N download_harmbench_cls

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0

source ~/DiffuGuard/venv/bin/activate

pip install huggingface_hub --quiet

# Download HarmBench classifier to a local folder
# cais/HarmBench-Llama-2-13b-cls is ~26GB in bf16
huggingface-cli download cais/HarmBench-Llama-2-13b-cls \
    --local-dir ~/hf_models/HarmBench-Llama-2-13b-cls \
    --local-dir-use-symlinks False

echo "=== Download complete ==="
echo "Model saved to: ~/hf_models/HarmBench-Llama-2-13b-cls"