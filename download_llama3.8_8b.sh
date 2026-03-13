#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -N download_llama2_7b

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0

source ~/Saftey_Critical_CW_code/venv/bin/activate

export HF_TOKEN=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

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='meta-llama/Llama-3.1-8B', local_dir='/rds/general/user/gsc125/home/hf_models/Llama-3.1-8B', token='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')"

echo "Done. Model saved to: ~/hf_models/Llama-3.1-8B"
