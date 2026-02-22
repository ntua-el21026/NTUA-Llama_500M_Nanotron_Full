#!/bin/bash
#SBATCH --job-name=vibe_sft
#SBATCH --output=/leonardo_scratch/large/userexternal/mpeppas0/sft_project/logs/vibe_sft_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/mpeppas0/sft_project/logs/vibe_sft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=boost_qos_dbg

set -euo pipefail

source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron_env/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "HOST: $(hostname)"
nvidia-smi || true
python -u /leonardo/home/userexternal/mpeppas0/Smol_Project/scripts/vibe_test_sft_batch.py
