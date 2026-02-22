#!/bin/bash
#SBATCH --job-name=VIBE_smoltalk
#SBATCH --output=/leonardo_scratch/large/userexternal/mpeppas0/sft_project/logs/vibe_sft_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/mpeppas0/sft_project/logs/vibe_sft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=boost_qos_dbg

set -euo pipefail

source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron_env/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python -u /leonardo/home/userexternal/mpeppas0/Smol_Project/scripts/vibe_test_sft_chat.py
