#!/bin/bash
#SBATCH --job-name=VIBE_smoltalk
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
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
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export MODEL_DIR=/leonardo_scratch/large/userexternal/mpeppas0/sft_project/hf_exports/sft_smoltalk_5000
# Αν έχεις custom tokenizer με chat template, ξεκλείδωσε:
# export TOKENIZER_DIR=/path/to/your_custom_tokenizer

srun python $HOME/Smol_Project/scripts/vibe_test_sft_5000.py
