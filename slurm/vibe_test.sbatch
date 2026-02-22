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

LOGDIR=/leonardo_scratch/large/userexternal/mpeppas0/sft_project/logs
mkdir -p "$LOGDIR"

source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron_env/bin/activate
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi
python -u /leonardo/home/userexternal/mpeppas0/Smol_Project/scripts/vibetest_sft.py --mode vibe \
  --temperature 0.7 --top_p 0.9 --top_k 50 \
  --repetition_penalty 1.05 --no_repeat_ngram_size 0 \
  --max_new_tokens 100
