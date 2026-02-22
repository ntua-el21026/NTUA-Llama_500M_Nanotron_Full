#!/bin/bash
#SBATCH --job-name=SFT_stage3_run1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=normal

set -euo pipefail

module purge
module load python/3.11
module load cuda/12.1
module load gcc/12

GCC_HOME=$(dirname $(dirname $(which g++)))
export LD_PRELOAD="$GCC_HOME/lib64/libstdc++.so.6"
export LD_LIBRARY_PATH="$GCC_HOME/lib64:$LD_LIBRARY_PATH"

source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron_env/bin/activate

SCR=/leonardo_scratch/large/userexternal/mpeppas0/sft_project
PRE_BASE=/leonardo_scratch/large/userexternal/mpeppas0/ckpts/stage3

STEP=$(cat $PRE_BASE/latest.txt | tr -d ' \n')
PRE=$PRE_BASE/$STEP
echo "Using stage3 checkpoint: $PRE"

mkdir -p $SCR/{ckpts,logs,init_from_stage3}

if [ ! -d "$SCR/init_from_stage3/model" ]; then
  echo "Creating weights-only init at $SCR/init_from_stage3"
  rsync -a $PRE/model/ $SCR/init_from_stage3/model/
  cp $PRE/model_config.json $SCR/init_from_stage3/
fi

export HF_HOME=/leonardo_scratch/large/userexternal/mpeppas0/sft_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export WANDB_PROJECT="sft_stage3_smoltalk"
export WANDB_NAME="run1-$(date +%Y%m%d_%H%M%S)"
export WANDB_MODE="offline"

export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

CONFIG=$HOME/Smol_Project/config/sft_stage3_smoltalk_train.yaml
MASTER_PORT=$((12000 + RANDOM % 20000))

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
  $HOME/Smol_Project/scripts/run_train.py \
  --config-file $CONFIG
