#!/bin/bash
#SBATCH --job-name=SFT_stage3_smoltalk
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=boost_qos_dbg

module purge
module load python/3.11
module load cuda/12.1
module load gcc/12

GCC_HOME=$(dirname $(dirname $(which g++)))
export LD_PRELOAD="$GCC_HOME/lib64/libstdc++.so.6"
export LD_LIBRARY_PATH="$GCC_HOME/lib64:$LD_LIBRARY_PATH"

source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron_env/bin/activate

# --- paths ---
SCR=/leonardo_scratch/large/userexternal/mpeppas0/sft_project
PRE_BASE=/leonardo_scratch/large/userexternal/mpeppas0/ckpts/stage3

# πάρε το latest step από latest.txt
STEP=$(cat $PRE_BASE/latest.txt | tr -d ' \n')
PRE=$PRE_BASE/$STEP

echo "Using stage3 checkpoint: $PRE"

mkdir -p $SCR/{ckpts,logs,hf_cache,init_from_stage3}

# --- weights-only init (μοντέλο μόνο) ---
if [ ! -d "$SCR/init_from_stage3/model" ]; then
  echo "Creating weights-only init at $SCR/init_from_stage3"
  rsync -a $PRE/model/ $SCR/init_from_stage3/model/
  cp $PRE/model_config.json $SCR/init_from_stage3/
fi

# --- HF caches ---
export HF_HOME=/leonardo_scratch/large/userexternal/mpeppas0/sft_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_PROJECT="sft_stage3_smoltalk"
export WANDB_NAME="smoltalk-stage3-$(date +%Y%m%d_%H%M)"
export WANDB_MODE="offline"

export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1

CONFIG=$HOME/Smol_Project/config/sft_stage3_smoltalk.yaml

torchrun --nproc_per_node=4 \
  $HOME/Smol_Project/scripts/run_train.py \
  --config-file $CONFIG
