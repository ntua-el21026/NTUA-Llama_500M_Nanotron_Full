#!/bin/bash
#SBATCH --job-name=Run_stage3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:0:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=normal

# --- 1. Environment Setup ---
echo "--- [1] Setting up Environment ---"
module purge
module load python/3.11  # Ή 3.10, ό,τι έβαλες στο setup
module load cuda/12.1    # <-- ΑΥΤΟ που χρησιμοποίησες στο compile
# ΣΗΜΑΝΤΙΚΟ: Φορτώνουμε τον GCC 12 ΤΕΛΕΥΤΑΙΟ για να μπει μπροστά στο path
module load gcc/12

# Βρίσκουμε το μονοπάτι του GCC 12
GCC_HOME=$(dirname $(dirname $(which g++)))
GCC_LIB="$GCC_HOME/lib64/libstdc++.so.6"

echo "--- Fixing Library Path ---"
echo "Forcing usage of: $GCC_LIB"

# ΤΟ ΑΠΟΛΥΤΟ FIX:
# Αναγκάζουμε το Linux να φορτώσει ΑΥΤΗ τη βιβλιοθήκη πριν από οτιδήποτε άλλο
export LD_PRELOAD="$GCC_LIB"
export LD_LIBRARY_PATH="$GCC_HOME/lib64:$LD_LIBRARY_PATH"

source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron_env/bin/activate

export WANDB_PROJECT="pretrain_stage1"
export WANDB_NAME="stage1-run-$(date +%Y%m%d_%H%M)"
export WANDB_MODE="offline"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache


# Optimization Flags
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

# --- 3. Checkpoint Logic (Auto-Resume) ---
# Αν υπάρχει checkpoint, το Nanotron το βρίσκει μόνο του αρκεί το config να δείχνει στο σωστό path.
# Εμείς απλά βεβαιωνόμαστε ότι το output folder υπάρχει.
CHECKPOINT_PATH="/leonardo_scratch/large/userexternal/mpeppas0/ckpts/stage3"
mkdir -p $CHECKPOINT_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1  # <--- ΣΗΜΑΝΤΙΚΟ ΓΙΑ NANOTRON
# --- 4. Launch Training ---
echo "--- [3] Launching Nanotron Training ---"

# Χρησιμοποιούμε torchrun για 1 GPU
torchrun --nproc_per_node=4 $HOME/Smol_Project/scripts/run_train.py --config-file $HOME/Smol_Project/config/config_stage3.yaml

echo "--- Training Job Completed ---"
