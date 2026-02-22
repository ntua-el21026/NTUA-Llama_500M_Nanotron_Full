#!/bin/bash
#SBATCH --job-name=FINEWEB_REASONING
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=29000MB
#SBATCH --time=04:00:00
#SBATCH --partition=lrd_all_serial
#SBATCH --gres=tmpfs:10g
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err

# Παράμετροι που θα περάσουμε από το command line
SKIP=$1
LIMIT=$2
PART_ID=$3

module load python/3.11
source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron/bin/activate
export HF_HOME="/leonardo_scratch/large/userexternal/mpeppas0/hf_cache"
export OMP_NUM_THREADS=4

cd "$SLURM_SUBMIT_DIR/.."
echo "Working dir: $(pwd)"

OUTPUT_DIR="/leonardo_scratch/large/userexternal/mpeppas0/STAGE_3_RUN/STAGE_3_FINEWEB/part_${PART_ID}"

echo "Creating directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"  

echo "Running Part $PART_ID: Skip=$SKIP, Limit=$LIMIT"

python scripts/prepare_dataset_FINEWEB.py \
    --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B \
    --output-folder /leonardo_scratch/large/userexternal/mpeppas0/STAGE_3_RUN/STAGE_3_FINEWEB/part_${PART_ID} \
    --n-tasks 4 \
    --skip $SKIP \
    --limit $LIMIT \
    --min-int-score 4 \
    hf \
    --dataset HuggingFaceFW/fineweb-edu \
    --name sample-100BT \
    --split train \
    --column text
