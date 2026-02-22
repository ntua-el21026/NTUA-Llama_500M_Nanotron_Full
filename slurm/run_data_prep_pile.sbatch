#!/bin/bash
#SBATCH --job-name=prep_pile_part
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=29000MB
#SBATCH --time=04:00:00
#SBATCH --partition=lrd_all_serial
#SBATCH --gres=tmpfs:10g
#SBATCH --output=logs/prep_pile_%j.out
#SBATCH --error=logs/prep_pile_%j.err

cd /leonardo/home/userexternal/mpeppas0/Smol_Project

# Παράμετροι από command line
SKIP=$1
LIMIT=$2
PART_ID=$3

module load python/3.11
source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron/bin/activate

export HF_HOME="/leonardo_scratch/large/userexternal/mpeppas0/hf_cache"
export OMP_NUM_THREADS=4

# Βάζουμε τα “επιπλέον 5B tokens” σε ξεχωριστό folder (αλλά θα τα βάλεις στο stage1 config)
BASE_OUT="/leonardo_scratch/large/userexternal/mpeppas0/nanoset_stage1_extra5B_pile"
OUTPUT_DIR="${BASE_OUT}/part_${PART_ID}"

mkdir -p logs_pile
mkdir -p "$OUTPUT_DIR"

echo "Running Part ${PART_ID}: Skip=${SKIP}, Limit=${LIMIT}"
echo "Output dir: ${OUTPUT_DIR}"

# Premium subsets από The Pile (quality-first)
INCLUDE="ArXiv,PubMed Central,PubMed Abstracts,Stack Exchange,Github,Wikipedia (en),DM Mathematics,EuroParl,FreeLaw,NIH ExPorter,USPTO"

python scripts/prepare_dataset_pile.py \
    --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B \
    --output-folder "${OUTPUT_DIR}" \
    --n-tasks 4 \
    --skip "${SKIP}" \
    --limit "${LIMIT}" \
    hf \
    --dataset EleutherAI/pile \
    --hf-config all \
    --split train \
    --column text \
    --include-pile-sets "${INCLUDE}"
