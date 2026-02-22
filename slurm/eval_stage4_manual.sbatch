#!/bin/bash
#SBATCH --job-name=eval_stage4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1

module purge
module load python/3.11
module load gcc/12
source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron_env/bin/activate

cd "$SLURM_SUBMIT_DIR"

CHECKPOINT_PATH=${CHECKPOINT_PATH:?Missing CHECKPOINT_PATH}
STEP=${STEP:?Missing STEP}
TOKENIZER_NAME_OR_PATH=${TOKENIZER_NAME_OR_PATH:?Missing TOKENIZER_NAME_OR_PATH}
LIGHTEVAL_CMD="${LIGHTEVAL_CMD:-bash scripts/run_lighteval_stage4_real.sh}"
export LIGHTEVAL_CMD
SEQ_LEN=${SEQ_LEN:?Missing SEQ_LEN}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:?Missing EVAL_BATCH_SIZE}
EVAL_DATA_DIR=${EVAL_DATA_DIR:?Missing EVAL_DATA_DIR}
EVAL_RESULTS_DIR=${EVAL_RESULTS_DIR:?Missing EVAL_RESULTS_DIR}
EVAL_LOG_DIR=${EVAL_LOG_DIR:?Missing EVAL_LOG_DIR}
export HF_DATASETS_TRUST_REMOTE_CODE=1

if [[ "$CHECKPOINT_PATH" != /* ]]; then
  echo "CHECKPOINT_PATH must be an absolute path: $CHECKPOINT_PATH"
  exit 1
fi
if [ ! -d "$CHECKPOINT_PATH" ]; then
  echo "CHECKPOINT_PATH is not a directory: $CHECKPOINT_PATH"
  exit 1
fi
if ! [[ "$STEP" =~ ^[0-9]+$ ]]; then
  echo "STEP must be an integer, got: $STEP"
  exit 1
fi
if [[ "$TOKENIZER_NAME_OR_PATH" == /* ]] && [ ! -d "$TOKENIZER_NAME_OR_PATH" ]; then
  echo "TOKENIZER_NAME_OR_PATH absolute path not found: $TOKENIZER_NAME_OR_PATH"
  exit 1
fi
if [ ! -f "$CHECKPOINT_PATH/config.yaml" ] || [ ! -f "$CHECKPOINT_PATH/model_config.json" ] || [ ! -d "$CHECKPOINT_PATH/model" ]; then
  echo "CHECKPOINT_PATH is not Nanotron-layout: $CHECKPOINT_PATH"
  echo "Expected: config.yaml, model_config.json, and model/ directory"
  exit 1
fi

mkdir -p "$EVAL_DATA_DIR" "$EVAL_RESULTS_DIR" "$EVAL_LOG_DIR"

echo "=== Stage 4 Nanotron Manual Eval ==="
echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "STEP=$STEP"
echo "TOKENIZER_NAME_OR_PATH=$TOKENIZER_NAME_OR_PATH"
echo "SEQ_LEN=$SEQ_LEN"
echo "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
echo "EVAL_DATA_DIR=$EVAL_DATA_DIR"
echo "EVAL_RESULTS_DIR=$EVAL_RESULTS_DIR"
echo "EVAL_LOG_DIR=$EVAL_LOG_DIR"
echo "LIGHTEVAL_CMD=$LIGHTEVAL_CMD"
echo "RESULT_DIR=$EVAL_RESULTS_DIR/by_checkpoint/step_${STEP}"
echo "=============================="

python -u evaluation/run_eval_stage4.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --step "$STEP" \
  --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH" \
  --eval_data_dir "$EVAL_DATA_DIR" \
  --eval_results_dir "$EVAL_RESULTS_DIR" \
  --seq_len "$SEQ_LEN"

RESULT_DIR="$EVAL_RESULTS_DIR/by_checkpoint/step_${STEP}"
if [ ! -f "$RESULT_DIR/tier2_ppl.json" ] || [ ! -f "$RESULT_DIR/tier3_cf.json" ] || [ ! -f "$RESULT_DIR/tier4_sft_native.json" ]; then
  echo "Missing Tier2/Tier3/Tier4 outputs under $RESULT_DIR"
  exit 1
fi
