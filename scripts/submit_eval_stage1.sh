#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "Usage: $0"
  echo "Submits fixed Stage-1 checkpoints: 2000 10000 20000 32000 40000 50000"
  echo "Optional env override: LIGHTEVAL_CMD (default: bash scripts/run_lighteval_stage1_real.sh)"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${SCRATCH:?SCRATCH is not set. Run this on Leonardo where SCRATCH is defined.}"
: "${LIGHTEVAL_CMD:=bash scripts/run_lighteval_stage1_real.sh}"
export LIGHTEVAL_CMD

CONFIG_PATH="$REPO_ROOT/config/config_stage1.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "Missing training config: $CONFIG_PATH"
  exit 1
fi

TRAIN_TOKENIZER=$(
  awk -F': *' '
    /^[[:space:]]*tokenizer_name_or_path:[[:space:]]*/ {
      gsub(/["'"'"']/, "", $2)
      print $2
      exit
    }
  ' "$CONFIG_PATH"
)

TRAIN_SEQ_LEN=$(
  awk '
    /^tokens:[[:space:]]*$/ { in_tokens=1; next }
    in_tokens && /^[^[:space:]]/ { in_tokens=0 }
    in_tokens && /^[[:space:]]*sequence_length:[[:space:]]*/ {
      print $2
      exit
    }
  ' "$CONFIG_PATH"
)

if [ -z "$TRAIN_TOKENIZER" ]; then
  echo "Could not read tokenizer_name_or_path from $CONFIG_PATH"
  exit 1
fi
if [ -z "$TRAIN_SEQ_LEN" ]; then
  echo "Could not read tokens.sequence_length from $CONFIG_PATH"
  exit 1
fi

CKPT_ROOT="$SCRATCH/ckpts/stage1"
RESULT_ROOT="$SCRATCH/evaluation_results/stage1"
EVAL_DATA_DIR="$RESULT_ROOT/eval_data"
EVAL_RESULTS_DIR="$RESULT_ROOT"
EVAL_LOG_DIR="$RESULT_ROOT/eval_logs"
SLURM_LOG_DIR="$EVAL_LOG_DIR/slurm"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-$TRAIN_TOKENIZER}"
SEQ_LEN="${SEQ_LEN:-$TRAIN_SEQ_LEN}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

STEPS=(2000 10000 20000 32000 40000 50000)

if [[ "$TOKENIZER_NAME_OR_PATH" == /* ]] && [ ! -d "$TOKENIZER_NAME_OR_PATH" ]; then
  echo "Tokenizer path not found: $TOKENIZER_NAME_OR_PATH"
  exit 1
fi

mkdir -p "$EVAL_DATA_DIR" "$EVAL_RESULTS_DIR" "$EVAL_LOG_DIR" "$SLURM_LOG_DIR"

cd "$REPO_ROOT"
for STEP in "${STEPS[@]}"; do
  CHECKPOINT_PATH="$CKPT_ROOT/$STEP"
  if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Missing checkpoint directory: $CHECKPOINT_PATH"
    exit 1
  fi
  if [ -z "$(ls -A "$CHECKPOINT_PATH" 2>/dev/null)" ]; then
    echo "Checkpoint directory is empty: $CHECKPOINT_PATH"
    exit 1
  fi

  if [ ! -f "$CHECKPOINT_PATH/config.yaml" ] || [ ! -f "$CHECKPOINT_PATH/model_config.json" ] || [ ! -d "$CHECKPOINT_PATH/model" ]; then
    echo "Checkpoint is not Nanotron-layout: $CHECKPOINT_PATH"
    echo "Expected: config.yaml, model_config.json, and model/ directory"
    exit 1
  fi

  EXPORTS="ALL,CHECKPOINT_PATH=${CHECKPOINT_PATH},STEP=${STEP},TOKENIZER_NAME_OR_PATH=${TOKENIZER_NAME_OR_PATH},SEQ_LEN=${SEQ_LEN},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},EVAL_DATA_DIR=${EVAL_DATA_DIR},EVAL_RESULTS_DIR=${EVAL_RESULTS_DIR},EVAL_LOG_DIR=${EVAL_LOG_DIR},HF_DATASETS_TRUST_REMOTE_CODE=1"

  sbatch \
    --output="$SLURM_LOG_DIR/eval_stage1_${STEP}_%j.out" \
    --error="$SLURM_LOG_DIR/eval_stage1_${STEP}_%j.err" \
    --export="$EXPORTS" \
    slurm/eval_stage1_manual.sbatch
done
