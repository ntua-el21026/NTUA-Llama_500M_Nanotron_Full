#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "Usage: $0"
  echo "Submits fixed Stage-2 checkpoints (default): 2000 6000 10000 14000 18000 22000"
  echo "Also submits Stage-1 bridge checkpoint by default (step 50000 into Stage-2 suite)."
  echo "Optional env overrides:"
  echo "  STAGE2_STEPS=\"2000 6000 10000 14000 18000 22000\""
  echo "  STAGE2_INCLUDE_STAGE1_BRIDGE=1|0 (default: 1)"
  echo "  STAGE2_BRIDGE_STEP=50000"
  echo "  STAGE2_BRIDGE_CKPT_PATH=\$SCRATCH/ckpts/stage1/\$STAGE2_BRIDGE_STEP"
  echo "  LIGHTEVAL_CMD (default: bash scripts/run_lighteval_stage2_real.sh)"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${SCRATCH:?SCRATCH is not set. Run this on Leonardo where SCRATCH is defined.}"
: "${LIGHTEVAL_CMD:=bash scripts/run_lighteval_stage2_real.sh}"
export LIGHTEVAL_CMD

CONFIG_PATH="$REPO_ROOT/config/config_stage2.yaml"
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

CKPT_ROOT="$SCRATCH/ckpts/stage2"
RESULT_ROOT="$SCRATCH/evaluation_results/stage2"
EVAL_DATA_DIR="$RESULT_ROOT/eval_data"
EVAL_RESULTS_DIR="$RESULT_ROOT"
EVAL_LOG_DIR="$RESULT_ROOT/eval_logs"
SLURM_LOG_DIR="$EVAL_LOG_DIR/slurm"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-$TRAIN_TOKENIZER}"
SEQ_LEN="${SEQ_LEN:-$TRAIN_SEQ_LEN}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

STAGE2_STEPS="${STAGE2_STEPS:-2000 6000 10000 14000 18000 22000}"
read -r -a STEPS <<<"$STAGE2_STEPS"
STAGE2_INCLUDE_STAGE1_BRIDGE="${STAGE2_INCLUDE_STAGE1_BRIDGE:-1}"
STAGE2_BRIDGE_STEP="${STAGE2_BRIDGE_STEP:-50000}"
STAGE2_BRIDGE_CKPT_PATH="${STAGE2_BRIDGE_CKPT_PATH:-$SCRATCH/ckpts/stage1/$STAGE2_BRIDGE_STEP}"

if [[ "$TOKENIZER_NAME_OR_PATH" == /* ]] && [ ! -d "$TOKENIZER_NAME_OR_PATH" ]; then
  echo "Tokenizer path not found: $TOKENIZER_NAME_OR_PATH"
  exit 1
fi

mkdir -p "$EVAL_DATA_DIR" "$EVAL_RESULTS_DIR" "$EVAL_LOG_DIR" "$SLURM_LOG_DIR"

validate_checkpoint() {
  local checkpoint_path="$1"
  if [ ! -d "$checkpoint_path" ]; then
    echo "Missing checkpoint directory: $checkpoint_path"
    return 1
  fi
  if [ -z "$(ls -A "$checkpoint_path" 2>/dev/null)" ]; then
    echo "Checkpoint directory is empty: $checkpoint_path"
    return 1
  fi
  if [ ! -f "$checkpoint_path/config.yaml" ] || [ ! -f "$checkpoint_path/model_config.json" ] || [ ! -d "$checkpoint_path/model" ]; then
    echo "Checkpoint is not Nanotron-layout: $checkpoint_path"
    echo "Expected: config.yaml, model_config.json, and model/ directory"
    return 1
  fi
}

submit_step() {
  local step="$1"
  local checkpoint_path="$2"
  local tag="$3"

  if ! [[ "$step" =~ ^[0-9]+$ ]]; then
    echo "Invalid STEP (must be integer): $step [$tag]"
    return 1
  fi
  validate_checkpoint "$checkpoint_path" || return 1

  local exports
  exports="ALL,CHECKPOINT_PATH=${checkpoint_path},STEP=${step},TOKENIZER_NAME_OR_PATH=${TOKENIZER_NAME_OR_PATH},SEQ_LEN=${SEQ_LEN},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},EVAL_DATA_DIR=${EVAL_DATA_DIR},EVAL_RESULTS_DIR=${EVAL_RESULTS_DIR},EVAL_LOG_DIR=${EVAL_LOG_DIR},HF_DATASETS_TRUST_REMOTE_CODE=1"

  echo "Submitting Stage-2 eval [$tag]: STEP=$step CHECKPOINT_PATH=$checkpoint_path"
  sbatch \
    --output="$SLURM_LOG_DIR/eval_stage2_${step}_%j.out" \
    --error="$SLURM_LOG_DIR/eval_stage2_${step}_%j.err" \
    --export="$exports" \
    slurm/eval_stage2_manual.sbatch
}

cd "$REPO_ROOT"
for STEP in "${STEPS[@]}"; do
  CHECKPOINT_PATH="$CKPT_ROOT/$STEP"
  submit_step "$STEP" "$CHECKPOINT_PATH" "stage2"
done

if [ "$STAGE2_INCLUDE_STAGE1_BRIDGE" = "1" ]; then
  submit_step "$STAGE2_BRIDGE_STEP" "$STAGE2_BRIDGE_CKPT_PATH" "stage1-bridge"
elif [ "$STAGE2_INCLUDE_STAGE1_BRIDGE" != "0" ]; then
  echo "Invalid STAGE2_INCLUDE_STAGE1_BRIDGE value: $STAGE2_INCLUDE_STAGE1_BRIDGE (use 0 or 1)"
  exit 1
else
  echo "Skipping Stage-1 bridge submission (STAGE2_INCLUDE_STAGE1_BRIDGE=0)"
fi
