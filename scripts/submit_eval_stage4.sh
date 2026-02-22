#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "Usage: $0"
  echo "Submits Stage-4 (SFT) checkpoints for sft_stage3_smoltalk_run1 by default."
  echo "Also submits Stage-3 bridge checkpoint by default into Stage-4 suite."
  echo "Optional env overrides:"
  echo "  STAGE4_CONFIG_PATH=$HOME/Smol_Project/config/sft_stage3_smoltalk.yaml"
  echo "  STAGE4_STEPS=\"500 1500 3500 5000\""
  echo "  CKPT_ROOT=\$SCRATCH/sft_project/ckpts/sft_stage3_smoltalk_run1"
  echo "  STAGE4_INCLUDE_STAGE3_BRIDGE=1|0 (default: 1)"
  echo "  STAGE4_BRIDGE_STEP=12000"
  echo "  STAGE4_BRIDGE_CKPT_PATH=\$SCRATCH/ckpts/stage3/\$STAGE4_BRIDGE_STEP"
  echo "  STAGE4_RESULT_ROOT=\$SCRATCH/evaluation_results/stage4"
  echo "  LIGHTEVAL_CMD (default: bash scripts/run_lighteval_stage4_real.sh)"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${SCRATCH:?SCRATCH is not set. Run this on Leonardo where SCRATCH is defined.}"
: "${LIGHTEVAL_CMD:=bash scripts/run_lighteval_stage4_real.sh}"
export LIGHTEVAL_CMD

CONFIG_PATH="${STAGE4_CONFIG_PATH:-$REPO_ROOT/config/sft_stage3_smoltalk.yaml}"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "Missing Stage-4/SFT config: $CONFIG_PATH"
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

TRAIN_CHECKPOINTS_PATH=$(
  awk -F': *' '
    /^[[:space:]]*checkpoints_path:[[:space:]]*/ {
      gsub(/["'"'"']/, "", $2)
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
if [ -z "$TRAIN_CHECKPOINTS_PATH" ]; then
  echo "Could not read checkpoints_path from $CONFIG_PATH"
  exit 1
fi

DEFAULT_CKPT_ROOT="$SCRATCH/sft_project/ckpts/sft_stage3_smoltalk_run1"
CKPT_ROOT="${CKPT_ROOT:-$DEFAULT_CKPT_ROOT}"
if [ ! -d "$CKPT_ROOT" ] && [ -d "$TRAIN_CHECKPOINTS_PATH" ]; then
  CKPT_ROOT="$TRAIN_CHECKPOINTS_PATH"
fi
RESULT_ROOT="${STAGE4_RESULT_ROOT:-$SCRATCH/evaluation_results/stage4}"
EVAL_DATA_DIR="$RESULT_ROOT/eval_data"
EVAL_RESULTS_DIR="$RESULT_ROOT"
EVAL_LOG_DIR="$RESULT_ROOT/eval_logs"
SLURM_LOG_DIR="$EVAL_LOG_DIR/slurm"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-$TRAIN_TOKENIZER}"
SEQ_LEN="${SEQ_LEN:-$TRAIN_SEQ_LEN}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

if [ ! -d "$CKPT_ROOT" ]; then
  echo "Checkpoint root directory not found: $CKPT_ROOT"
  exit 1
fi

STAGE4_STEPS="${STAGE4_STEPS:-500 1500 3500 5000}"
read -r -a STEPS <<<"$STAGE4_STEPS"

if [ "${#STEPS[@]}" -eq 0 ]; then
  echo "No Stage-4 steps resolved. Set STAGE4_STEPS explicitly."
  exit 1
fi

DEFAULT_STAGE3_BRIDGE_STEP="12000"
if [ -f "$SCRATCH/ckpts/stage3/latest.txt" ]; then
  maybe_step="$(tr -d ' \n' < "$SCRATCH/ckpts/stage3/latest.txt")"
  if [[ "$maybe_step" =~ ^[0-9]+$ ]]; then
    DEFAULT_STAGE3_BRIDGE_STEP="$maybe_step"
  fi
fi

STAGE4_INCLUDE_STAGE3_BRIDGE="${STAGE4_INCLUDE_STAGE3_BRIDGE:-1}"
STAGE4_BRIDGE_STEP="${STAGE4_BRIDGE_STEP:-$DEFAULT_STAGE3_BRIDGE_STEP}"
STAGE4_BRIDGE_CKPT_PATH="${STAGE4_BRIDGE_CKPT_PATH:-$SCRATCH/ckpts/stage3/$STAGE4_BRIDGE_STEP}"

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

  echo "Submitting Stage-4 eval [$tag]: STEP=$step CHECKPOINT_PATH=$checkpoint_path"
  sbatch \
    --output="$SLURM_LOG_DIR/eval_stage4_${step}_%j.out" \
    --error="$SLURM_LOG_DIR/eval_stage4_${step}_%j.err" \
    --export="$exports" \
    slurm/eval_stage4_manual.sbatch
}

cd "$REPO_ROOT"
for STEP in "${STEPS[@]}"; do
  CHECKPOINT_PATH="$CKPT_ROOT/$STEP"
  submit_step "$STEP" "$CHECKPOINT_PATH" "stage4"
done

if [ "$STAGE4_INCLUDE_STAGE3_BRIDGE" = "1" ]; then
  submit_step "$STAGE4_BRIDGE_STEP" "$STAGE4_BRIDGE_CKPT_PATH" "stage3-bridge"
elif [ "$STAGE4_INCLUDE_STAGE3_BRIDGE" != "0" ]; then
  echo "Invalid STAGE4_INCLUDE_STAGE3_BRIDGE value: $STAGE4_INCLUDE_STAGE3_BRIDGE (use 0 or 1)"
  exit 1
else
  echo "Skipping Stage-3 bridge submission (STAGE4_INCLUDE_STAGE3_BRIDGE=0)"
fi
