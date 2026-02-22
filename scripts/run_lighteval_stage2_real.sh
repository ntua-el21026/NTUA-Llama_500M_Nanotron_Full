#!/usr/bin/env bash
set -euo pipefail

# Required by evaluation/run_eval_stage2.py / run_lighteval contract
: "${LIGHTEVAL_CHECKPOINT_PATH:?Missing LIGHTEVAL_CHECKPOINT_PATH}"
: "${LIGHTEVAL_RESULTS_DIR:?Missing LIGHTEVAL_RESULTS_DIR}"
: "${LIGHTEVAL_STEP:?Missing LIGHTEVAL_STEP}"
: "${LIGHTEVAL_TOKENIZER:?Missing LIGHTEVAL_TOKENIZER}"
: "${LIGHTEVAL_SEQ_LEN:?Missing LIGHTEVAL_SEQ_LEN}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-}"
if [ -z "$EVAL_DATA_DIR" ]; then
  echo "Missing EVAL_DATA_DIR (exported by submit launcher)."
  exit 1
fi

CHECKPOINT_PATH="$LIGHTEVAL_CHECKPOINT_PATH"
RESULTS_DIR="$LIGHTEVAL_RESULTS_DIR"
STEP="$LIGHTEVAL_STEP"
TOKENIZER_PATH="$LIGHTEVAL_TOKENIZER"
SEQ_LEN="$LIGHTEVAL_SEQ_LEN"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

NANOTRON_ROOT="${NANOTRON_ROOT:-/leonardo_scratch/large/userexternal/${USER}/nanotron_smollm3}"
SCORER_PY="$REPO_ROOT/scripts/run_lighteval_stage2_backend.py"

mkdir -p "$RESULTS_DIR"

# Ensure grouped_gemm sees a modern libstdc++ (GLIBCXX >= 3.4.29).
if command -v g++ >/dev/null 2>&1; then
  GCC_HOME="$(dirname "$(dirname "$(command -v g++)")")"
  GCC_LIB="$GCC_HOME/lib64/libstdc++.so.6"
  if [ -f "$GCC_LIB" ]; then
    export LD_PRELOAD="$GCC_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
    export LD_LIBRARY_PATH="$GCC_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

echo "[stage2-real] checkpoint=$CHECKPOINT_PATH"
echo "[stage2-real] step=$STEP"
echo "[stage2-real] eval_data_dir=$EVAL_DATA_DIR"
echo "[stage2-real] results_dir=$RESULTS_DIR"
echo "[stage2-real] tokenizer=$TOKENIZER_PATH"
echo "[stage2-real] seq_len=$SEQ_LEN"
echo "[stage2-real] batch_size=$EVAL_BATCH_SIZE"
echo "[stage2-real] nanotron_root=$NANOTRON_ROOT"
echo "[stage2-real] scorer=$SCORER_PY"

if [ ! -f "$SCORER_PY" ]; then
  echo "Missing scorer script: $SCORER_PY"
  exit 1
fi
if [ ! -d "$NANOTRON_ROOT" ]; then
  echo "Missing NANOTRON_ROOT directory: $NANOTRON_ROOT"
  exit 1
fi

export STAGE2_REPO_ROOT="$REPO_ROOT"
export STAGE2_EVAL_DATA_DIR="$EVAL_DATA_DIR"
export STAGE2_RESULTS_DIR="$RESULTS_DIR"
export NANOTRON_ROOT
export EVAL_BATCH_SIZE

python -u "$SCORER_PY"
