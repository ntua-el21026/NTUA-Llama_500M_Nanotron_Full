#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

: "${SCRATCH:?SCRATCH is not set. Run on Leonardo where SCRATCH is defined.}"

# Force offline mode on compute nodes (expects cache already prefetched).
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Stage 2 Offline Eval Submit ==="
echo "HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "==================================="

exec scripts/submit_eval_stage2.sh
