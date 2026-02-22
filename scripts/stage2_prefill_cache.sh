#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

: "${SCRATCH:?SCRATCH is not set. Run on Leonardo where SCRATCH is defined.}"

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

if [ -z "$TRAIN_TOKENIZER" ] || [ -z "$TRAIN_SEQ_LEN" ]; then
  echo "Could not parse tokenizer/sequence_length from $CONFIG_PATH"
  exit 1
fi

RESULT_ROOT="$SCRATCH/evaluation_results/stage2"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$RESULT_ROOT/eval_data}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-$RESULT_ROOT}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-$RESULT_ROOT/eval_logs}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-$TRAIN_TOKENIZER}"
SEQ_LEN="${SEQ_LEN:-$TRAIN_SEQ_LEN}"

mkdir -p "$EVAL_DATA_DIR" "$EVAL_RESULTS_DIR" "$EVAL_LOG_DIR"
export EVAL_DATA_DIR EVAL_RESULTS_DIR EVAL_LOG_DIR
export HF_DATASETS_TRUST_REMOTE_CODE=1

echo "=== Stage 2 Cache Prefill ==="
echo "TOKENIZER_NAME_OR_PATH=$TOKENIZER_NAME_OR_PATH"
echo "SEQ_LEN=$SEQ_LEN"
echo "EVAL_DATA_DIR=$EVAL_DATA_DIR"
echo "EVAL_RESULTS_DIR=$EVAL_RESULTS_DIR"
echo "EVAL_LOG_DIR=$EVAL_LOG_DIR"
echo "============================="

python - "$TOKENIZER_NAME_OR_PATH" "$SEQ_LEN" "$EVAL_DATA_DIR" <<'PY'
import importlib.util
import sys
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("stage2_eval_impl", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


tokenizer_name = sys.argv[1]
seq_len = int(sys.argv[2])
eval_data_dir = Path(sys.argv[3])

repo_root = Path.cwd()
eval_impl = load_module(repo_root / "evaluation" / "run_eval_stage2.py")

checkpoint_stub = Path("/tmp/stage2_cache_prefill")
tokenizer = eval_impl.load_tokenizer(tokenizer_name, checkpoint_stub)
ppl_cfg = eval_impl.load_yaml(repo_root / "evaluation" / "task_suites" / "stage2_ppl_slices.yaml")
cf_cfg = eval_impl.load_yaml(repo_root / "evaluation" / "task_suites" / "stage2_cf_core.yaml")

eval_impl.log("Prefill: preparing Tier2 PPL slices")
eval_impl.prepare_ppl_slices(ppl_cfg, tokenizer, seq_len, eval_data_dir)
eval_impl.log("Prefill: preparing Tier3 CF tasks")
eval_impl.prepare_cf_tasks(cf_cfg, eval_data_dir)
eval_impl.log("Prefill complete")
PY

echo "Cache check:"
echo "Tier2 packed slices: $(find "$EVAL_DATA_DIR/ppl_slices" -name packed_tokens.pt 2>/dev/null | wc -l)"
echo "Tier3 task files:    $(find "$EVAL_DATA_DIR/cf_tasks" -name examples.jsonl 2>/dev/null | wc -l)"
