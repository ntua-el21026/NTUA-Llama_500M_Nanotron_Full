# Scripts Folder

This folder contains executable project scripts grouped by function.

## Data preparation

- `prepare_dataset*.py`
- `prep_smol_smoltalk_for_nanotron.py`
- `build_smoltalk_jsonl.py`

## Evaluation orchestration

- `stage{1,2,3,4}_prefill_cache.sh`
- `stage{1,2,3,4}_eval_offline.sh`
- `submit_eval_stage{1,2,3,4}.sh`

## Evaluation backends

- `run_lighteval_stage{1,2,3,4}_real.sh`
- `run_lighteval_stage{1,2,3,4}_backend.py`

## Training and utilities

- `run_train.py`
- `export_nanotron_ckpt_to_hf.py`
- SFT checks and vibe-test scripts (`vibe_test_*`, `sft_check.py`)

These scripts are designed to run from repository root.
