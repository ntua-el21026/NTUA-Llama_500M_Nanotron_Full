# Evaluation Folder

This folder contains the full multi-stage evaluation pipeline.

## Core runners

- `run_eval_stage1.py`
- `run_eval_stage2.py`
- `run_eval_stage3.py`
- `run_eval_stage4.py`

Each runner prepares cached inputs, runs scoring through backend scripts, and writes per-checkpoint JSON outputs.

## Task suites

`task_suites/` defines stage-specific evaluation suites:

- Tier 2: perplexity slices
- Tier 3: conditional-likelihood QA tasks
- Tier 4 (Stage 4): SFT-native prompt checks

## Results and diagrams

- `_runs_stage1/` ... `_runs_stage4/`: fetched results, logs (if present), and stage diagrams
- `paper/`: scripts and figures used for report-ready plots

Primary output files per checkpoint are:

- `tier2_ppl.json`
- `tier3_cf.json`
- `tier4_sft_native.json` (Stage 4)
