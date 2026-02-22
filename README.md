# NTUA-LLaMA-500M Nanotron (Full Repository)

Official Team 12 repository for the NTUA ECE course "Pattern Recognition" (Winter 2025-2026).

## Scope

This repository contains the full project workflow for:

- Stage 1 pretraining
- Stage 2 high-quality code/math continuation
- Stage 3 reasoning continuation
- Stage 4 supervised fine-tuning (SFT)
- Tiered evaluation, analysis, and paper assets

## Repository Layout

- `config/`: canonical training and SFT YAML configurations
- `scripts/`: data preparation, prefill, offline evaluation, submission, and backends
- `slurm/`: Slurm job templates used on Leonardo
- `evaluation/`: stage evaluators, task suites, fetched results, and diagrams
- `docs/`: paper source, report artifact, and maintenance utilities
- `final_model/`: selected model-only exported weights used as final artifacts

## Notes

- This repo is the full project artifact set (code + configs + selected outputs).
- Run-specific scratch logs and temporary caches are intentionally excluded.
- License: MIT (`LICENSE`).
