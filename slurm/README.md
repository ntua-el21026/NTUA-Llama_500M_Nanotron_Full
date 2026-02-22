# Slurm Folder

This folder contains Slurm job templates used on Leonardo HPC.

## Evaluation jobs

- `eval_stage1_manual.sbatch`
- `eval_stage2_manual.sbatch`
- `eval_stage3_manual.sbatch`
- `eval_stage4_manual.sbatch`

## Training jobs

- `run_stage1.sbatch`, `run_stage2.sbatch`, `run_stage3.sbatch`
- SFT jobs (`sft_stage3_smoltalk*.sbatch`, smoke variants)

## Data and environment jobs

- `run_data_prep*.sbatch`
- environment/setup helpers (e.g., grouped-gemm installation)
- vibe-test and smoke-check job templates

Keep this folder for scheduler templates only; run logs should remain outside version control.
