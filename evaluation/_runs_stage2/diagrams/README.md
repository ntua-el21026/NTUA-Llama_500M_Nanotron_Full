# Stage 2 Diagrams README

This folder contains generated visualizations for Stage-2 evaluation results, plus Stage1-vs-Stage2 comparison charts.

## Source and generation

- Stage-2 input JSONs:
  - `evaluation/_runs_stage2/eval_results/by_checkpoint/step_<STEP>/tier2_ppl.json`
  - `evaluation/_runs_stage2/eval_results/by_checkpoint/step_<STEP>/tier3_cf.json`
- Stage-1 input JSONs (for comparisons):
  - `evaluation/_runs_stage1/eval_results/by_checkpoint/step_<STEP>/tier2_ppl.json`
  - `evaluation/_runs_stage1/eval_results/by_checkpoint/step_<STEP>/tier3_cf.json`
- Generator script:
  - `evaluation/_runs_stage2/diagrams/generate_stage2_diagrams.py`
- Generation manifest:
  - `evaluation/_runs_stage2/diagrams/manifest.json`

Run from repo root:

```bash
python evaluation/_runs_stage2/diagrams/generate_stage2_diagrams.py
```

Optional arguments:

```bash
python evaluation/_runs_stage2/diagrams/generate_stage2_diagrams.py \
  --snapshots 2000,10000,22000 \
  --compare-steps 2000,10000
```

## Metric direction

- Tier2 PPL: lower is better.
- Tier3 accuracy: higher is better.
- Stage-2 bridge baseline is `step_50000` when present (Stage-1 last checkpoint evaluated on Stage-2 suite).
- Relative charts vs step 2000:
  - Tier2 positive percentage means PPL improved (decreased).
  - Tier3 positive percentage means accuracy improved (increased).
- Stage2-Stage1 delta charts:
  - Tier2 delta < 0 means Stage2 is better.
  - Tier3 delta > 0 means Stage2 is better.

## Folder contents and meaning

### `tier2/series`

- `tier2_ppl_by_slice_vs_step.png`
  - One line per Tier2 slice across Stage-2 checkpoints.

### `tier2/macro`

- `tier2_macro_ppl_vs_step.png`
  - Stage-2 macro Tier2 PPL trajectory.

### `tier2/relative`

- `tier2_relative_vs_step2000.png`
  - Relative per-slice PPL improvement vs step 2000.
- `tier2_macro_relative_vs_step2000.png`
  - Relative macro PPL improvement vs step 2000.

### `tier2/deltas`

- `tier2_delta_step_22000_minus_2000_by_slice.png`
  - Absolute Tier2 per-slice delta (late minus early).

### `tier2/snapshots`

- `tier2_step_2000_by_slice.png`
- `tier2_step_10000_by_slice.png`
- `tier2_step_22000_by_slice.png`
  - Early/mid/late per-slice bars.

### `tier2/domains`

- `tier2_domain_early_mid_late.png`
  - Domain aggregation (web/code/math) over early/mid/late checkpoints.

### `tier2/hq`

- `tier2_hq_ppl_by_slice_vs_step.png`
  - Trends only for Stage-2 added HQ probes.
- `tier2_anchor_vs_hq_macro_ppl_vs_step.png`
  - Anchor slices vs HQ slices macro PPL trend.
- `tier2_anchor_vs_hq_relative_vs_step2000.png`
  - Relative improvement comparison between anchor and HQ groups.

### `tier3/series`

- `tier3_acc_by_task_vs_step.png`
  - One line per Tier3 task across Stage-2 checkpoints.

### `tier3/macro`

- `tier3_macro_acc_vs_step.png`
  - Stage-2 macro Tier3 accuracy trajectory.

### `tier3/relative`

- `tier3_relative_vs_step2000.png`
  - Relative per-task accuracy improvement vs step 2000.
- `tier3_macro_relative_vs_step2000.png`
  - Relative macro accuracy improvement vs step 2000.

### `tier3/deltas`

- `tier3_delta_step_22000_minus_2000_by_task.png`
  - Absolute Tier3 per-task delta (late minus early).

### `tier3/snapshots`

- `tier3_step_2000_by_task.png`
- `tier3_step_10000_by_task.png`
- `tier3_step_22000_by_task.png`
  - Early/mid/late per-task bars.

### `tier3/domains`

- `tier3_domain_early_mid_late.png`
  - Domain-grouped Tier3 accuracy (commonsense/science_qa/reasoning_physical).

### `tier3/support`

- `tier3_task_support_n.png`
  - Number of evaluated examples (`n`) per task.

### `cross_tier/tradeoff`

- `tier2_vs_tier3_tradeoff.png`
  - Stage-2 checkpoint tradeoff: Tier2 macro PPL vs Tier3 macro accuracy.

### `cross_tier/objective`

- `stage2_objective_quadrant_vs_baseline.png`
  - Objective quadrant chart:
    - baseline is bridge `step_50000` when available, otherwise step 2000
    - x-axis: Tier2 improvement vs baseline (positive is better)
    - y-axis: Tier3 change vs baseline (positive is better)
  - Top-right quadrant means "better Tier2 and better/no loss Tier3".

### `stage_compare/bridge`

- `stage2_vs_stage1_bridge_macro_trends.png`
  - Stage-2 macro trajectories with horizontal Stage-1 bridge reference lines.
- `stage2_minus_stage1_bridge_macro_delta_by_step.png`
  - Per-step macro deltas versus Stage-1 bridge:
    - Tier2 delta < 0 is better
    - Tier3 delta > 0 is better

### `stage_compare/tier2`

- `stage1_vs_stage2_shared_macro_ppl_vs_step.png`
  - Stage1 vs Stage2 macro PPL on shared Tier2 slices only.
- `stage2_minus_stage1_shared_slice_ppl_delta_overlap.png`
  - Per-slice deltas on overlap steps (Stage2 - Stage1).
- `stage1_vs_stage2_shared_slice_ppl_step_10000.png`
  - Side-by-side absolute PPL at overlap checkpoint 10000.
  - Bridge `step_50000` is excluded from these trend/overlap plots (it is shown under `stage_compare/bridge`).

### `stage_compare/tier3`

- `stage1_vs_stage2_macro_acc_vs_step.png`
  - Stage1 vs Stage2 Tier3 macro accuracy trend.
- `stage2_minus_stage1_task_acc_delta_overlap.png`
  - Per-task deltas on overlap steps (Stage2 - Stage1).
- `stage1_vs_stage2_task_acc_step_10000.png`
  - Side-by-side absolute Tier3 task accuracy at overlap checkpoint 10000.

### `stage_compare/summary`

- `stage2_minus_stage1_macro_delta_overlap_steps.png`
  - Two-panel summary of Stage2-Stage1 macro deltas at overlap steps:
    - Tier2 macro PPL delta (negative means Stage2 better)
    - Tier3 macro accuracy delta (positive means Stage2 better)

## Notes

- Default snapshot steps are `2000,10000,22000`.
- If new Stage-2 results arrive, rerun the generator to refresh all plots.
- If Stage-1 results are missing, cross-stage comparison charts are skipped and noted in `manifest.json`.
