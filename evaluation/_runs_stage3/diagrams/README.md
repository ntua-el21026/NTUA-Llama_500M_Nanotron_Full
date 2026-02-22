# Stage 3 Diagrams README

This folder contains generated visualizations for Stage-3 evaluation results, plus cross-stage comparisons against Stage-1 and Stage-2.

## Source and generation

- Stage-3 input JSONs:
  - `evaluation/_runs_stage3/eval_results/by_checkpoint/step_<STEP>/tier2_ppl.json`
  - `evaluation/_runs_stage3/eval_results/by_checkpoint/step_<STEP>/tier3_cf.json`
- Stage-2 input JSONs (for comparisons):
  - `evaluation/_runs_stage2/eval_results/by_checkpoint/step_<STEP>/tier2_ppl.json`
  - `evaluation/_runs_stage2/eval_results/by_checkpoint/step_<STEP>/tier3_cf.json`
- Stage-1 input JSONs (for comparisons):
  - `evaluation/_runs_stage1/eval_results/by_checkpoint/step_<STEP>/tier2_ppl.json`
  - `evaluation/_runs_stage1/eval_results/by_checkpoint/step_<STEP>/tier3_cf.json`
- Generator script:
  - `evaluation/_runs_stage3/diagrams/generate_stage3_diagrams.py`
- Generation manifest:
  - `evaluation/_runs_stage3/diagrams/manifest.json`

Run from repo root:

```bash
python evaluation/_runs_stage3/diagrams/generate_stage3_diagrams.py
```

Optional:

```bash
python evaluation/_runs_stage3/diagrams/generate_stage3_diagrams.py \
  --snapshots 2000,8000,12000 \
  --compare-steps 2000,10000
```

## Metric direction

- Tier2 PPL: lower is better.
- Tier3 accuracy: higher is better.
- Stage-3 bridge baseline is `step_22000` when present (Stage-2 last checkpoint evaluated on Stage-3 suite).
- Relative vs step 2000:
  - Tier2 positive % means PPL improved (decreased).
  - Tier3 positive % means ACC improved (increased).
- Stage deltas:
  - Tier2 delta < 0 is better for the newer stage.
  - Tier3 delta > 0 is better for the newer stage.
- Objective charts in Stage-3 use:
  - Tier2 loss axis (`<= 0` means no regression on Tier2)
  - Tier3 gain axis (`>= 0` means improvement on Tier3).

## Folder contents

## `tier2/series`
- `tier2_ppl_by_slice_vs_step.png`

## `tier2/macro`
- `tier2_macro_ppl_vs_step.png`

## `tier2/relative`
- `tier2_relative_vs_step2000.png`
- `tier2_macro_relative_vs_step2000.png`

## `tier2/deltas`
- `tier2_delta_step_12000_minus_2000_by_slice.png`

## `tier2/snapshots`
- `tier2_step_2000_by_slice.png`
- `tier2_step_8000_by_slice.png`
- `tier2_step_12000_by_slice.png`

## `tier2/domains`
- `tier2_domain_early_mid_late.png`

## `tier2/reasoning`
- `tier2_reasoning_ppl_by_slice_vs_step.png`
- `tier2_anchor_hq_reasoning_macro_ppl_vs_step.png`
- `tier2_anchor_hq_reasoning_relative_vs_baseline.png`

These focus on the Stage-3 reasoning additions and compare them against Stage-1 anchors and Stage-2 HQ slices.

## `tier3/series`
- `tier3_acc_by_task_vs_step.png`

## `tier3/macro`
- `tier3_macro_acc_vs_step.png`

## `tier3/relative`
- `tier3_relative_vs_step2000.png`
- `tier3_macro_relative_vs_step2000.png`

## `tier3/deltas`
- `tier3_delta_step_12000_minus_2000_by_task.png`

## `tier3/snapshots`
- `tier3_step_2000_by_task.png`
- `tier3_step_8000_by_task.png`
- `tier3_step_12000_by_task.png`

## `tier3/domains`
- `tier3_domain_early_mid_late.png`

## `tier3/support`
- `tier3_task_support_n.png`

## `cross_tier/tradeoff`
- `tier2_vs_tier3_tradeoff.png`

## `cross_tier/objective`
- `stage3_objective_quadrant_vs_baseline.png`
- `stage3_reasoning_objective_quadrant_vs_baseline.png`

These are the key objective charts:
- improve Tier3
- while keeping Tier2 without regression.
- baseline is bridge `step_22000` when available (otherwise step 2000).

## `stage_compare/bridge`
- `stage3_vs_stage2_bridge_macro_trends.png`
  - Stage-3 macro trajectories with horizontal Stage-2 bridge reference lines.
- `stage3_minus_stage2_bridge_macro_delta_by_step.png`
  - Per-step macro deltas versus Stage-2 bridge:
    - Tier2 delta < 0 is better
    - Tier3 delta > 0 is better

## `stage_compare/three_stage/tier2`
- `stage1_vs_stage2_vs_stage3_shared_macro_ppl_vs_step.png`
- `stage123_shared_slice_ppl_step_10000.png`
- `stage3_minus_stage2_shared_slice_ppl_delta_overlap.png`
- `stage3_minus_stage1_shared_slice_ppl_delta_overlap.png`
  - Stage-2 bridge `step_50000` and Stage-3 bridge `step_22000` are excluded from overlap trends (bridge views are under `stage_compare/bridge`).

## `stage_compare/three_stage/tier3`
- `stage1_vs_stage2_vs_stage3_macro_acc_vs_step.png`
- `stage123_task_acc_step_10000.png`
- `stage3_minus_stage2_task_acc_delta_overlap.png`
- `stage3_minus_stage1_task_acc_delta_overlap.png`

## `stage_compare/three_stage/summary`
- `stage123_macro_overlap_steps.png`
- `stage3_minus_stage2_and_stage1_macro_delta_overlap.png`
- `stage3_objective_quadrants_vs_stage1_stage2_overlap.png`
- `stage3_objective_pass_fail_overlap.png`

These summarize whether Stage 3 achieves the objective relative to prior stages at overlap checkpoints:
- Tier3 gain (`ACC delta >= 0`)
- Tier2 no-loss (`PPL delta <= 0`).

## Notes

- Default snapshots for Stage-3 are `2000, 8000, 12000`.
- 3-stage overlap steps are automatically inferred from available results (currently typically `2000` and `10000`).
- If Stage-1 or Stage-2 results are missing, 3-stage comparison charts are skipped and the reason is recorded in `manifest.json`.
