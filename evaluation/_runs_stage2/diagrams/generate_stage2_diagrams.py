#!/usr/bin/env python3
"""Generate Stage-2 evaluation diagrams + Stage1-vs-Stage2 comparisons.

Reads Stage-2 results:
  evaluation/_runs_stage2/eval_results/by_checkpoint/step_<STEP>/
    - tier2_ppl.json
    - tier3_cf.json

Also reads Stage-1 results for cross-stage comparison charts:
  evaluation/_runs_stage1/eval_results/by_checkpoint/step_<STEP>/

Writes diagrams under:
  evaluation/_runs_stage2/diagrams/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


TIER2_SLICE_ORDER = [
    "general_web",
    "web_fineweb_edu_sample10bt",
    "wiki_en_20231101",
    "books_newslike",
    "code",
    "code_stackv2_edu_filtered",
    "code_starcoder2data_extras_lhq",
    "math_finemath3plus",
    "math_finemath4plus",
    "math_infiwebmath3plus",
    "math_infiwebmath4plus",
]

TIER2_ANCHOR_SLICES = [
    "general_web",
    "web_fineweb_edu_sample10bt",
    "wiki_en_20231101",
    "books_newslike",
    "code",
    "math_finemath3plus",
    "math_infiwebmath3plus",
]

TIER2_HQ_SLICES = [
    "code_stackv2_edu_filtered",
    "code_starcoder2data_extras_lhq",
    "math_finemath4plus",
    "math_infiwebmath4plus",
]

TIER2_DOMAIN_MAP = {
    "general_web": "web",
    "web_fineweb_edu_sample10bt": "web",
    "wiki_en_20231101": "web",
    "books_newslike": "web",
    "code": "code",
    "code_stackv2_edu_filtered": "code",
    "code_starcoder2data_extras_lhq": "code",
    "math_finemath3plus": "math",
    "math_finemath4plus": "math",
    "math_infiwebmath3plus": "math",
    "math_infiwebmath4plus": "math",
}

TIER2_DOMAIN_COLORS = {
    "web": "#1f77b4",
    "code": "#ff7f0e",
    "math": "#2ca02c",
}

TIER3_TASK_ORDER = [
    "hellaswag",
    "commonsense_qa",
    "social_i_qa",
    "copa",
    "arc_easy",
    "openbookqa",
    "qasc",
    "sciq",
    "piqa",
    "winogrande",
]

TIER3_GROUP_MAP = {
    "hellaswag": "commonsense",
    "commonsense_qa": "commonsense",
    "social_i_qa": "commonsense",
    "copa": "commonsense",
    "arc_easy": "science_qa",
    "openbookqa": "science_qa",
    "qasc": "science_qa",
    "sciq": "science_qa",
    "piqa": "reasoning_physical",
    "winogrande": "reasoning_physical",
}

TIER3_GROUP_COLORS = {
    "commonsense": "#9467bd",
    "science_qa": "#17becf",
    "reasoning_physical": "#d62728",
}

DEFAULT_SNAPSHOTS = [2000, 10000, 22000]
STAGE2_BRIDGE_STEP = 50000


def parse_int_csv(raw: str) -> List[int]:
    vals: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    return vals


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]

    parser = argparse.ArgumentParser(description="Generate Stage-2 evaluation diagrams")
    parser.add_argument(
        "--stage2-results-root",
        type=Path,
        default=repo_root / "evaluation" / "_runs_stage2" / "eval_results" / "by_checkpoint",
        help="Stage-2 by_checkpoint root",
    )
    parser.add_argument(
        "--stage1-results-root",
        type=Path,
        default=repo_root / "evaluation" / "_runs_stage1" / "eval_results" / "by_checkpoint",
        help="Stage-1 by_checkpoint root (for cross-stage charts)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "evaluation" / "_runs_stage2" / "diagrams",
        help="Output diagrams root",
    )
    parser.add_argument(
        "--snapshots",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SNAPSHOTS),
        help="Comma-separated snapshot steps for Stage-2 snapshot/domain charts",
    )
    parser.add_argument(
        "--compare-steps",
        type=str,
        default="2000,10000",
        help="Preferred overlap steps for Stage1-vs-Stage2 comparison charts",
    )
    parser.add_argument(
        "--skip-stage-compare",
        action="store_true",
        help="Generate only Stage-2 charts (skip Stage1-vs-Stage2 comparisons)",
    )
    return parser.parse_args()


def load_stage1_plot_impl(repo_root: Path) -> ModuleType:
    impl_path = repo_root / "evaluation" / "_runs_stage1" / "diagrams" / "generate_stage1_diagrams.py"
    if not impl_path.exists():
        raise FileNotFoundError(f"Missing Stage-1 diagram generator: {impl_path}")

    spec = importlib.util.spec_from_file_location("stage1_diagrams_impl", str(impl_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {impl_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_impl_for_stage2(impl: ModuleType) -> None:
    impl.TIER2_SLICE_ORDER = list(TIER2_SLICE_ORDER)
    impl.TIER2_DOMAIN_MAP = dict(TIER2_DOMAIN_MAP)
    impl.TIER2_DOMAIN_COLORS = dict(TIER2_DOMAIN_COLORS)
    impl.TIER3_TASK_ORDER = list(TIER3_TASK_ORDER)
    impl.TIER3_GROUP_MAP = dict(TIER3_GROUP_MAP)
    impl.TIER3_GROUP_COLORS = dict(TIER3_GROUP_COLORS)


def finite_or_nan(x: float | None) -> float:
    if x is None:
        return float("nan")
    if not math.isfinite(float(x)):
        return float("nan")
    return float(x)


def choose_stage2_baseline_step(all_steps: Sequence[int]) -> int:
    if STAGE2_BRIDGE_STEP in all_steps:
        return STAGE2_BRIDGE_STEP
    if 2000 in all_steps:
        return 2000
    return all_steps[0]


def compute_group_macro_ppl(data: Dict[int, dict], steps: Sequence[int], slice_names: Sequence[str]) -> List[float]:
    out: List[float] = []
    for step in steps:
        vals: List[float] = []
        for slc in slice_names:
            v = data[step]["tier2"]["slices"].get(slc, {}).get("ppl")
            v = finite_or_nan(v)
            if math.isfinite(v):
                vals.append(v)
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def plot_stage2_hq_focus(
    impl: ModuleType,
    steps: List[int],
    data: Dict[int, dict],
    output_root: Path,
    generated: List[Path],
) -> None:
    # 1) Only the new HQ slices as individual trends.
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap("tab10")
    for i, slc in enumerate(TIER2_HQ_SLICES):
        ys = [finite_or_nan(data[s]["tier2"]["slices"].get(slc, {}).get("ppl")) for s in steps]
        ax.plot(steps, ys, marker="o", linewidth=2.2, label=slc, color=cmap(i % 10))

    ax.set_title("Stage2 Tier2 HQ Slices PPL Across Steps", fontsize=12, weight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_xticks(steps)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    impl.save_figure(
        fig,
        output_root / "tier2" / "hq" / "tier2_hq_ppl_by_slice_vs_step.png",
        generated,
        right_legend=True,
    )

    # 2) Anchors vs HQ macro PPL trend.
    anchors = [x for x in TIER2_ANCHOR_SLICES if any(x in data[s]["tier2"]["slices"] for s in steps)]
    hq = [x for x in TIER2_HQ_SLICES if any(x in data[s]["tier2"]["slices"] for s in steps)]
    if not anchors or not hq:
        return

    anchor_macro = compute_group_macro_ppl(data, steps, anchors)
    hq_macro = compute_group_macro_ppl(data, steps, hq)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(steps, anchor_macro, marker="o", linewidth=2.4, color="#1f77b4", label="Anchors (Stage1 slices)")
    ax.plot(steps, hq_macro, marker="o", linewidth=2.4, color="#ff7f0e", label="HQ additions (Stage2 new slices)")
    ax.set_title("Stage2 Tier2 Macro PPL: Anchors vs HQ", fontsize=12, weight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Macro PPL (lower is better)")
    ax.set_xticks(steps)
    ax.legend()
    impl.save_figure(
        fig,
        output_root / "tier2" / "hq" / "tier2_anchor_vs_hq_macro_ppl_vs_step.png",
        generated,
    )

    # 3) Anchors vs HQ relative improvement vs baseline step.
    baseline_step = 2000 if 2000 in steps else steps[0]
    base_anchor = finite_or_nan(compute_group_macro_ppl(data, [baseline_step], anchors)[0])
    base_hq = finite_or_nan(compute_group_macro_ppl(data, [baseline_step], hq)[0])

    rel_anchor: List[float] = []
    rel_hq: List[float] = []
    for a, b in zip(anchor_macro, hq_macro):
        if not math.isfinite(base_anchor) or base_anchor == 0 or not math.isfinite(a):
            rel_anchor.append(float("nan"))
        else:
            rel_anchor.append((base_anchor - a) / base_anchor * 100.0)

        if not math.isfinite(base_hq) or base_hq == 0 or not math.isfinite(b):
            rel_hq.append(float("nan"))
        else:
            rel_hq.append((base_hq - b) / base_hq * 100.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(steps, rel_anchor, marker="o", linewidth=2.4, color="#1f77b4", label="Anchors")
    ax.plot(steps, rel_hq, marker="o", linewidth=2.4, color="#ff7f0e", label="HQ additions")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Stage2 Tier2 Relative PPL Improvement vs Baseline {baseline_step}", fontsize=12, weight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Improvement (%)")
    ax.set_xticks(steps)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()
    impl.save_figure(
        fig,
        output_root / "tier2" / "hq" / "tier2_anchor_vs_hq_relative_vs_step2000.png",
        generated,
    )


def plot_stage2_objective_quadrant(
    impl: ModuleType,
    steps: List[int],
    data: Dict[int, dict],
    baseline_step: int,
    output_root: Path,
    generated: List[Path],
) -> None:
    base_t2 = finite_or_nan(data[baseline_step]["tier2"]["macro"])
    base_t3 = finite_or_nan(data[baseline_step]["tier3"]["macro"])
    if not math.isfinite(base_t2) or not math.isfinite(base_t3) or base_t2 == 0 or base_t3 == 0:
        return

    xs: List[float] = []
    ys: List[float] = []
    for step in steps:
        curr_t2 = finite_or_nan(data[step]["tier2"]["macro"])
        curr_t3 = finite_or_nan(data[step]["tier3"]["macro"])
        if not math.isfinite(curr_t2) or not math.isfinite(curr_t3):
            xs.append(float("nan"))
            ys.append(float("nan"))
            continue
        xs.append((base_t2 - curr_t2) / base_t2 * 100.0)  # positive is better
        ys.append((curr_t3 - base_t3) / base_t3 * 100.0)  # positive is better

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    sc = ax.scatter(xs, ys, c=range(len(steps)), cmap="viridis", s=95)

    for step, x, y in zip(steps, xs, ys):
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        ax.annotate(str(step), (x, y), textcoords="offset points", xytext=(6, 5), fontsize=9)

    ax.set_title(
        f"Stage2 Objective vs Baseline {baseline_step}: Tier2 Win Without Tier3 Loss",
        fontsize=12,
        weight="bold",
    )
    ax.set_xlabel("Tier2 macro PPL improvement (%) [positive=better]")
    ax.set_ylabel("Tier3 macro ACC change (%) [positive=better]")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Step index (early -> late)")

    impl.save_figure(
        fig,
        output_root / "cross_tier" / "objective" / "stage2_objective_quadrant_vs_baseline.png",
        generated,
    )


def plot_stage2_bridge_focus(
    impl: ModuleType,
    stage2_steps: List[int],
    data: Dict[int, dict],
    baseline_step: int,
    output_root: Path,
    generated: List[Path],
) -> None:
    if baseline_step not in data:
        return
    if not stage2_steps:
        return

    base_t2 = finite_or_nan(data[baseline_step]["tier2"]["macro"])
    base_t3 = finite_or_nan(data[baseline_step]["tier3"]["macro"])
    if not math.isfinite(base_t2) or not math.isfinite(base_t3):
        return

    # 1) Stage-2 trend with bridge reference lines.
    t2_vals = [finite_or_nan(data[s]["tier2"]["macro"]) for s in stage2_steps]
    t3_vals = [finite_or_nan(data[s]["tier3"]["macro"]) for s in stage2_steps]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(stage2_steps, t2_vals, marker="o", linewidth=2.3, color="#1f77b4")
    axes[0].axhline(base_t2, color="#9467bd", linestyle="--", linewidth=1.5, label=f"Bridge {baseline_step}")
    axes[0].set_title("Stage2 Tier2 Macro PPL vs Stage1 Bridge", fontsize=12, weight="bold")
    axes[0].set_xlabel("Stage2 training step")
    axes[0].set_ylabel("Macro PPL (lower is better)")
    axes[0].legend()

    axes[1].plot(stage2_steps, t3_vals, marker="o", linewidth=2.3, color="#d62728")
    axes[1].axhline(base_t3, color="#9467bd", linestyle="--", linewidth=1.5, label=f"Bridge {baseline_step}")
    axes[1].set_title("Stage2 Tier3 Macro ACC vs Stage1 Bridge", fontsize=12, weight="bold")
    axes[1].set_xlabel("Stage2 training step")
    axes[1].set_ylabel("Macro accuracy (higher is better)")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].legend()

    impl.save_figure(
        fig,
        output_root / "stage_compare" / "bridge" / "stage2_vs_stage1_bridge_macro_trends.png",
        generated,
    )

    # 2) Macro deltas versus bridge by Stage-2 step.
    t2_delta = [finite_or_nan(v) - base_t2 if math.isfinite(finite_or_nan(v)) else float("nan") for v in t2_vals]
    t3_delta = [finite_or_nan(v) - base_t3 if math.isfinite(finite_or_nan(v)) else float("nan") for v in t3_vals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors_t2 = ["#2ca02c" if math.isfinite(v) and v <= 0 else "#d62728" for v in t2_delta]
    colors_t3 = ["#2ca02c" if math.isfinite(v) and v >= 0 else "#d62728" for v in t3_delta]

    bars = axes[0].bar([str(s) for s in stage2_steps], t2_delta, color=colors_t2, edgecolor="black", linewidth=0.6)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    impl.annotate_bars(axes[0], bars, "{:+.2f}")
    axes[0].set_title("Tier2 Delta vs Stage1 Bridge", fontsize=12, weight="bold")
    axes[0].set_ylabel("Delta PPL (negative is better)")
    axes[0].set_xlabel("Stage2 step")

    bars = axes[1].bar([str(s) for s in stage2_steps], t3_delta, color=colors_t3, edgecolor="black", linewidth=0.6)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    impl.annotate_bars(axes[1], bars, "{:+.3f}")
    axes[1].set_title("Tier3 Delta vs Stage1 Bridge", fontsize=12, weight="bold")
    axes[1].set_ylabel("Delta accuracy (positive is better)")
    axes[1].set_xlabel("Stage2 step")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    impl.save_figure(
        fig,
        output_root / "stage_compare" / "bridge" / "stage2_minus_stage1_bridge_macro_delta_by_step.png",
        generated,
    )


def plot_stage_compare(
    impl: ModuleType,
    stage1_steps: List[int],
    stage1_data: Dict[int, dict],
    stage2_steps: List[int],
    stage2_data: Dict[int, dict],
    preferred_compare_steps: List[int],
    output_root: Path,
    generated: List[Path],
) -> None:
    stage2_steps = [s for s in stage2_steps if s != STAGE2_BRIDGE_STEP]
    if not stage2_steps:
        return

    shared_slices = sorted(
        {k for s in stage1_steps for k in stage1_data[s]["tier2"]["slices"].keys()}
        & {k for s in stage2_steps for k in stage2_data[s]["tier2"]["slices"].keys()},
        key=lambda k: (TIER2_SLICE_ORDER.index(k) if k in TIER2_SLICE_ORDER else 999, k),
    )
    shared_tasks = sorted(
        {k for s in stage1_steps for k in stage1_data[s]["tier3"]["tasks"].keys()}
        & {k for s in stage2_steps for k in stage2_data[s]["tier3"]["tasks"].keys()},
        key=lambda k: (TIER3_TASK_ORDER.index(k) if k in TIER3_TASK_ORDER else 999, k),
    )
    overlap_steps = sorted(set(stage1_steps) & set(stage2_steps))
    if not shared_slices or not shared_tasks:
        return

    # 1) Stage1 vs Stage2 Tier2 shared-slice macro PPL trend.
    s1_t2_shared = compute_group_macro_ppl(stage1_data, stage1_steps, shared_slices)
    s2_t2_shared = compute_group_macro_ppl(stage2_data, stage2_steps, shared_slices)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(stage1_steps, s1_t2_shared, marker="o", linewidth=2.4, color="#1f77b4", label="Stage1")
    ax.plot(stage2_steps, s2_t2_shared, marker="o", linewidth=2.4, color="#ff7f0e", label="Stage2")
    ax.set_title("Stage1 vs Stage2 Tier2 Shared-Slice Macro PPL", fontsize=12, weight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Macro PPL on shared Tier2 slices (lower is better)")
    ax.legend()
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "tier2" / "stage1_vs_stage2_shared_macro_ppl_vs_step.png",
        generated,
    )

    # 2) Stage1 vs Stage2 Tier3 macro ACC trend.
    s1_t3_macro = [finite_or_nan(stage1_data[s]["tier3"]["macro"]) for s in stage1_steps]
    s2_t3_macro = [finite_or_nan(stage2_data[s]["tier3"]["macro"]) for s in stage2_steps]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(stage1_steps, s1_t3_macro, marker="o", linewidth=2.4, color="#1f77b4", label="Stage1")
    ax.plot(stage2_steps, s2_t3_macro, marker="o", linewidth=2.4, color="#ff7f0e", label="Stage2")
    ax.set_title("Stage1 vs Stage2 Tier3 Macro Accuracy", fontsize=12, weight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Macro accuracy (higher is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "tier3" / "stage1_vs_stage2_macro_acc_vs_step.png",
        generated,
    )

    if overlap_steps:
        # 3) Overlap-step macro deltas (stage2 - stage1).
        t2_delta = []
        t3_delta = []
        for step in overlap_steps:
            s1_t2 = compute_group_macro_ppl(stage1_data, [step], shared_slices)[0]
            s2_t2 = compute_group_macro_ppl(stage2_data, [step], shared_slices)[0]
            t2_delta.append(s2_t2 - s1_t2)  # negative is better

            s1_t3 = finite_or_nan(stage1_data[step]["tier3"]["macro"])
            s2_t3 = finite_or_nan(stage2_data[step]["tier3"]["macro"])
            t3_delta.append(s2_t3 - s1_t3)  # positive is better

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        colors_t2 = ["#2ca02c" if math.isfinite(v) and v < 0 else "#d62728" for v in t2_delta]
        colors_t3 = ["#2ca02c" if math.isfinite(v) and v >= 0 else "#d62728" for v in t3_delta]

        bars = axes[0].bar([str(s) for s in overlap_steps], t2_delta, color=colors_t2, edgecolor="black", linewidth=0.6)
        axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
        impl.annotate_bars(axes[0], bars, "{:+.2f}")
        axes[0].set_title("Tier2 macro delta (Stage2 - Stage1)")
        axes[0].set_ylabel("Delta PPL (negative is better)")
        axes[0].set_xlabel("Overlap step")

        bars = axes[1].bar([str(s) for s in overlap_steps], t3_delta, color=colors_t3, edgecolor="black", linewidth=0.6)
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
        impl.annotate_bars(axes[1], bars, "{:+.3f}")
        axes[1].set_title("Tier3 macro delta (Stage2 - Stage1)")
        axes[1].set_ylabel("Delta accuracy (positive is better)")
        axes[1].set_xlabel("Overlap step")
        axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        impl.save_figure(
            fig,
            output_root / "stage_compare" / "summary" / "stage2_minus_stage1_macro_delta_overlap_steps.png",
            generated,
        )

        # 4) Per-slice overlap deltas (stage2 - stage1) for Tier2.
        series_t2: Dict[int, List[float]] = {}
        for step in overlap_steps:
            vals: List[float] = []
            for slc in shared_slices:
                s1 = finite_or_nan(stage1_data[step]["tier2"]["slices"].get(slc, {}).get("ppl"))
                s2 = finite_or_nan(stage2_data[step]["tier2"]["slices"].get(slc, {}).get("ppl"))
                vals.append(s2 - s1)
            series_t2[step] = vals

        fig, ax = plt.subplots(figsize=(15, 7))
        impl.grouped_bar(
            ax=ax,
            categories=shared_slices,
            series=series_t2,
            y_label="Delta PPL (Stage2 - Stage1, negative is better)",
            title="Tier2 Shared-Slice Delta on Overlap Steps",
            percent_axis=False,
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.tick_params(axis="x", labelrotation=30)
        impl.save_figure(
            fig,
            output_root / "stage_compare" / "tier2" / "stage2_minus_stage1_shared_slice_ppl_delta_overlap.png",
            generated,
        )

        # 5) Per-task overlap deltas (stage2 - stage1) for Tier3.
        series_t3: Dict[int, List[float]] = {}
        for step in overlap_steps:
            vals = []
            for task in shared_tasks:
                s1 = finite_or_nan(stage1_data[step]["tier3"]["tasks"].get(task, {}).get("acc"))
                s2 = finite_or_nan(stage2_data[step]["tier3"]["tasks"].get(task, {}).get("acc"))
                vals.append(s2 - s1)
            series_t3[step] = vals

        fig, ax = plt.subplots(figsize=(15, 7))
        impl.grouped_bar(
            ax=ax,
            categories=shared_tasks,
            series=series_t3,
            y_label="Delta accuracy (Stage2 - Stage1, positive is better)",
            title="Tier3 Task Delta on Overlap Steps",
            percent_axis=True,
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.tick_params(axis="x", labelrotation=30)
        impl.save_figure(
            fig,
            output_root / "stage_compare" / "tier3" / "stage2_minus_stage1_task_acc_delta_overlap.png",
            generated,
        )

        # 6) Side-by-side absolute values at a selected overlap step.
        preferred_overlap = [s for s in preferred_compare_steps if s in overlap_steps]
        compare_step = preferred_overlap[-1] if preferred_overlap else max(overlap_steps)

        # Tier2 faceoff.
        x = list(range(len(shared_slices)))
        width = 0.38
        y1 = [finite_or_nan(stage1_data[compare_step]["tier2"]["slices"].get(k, {}).get("ppl")) for k in shared_slices]
        y2 = [finite_or_nan(stage2_data[compare_step]["tier2"]["slices"].get(k, {}).get("ppl")) for k in shared_slices]

        fig, ax = plt.subplots(figsize=(15, 7))
        bars1 = ax.bar([i - width / 2 for i in x], y1, width=width, label="Stage1", color="#1f77b4", edgecolor="black")
        bars2 = ax.bar([i + width / 2 for i in x], y2, width=width, label="Stage2", color="#ff7f0e", edgecolor="black")
        impl.annotate_bars(ax, bars1, "{:.2f}")
        impl.annotate_bars(ax, bars2, "{:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(shared_slices, rotation=30)
        ax.set_title(f"Tier2 Shared Slices: Stage1 vs Stage2 at Step {compare_step}", fontsize=12, weight="bold")
        ax.set_ylabel("PPL (lower is better)")
        ax.legend()
        impl.save_figure(
            fig,
            output_root / "stage_compare" / "tier2" / f"stage1_vs_stage2_shared_slice_ppl_step_{compare_step}.png",
            generated,
        )

        # Tier3 faceoff.
        x = list(range(len(shared_tasks)))
        y1 = [finite_or_nan(stage1_data[compare_step]["tier3"]["tasks"].get(k, {}).get("acc")) for k in shared_tasks]
        y2 = [finite_or_nan(stage2_data[compare_step]["tier3"]["tasks"].get(k, {}).get("acc")) for k in shared_tasks]

        fig, ax = plt.subplots(figsize=(15, 7))
        bars1 = ax.bar([i - width / 2 for i in x], y1, width=width, label="Stage1", color="#1f77b4", edgecolor="black")
        bars2 = ax.bar([i + width / 2 for i in x], y2, width=width, label="Stage2", color="#ff7f0e", edgecolor="black")
        impl.annotate_bars(ax, bars1, "{:.3f}")
        impl.annotate_bars(ax, bars2, "{:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(shared_tasks, rotation=30)
        ax.set_title(f"Tier3 Tasks: Stage1 vs Stage2 at Step {compare_step}", fontsize=12, weight="bold")
        ax.set_ylabel("Accuracy (higher is better)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend()
        impl.save_figure(
            fig,
            output_root / "stage_compare" / "tier3" / f"stage1_vs_stage2_task_acc_step_{compare_step}.png",
            generated,
        )


def main() -> None:
    args = parse_args()
    snapshots = parse_int_csv(args.snapshots)
    compare_steps = parse_int_csv(args.compare_steps)

    repo_root = Path(__file__).resolve().parents[3]
    impl = load_stage1_plot_impl(repo_root)
    configure_impl_for_stage2(impl)
    impl.setup_style()

    stage2_steps_all, stage2_data = impl.load_results(args.stage2_results_root)
    baseline_step = choose_stage2_baseline_step(stage2_steps_all)
    stage2_steps = [s for s in stage2_steps_all if s != baseline_step]
    if not stage2_steps:
        stage2_steps = list(stage2_steps_all)

    snapshots = [s for s in snapshots if s in stage2_steps]
    if not snapshots and stage2_steps:
        snapshots = [stage2_steps[0], stage2_steps[-1]]

    generated: List[Path] = []

    # Stage-2 charts with same filesystem/logic as Stage-1.
    impl.plot_tier2_series(stage2_steps, stage2_data, args.output_root, generated)
    impl.plot_tier3_series(stage2_steps, stage2_data, args.output_root, generated)
    impl.plot_macro_trends(stage2_steps, stage2_data, args.output_root, generated)
    impl.plot_relative_vs_baseline(stage2_steps, stage2_data, args.output_root, generated)
    impl.plot_tier3_support(stage2_steps, stage2_data, args.output_root, generated)
    impl.plot_snapshots(stage2_steps, stage2_data, snapshots, args.output_root, generated)
    impl.plot_domain_views(stage2_steps, stage2_data, snapshots, args.output_root, generated)
    impl.plot_early_late_delta(stage2_steps, stage2_data, args.output_root, generated)
    impl.plot_tradeoff(stage2_steps, stage2_data, args.output_root, generated)

    # Additional Stage-2 objective charts.
    plot_stage2_hq_focus(impl, stage2_steps, stage2_data, args.output_root, generated)
    plot_stage2_objective_quadrant(impl, stage2_steps, stage2_data, baseline_step, args.output_root, generated)
    plot_stage2_bridge_focus(impl, stage2_steps, stage2_data, baseline_step, args.output_root, generated)

    # Stage1-vs-Stage2 comparison charts.
    stage_compare_enabled = False
    stage1_steps: List[int] = []
    stage1_data: Dict[int, dict] = {}
    stage_compare_error = ""
    if not args.skip_stage_compare:
        try:
            stage1_steps, stage1_data = impl.load_results(args.stage1_results_root)
            plot_stage_compare(
                impl,
                stage1_steps,
                stage1_data,
                stage2_steps,
                stage2_data,
                compare_steps,
                args.output_root,
                generated,
            )
            stage_compare_enabled = True
        except Exception as exc:  # pragma: no cover - explicit resilience for missing Stage-1 data
            stage_compare_error = str(exc)

    manifest = {
        "stage2_results_root": str(args.stage2_results_root),
        "stage1_results_root": str(args.stage1_results_root),
        "output_root": str(args.output_root),
        "stage2_steps_all": stage2_steps_all,
        "stage2_steps_plotted": stage2_steps,
        "baseline_step": baseline_step,
        "stage1_steps": stage1_steps,
        "snapshots_requested": snapshots,
        "compare_steps_requested": compare_steps,
        "stage_compare_enabled": stage_compare_enabled,
        "stage_compare_error": stage_compare_error,
        "files_generated": [str(p) for p in generated],
    }

    manifest_path = args.output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    generated.append(manifest_path)

    print(f"Generated {len(generated)} files under {args.output_root}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
