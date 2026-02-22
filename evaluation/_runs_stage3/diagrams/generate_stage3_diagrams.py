#!/usr/bin/env python3
"""Generate Stage-3 evaluation diagrams + Stage1/Stage2/Stage3 comparisons.

Reads Stage-3 results:
  evaluation/_runs_stage3/eval_results/by_checkpoint/step_<STEP>/
    - tier2_ppl.json
    - tier3_cf.json

Also reads Stage-1 and Stage-2 results for cross-stage comparison charts.

Writes diagrams under:
  evaluation/_runs_stage3/diagrams/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Rectangle


TIER2_SLICE_ORDER = [
    "general_web",
    "web_fineweb_edu_sample10bt",
    "wiki_en_20231101",
    "books_newslike",
    "code",
    "code_stackv2_edu_filtered",
    "code_starcoder2data_extras_lhq",
    "code_opencodereasoning",
    "math_finemath3plus",
    "math_finemath4plus",
    "math_infiwebmath3plus",
    "math_infiwebmath4plus",
    "math_openmathreasoning",
    "math_openmathinstruct1",
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

TIER2_REASONING_SLICES = [
    "math_openmathreasoning",
    "math_openmathinstruct1",
    "code_opencodereasoning",
]

TIER2_DOMAIN_MAP = {
    "general_web": "web",
    "web_fineweb_edu_sample10bt": "web",
    "wiki_en_20231101": "web",
    "books_newslike": "web",
    "code": "code",
    "code_stackv2_edu_filtered": "code",
    "code_starcoder2data_extras_lhq": "code",
    "code_opencodereasoning": "code",
    "math_finemath3plus": "math",
    "math_finemath4plus": "math",
    "math_infiwebmath3plus": "math",
    "math_infiwebmath4plus": "math",
    "math_openmathreasoning": "math",
    "math_openmathinstruct1": "math",
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
    "arc_challenge",
    "openbookqa",
    "qasc",
    "sciq",
    "mmlu",
    "piqa",
    "winogrande",
]

TIER3_GROUP_MAP = {
    "hellaswag": "commonsense",
    "commonsense_qa": "commonsense",
    "social_i_qa": "commonsense",
    "copa": "commonsense",
    "arc_easy": "science_qa",
    "arc_challenge": "science_qa",
    "openbookqa": "science_qa",
    "qasc": "science_qa",
    "sciq": "science_qa",
    "mmlu": "knowledge_reasoning",
    "piqa": "reasoning_physical",
    "winogrande": "reasoning_physical",
}

TIER3_GROUP_COLORS = {
    "commonsense": "#9467bd",
    "science_qa": "#17becf",
    "reasoning_physical": "#d62728",
    "knowledge_reasoning": "#8c564b",
}

DEFAULT_SNAPSHOTS = [2000, 8000, 12000]
STAGE3_BRIDGE_STEP = 22000
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

    parser = argparse.ArgumentParser(description="Generate Stage-3 evaluation diagrams")
    parser.add_argument(
        "--stage3-results-root",
        type=Path,
        default=repo_root / "evaluation" / "_runs_stage3" / "eval_results" / "by_checkpoint",
        help="Stage-3 by_checkpoint root",
    )
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
        help="Stage-1 by_checkpoint root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "evaluation" / "_runs_stage3" / "diagrams",
        help="Output diagrams root",
    )
    parser.add_argument(
        "--snapshots",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SNAPSHOTS),
        help="Comma-separated snapshot steps for Stage-3 snapshot/domain charts",
    )
    parser.add_argument(
        "--compare-steps",
        type=str,
        default="2000,10000",
        help="Preferred overlap steps for cross-stage comparison charts",
    )
    parser.add_argument(
        "--skip-stage-compare",
        action="store_true",
        help="Generate only Stage-3 charts (skip Stage1/Stage2/Stage3 comparisons)",
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


def configure_impl_for_stage3(impl: ModuleType) -> None:
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


def choose_stage3_baseline_step(all_steps: Sequence[int]) -> int:
    if STAGE3_BRIDGE_STEP in all_steps:
        return STAGE3_BRIDGE_STEP
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


def grouped_bar_by_stage(
    impl: ModuleType,
    ax: plt.Axes,
    categories: List[str],
    stage_series: Dict[str, List[float]],
    y_label: str,
    title: str,
    percent_axis: bool = False,
) -> None:
    stage_names = list(stage_series.keys())
    total_width = 0.8
    width = total_width / max(1, len(stage_names))
    x = list(range(len(categories)))

    palette = {
        "Stage1": "#1f77b4",
        "Stage2": "#ff7f0e",
        "Stage3": "#2ca02c",
        "Stage3-Stage2": "#2ca02c",
        "Stage3-Stage1": "#9467bd",
    }

    for i, name in enumerate(stage_names):
        ys = stage_series[name]
        xs = [xi - total_width / 2 + (i + 0.5) * width for xi in x]
        bars = ax.bar(
            xs,
            ys,
            width=width,
            label=name,
            color=palette.get(name, None),
            edgecolor="black",
            linewidth=0.5,
        )
        impl.annotate_bars(ax, bars, "{:.2f}" if not percent_axis else "{:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.legend(title="Stage")
    if percent_axis:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


def plot_stage3_focus(
    impl: ModuleType,
    steps: List[int],
    data: Dict[int, dict],
    baseline_step: int,
    output_root: Path,
    generated: List[Path],
) -> None:
    # 1) Reasoning additions only.
    available_reasoning = [x for x in TIER2_REASONING_SLICES if any(x in data[s]["tier2"]["slices"] for s in steps)]
    if available_reasoning:
        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = plt.get_cmap("tab10")
        for i, slc in enumerate(available_reasoning):
            ys = [finite_or_nan(data[s]["tier2"]["slices"].get(slc, {}).get("ppl")) for s in steps]
            ax.plot(steps, ys, marker="o", linewidth=2.2, label=slc, color=cmap(i % 10))

        ax.set_title("Stage3 Tier2 Reasoning Slices PPL Across Steps", fontsize=12, weight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Perplexity (lower is better)")
        ax.set_xticks(steps)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
        impl.save_figure(
            fig,
            output_root / "tier2" / "reasoning" / "tier2_reasoning_ppl_by_slice_vs_step.png",
            generated,
            right_legend=True,
        )

    # 2) Anchor vs HQ vs Reasoning macro trend.
    anchors = [x for x in TIER2_ANCHOR_SLICES if any(x in data[s]["tier2"]["slices"] for s in steps)]
    hq = [x for x in TIER2_HQ_SLICES if any(x in data[s]["tier2"]["slices"] for s in steps)]
    reasoning = [x for x in TIER2_REASONING_SLICES if any(x in data[s]["tier2"]["slices"] for s in steps)]
    if anchors and hq and reasoning:
        anchor_macro = compute_group_macro_ppl(data, steps, anchors)
        hq_macro = compute_group_macro_ppl(data, steps, hq)
        reasoning_macro = compute_group_macro_ppl(data, steps, reasoning)

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(steps, anchor_macro, marker="o", linewidth=2.4, color="#1f77b4", label="Anchors (Stage1)")
        ax.plot(steps, hq_macro, marker="o", linewidth=2.4, color="#ff7f0e", label="HQ additions (Stage2)")
        ax.plot(steps, reasoning_macro, marker="o", linewidth=2.4, color="#2ca02c", label="Reasoning additions (Stage3)")
        ax.set_title("Stage3 Tier2 Macro PPL: Anchors vs HQ vs Reasoning", fontsize=12, weight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Macro PPL (lower is better)")
        ax.set_xticks(steps)
        ax.legend()
        impl.save_figure(
            fig,
            output_root / "tier2" / "reasoning" / "tier2_anchor_hq_reasoning_macro_ppl_vs_step.png",
            generated,
        )

        base_anchor = finite_or_nan(compute_group_macro_ppl(data, [baseline_step], anchors)[0])
        base_hq = finite_or_nan(compute_group_macro_ppl(data, [baseline_step], hq)[0])
        base_reasoning = finite_or_nan(compute_group_macro_ppl(data, [baseline_step], reasoning)[0])

        rel_anchor: List[float] = []
        rel_hq: List[float] = []
        rel_reasoning: List[float] = []
        for a, b, c in zip(anchor_macro, hq_macro, reasoning_macro):
            rel_anchor.append((base_anchor - a) / base_anchor * 100.0 if math.isfinite(base_anchor) and base_anchor != 0 and math.isfinite(a) else float("nan"))
            rel_hq.append((base_hq - b) / base_hq * 100.0 if math.isfinite(base_hq) and base_hq != 0 and math.isfinite(b) else float("nan"))
            rel_reasoning.append((base_reasoning - c) / base_reasoning * 100.0 if math.isfinite(base_reasoning) and base_reasoning != 0 and math.isfinite(c) else float("nan"))

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(steps, rel_anchor, marker="o", linewidth=2.4, color="#1f77b4", label="Anchors")
        ax.plot(steps, rel_hq, marker="o", linewidth=2.4, color="#ff7f0e", label="HQ additions")
        ax.plot(steps, rel_reasoning, marker="o", linewidth=2.4, color="#2ca02c", label="Reasoning additions")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Stage3 Tier2 Relative PPL Improvement vs Baseline {baseline_step}", fontsize=12, weight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Improvement (%)")
        ax.set_xticks(steps)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend()
        impl.save_figure(
            fig,
            output_root / "tier2" / "reasoning" / "tier2_anchor_hq_reasoning_relative_vs_baseline.png",
            generated,
        )


def plot_stage3_objective(
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

    # A) Overall objective: Tier3 improves while Tier2 does not regress.
    # x-axis is Tier2 loss (positive => worse, non-positive => no loss)
    # y-axis is Tier3 gain (positive => better)
    tier2_loss: List[float] = []
    tier3_gain: List[float] = []
    for step in steps:
        curr_t2 = finite_or_nan(data[step]["tier2"]["macro"])
        curr_t3 = finite_or_nan(data[step]["tier3"]["macro"])
        if not math.isfinite(curr_t2) or not math.isfinite(curr_t3):
            tier2_loss.append(float("nan"))
            tier3_gain.append(float("nan"))
            continue
        tier2_loss.append((curr_t2 - base_t2) / base_t2 * 100.0)
        tier3_gain.append((curr_t3 - base_t3) / base_t3 * 100.0)

    fig, ax = plt.subplots(figsize=(9, 7))
    finite_pairs = [(s, x, y) for s, x, y in zip(steps, tier2_loss, tier3_gain) if math.isfinite(x) and math.isfinite(y)]
    xs = [x for _, x, _ in finite_pairs]
    ys = [y for _, _, y in finite_pairs]

    if xs and ys:
        x_min = min(min(xs), 0.0)
        x_max = max(max(xs), 0.0)
        y_min = min(min(ys), 0.0)
        y_max = max(max(ys), 0.0)
        x_pad = max(0.5, (x_max - x_min) * 0.15)
        y_pad = max(0.5, (y_max - y_min) * 0.15)
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if x_min < 0 and y_max > 0:
            ax.add_patch(
                Rectangle(
                    (x_min, 0.0),
                    0.0 - x_min,
                    y_max - 0.0,
                    facecolor="#d8f3dc",
                    edgecolor="none",
                    alpha=0.25,
                    zorder=0,
                )
            )
            ax.text(
                x_min + (0.0 - x_min) * 0.03,
                y_max - (y_max - 0.0) * 0.08,
                "Objective region\n(Tier3 gain, Tier2 no loss)",
                ha="left",
                va="top",
                fontsize=9,
                color="#2d6a4f",
            )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    sc = ax.scatter(tier2_loss, tier3_gain, c=range(len(steps)), cmap="viridis", s=95)
    for step, x, y in zip(steps, tier2_loss, tier3_gain):
        if math.isfinite(x) and math.isfinite(y):
            ax.annotate(str(step), (x, y), textcoords="offset points", xytext=(6, 5), fontsize=9)
    ax.set_title(
        f"Stage3 Objective vs Baseline {baseline_step}: Tier3 Gain Without Tier2 Loss",
        fontsize=12,
        weight="bold",
    )
    ax.set_xlabel("Tier2 macro PPL change (%) [<= 0 means no loss]")
    ax.set_ylabel("Tier3 macro ACC change (%) [>= 0 means gain]")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Step index (early -> late)")
    impl.save_figure(
        fig,
        output_root / "cross_tier" / "objective" / "stage3_objective_quadrant_vs_baseline.png",
        generated,
    )

    # B) Reasoning-specific objective overlay (color = reasoning improvement).
    reasoning = [x for x in TIER2_REASONING_SLICES if any(x in data[s]["tier2"]["slices"] for s in steps)]
    if not reasoning:
        return

    base_reasoning = finite_or_nan(compute_group_macro_ppl(data, [baseline_step], reasoning)[0])
    if not math.isfinite(base_reasoning) or base_reasoning == 0:
        return

    reasoning_gain: List[float] = []
    for step in steps:
        curr_reasoning = finite_or_nan(compute_group_macro_ppl(data, [step], reasoning)[0])
        if not math.isfinite(curr_reasoning):
            reasoning_gain.append(float("nan"))
        else:
            reasoning_gain.append((base_reasoning - curr_reasoning) / base_reasoning * 100.0)

    fig, ax = plt.subplots(figsize=(9, 7))
    finite_pairs = [(x, y) for x, y in zip(tier2_loss, tier3_gain) if math.isfinite(x) and math.isfinite(y)]
    if finite_pairs:
        xs = [x for x, _ in finite_pairs]
        ys = [y for _, y in finite_pairs]
        x_min = min(min(xs), 0.0)
        x_max = max(max(xs), 0.0)
        y_min = min(min(ys), 0.0)
        y_max = max(max(ys), 0.0)
        x_pad = max(0.5, (x_max - x_min) * 0.15)
        y_pad = max(0.5, (y_max - y_min) * 0.15)
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if x_min < 0 and y_max > 0:
            ax.add_patch(
                Rectangle(
                    (x_min, 0.0),
                    0.0 - x_min,
                    y_max - 0.0,
                    facecolor="#d8f3dc",
                    edgecolor="none",
                    alpha=0.25,
                    zorder=0,
                )
            )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    sc = ax.scatter(
        tier2_loss,
        tier3_gain,
        c=reasoning_gain,
        cmap="RdYlGn",
        s=105,
        edgecolor="black",
        linewidth=0.6,
    )
    for step, x, y in zip(steps, tier2_loss, tier3_gain):
        if math.isfinite(x) and math.isfinite(y):
            ax.annotate(str(step), (x, y), textcoords="offset points", xytext=(6, 5), fontsize=9)
    ax.set_title(
        f"Stage3 Objective vs Baseline {baseline_step}: Tier3 Gain / Tier2 No-Loss (colored by reasoning gain)",
        fontsize=12,
        weight="bold",
    )
    ax.set_xlabel("Tier2 macro PPL change (%) [<= 0 means no loss]")
    ax.set_ylabel("Tier3 macro ACC change (%) [>= 0 means gain]")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Reasoning-slice macro PPL improvement (%)")
    impl.save_figure(
        fig,
        output_root / "cross_tier" / "objective" / "stage3_reasoning_objective_quadrant_vs_baseline.png",
        generated,
    )


def plot_stage3_bridge_focus(
    impl: ModuleType,
    stage3_steps: List[int],
    data: Dict[int, dict],
    baseline_step: int,
    output_root: Path,
    generated: List[Path],
) -> None:
    if baseline_step not in data:
        return
    if not stage3_steps:
        return

    base_t2 = finite_or_nan(data[baseline_step]["tier2"]["macro"])
    base_t3 = finite_or_nan(data[baseline_step]["tier3"]["macro"])
    if not math.isfinite(base_t2) or not math.isfinite(base_t3):
        return

    t2_vals = [finite_or_nan(data[s]["tier2"]["macro"]) for s in stage3_steps]
    t3_vals = [finite_or_nan(data[s]["tier3"]["macro"]) for s in stage3_steps]

    # 1) Stage-3 trend with Stage-2 bridge reference.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(stage3_steps, t2_vals, marker="o", linewidth=2.3, color="#1f77b4")
    axes[0].axhline(base_t2, color="#9467bd", linestyle="--", linewidth=1.5, label=f"Bridge {baseline_step}")
    axes[0].set_title("Stage3 Tier2 Macro PPL vs Stage2 Bridge", fontsize=12, weight="bold")
    axes[0].set_xlabel("Stage3 training step")
    axes[0].set_ylabel("Macro PPL (lower is better)")
    axes[0].legend()

    axes[1].plot(stage3_steps, t3_vals, marker="o", linewidth=2.3, color="#d62728")
    axes[1].axhline(base_t3, color="#9467bd", linestyle="--", linewidth=1.5, label=f"Bridge {baseline_step}")
    axes[1].set_title("Stage3 Tier3 Macro ACC vs Stage2 Bridge", fontsize=12, weight="bold")
    axes[1].set_xlabel("Stage3 training step")
    axes[1].set_ylabel("Macro accuracy (higher is better)")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].legend()

    impl.save_figure(
        fig,
        output_root / "stage_compare" / "bridge" / "stage3_vs_stage2_bridge_macro_trends.png",
        generated,
    )

    # 2) Macro deltas versus bridge by Stage-3 step.
    t2_delta = [finite_or_nan(v) - base_t2 if math.isfinite(finite_or_nan(v)) else float("nan") for v in t2_vals]
    t3_delta = [finite_or_nan(v) - base_t3 if math.isfinite(finite_or_nan(v)) else float("nan") for v in t3_vals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors_t2 = ["#2ca02c" if math.isfinite(v) and v <= 0 else "#d62728" for v in t2_delta]
    colors_t3 = ["#2ca02c" if math.isfinite(v) and v >= 0 else "#d62728" for v in t3_delta]

    bars = axes[0].bar([str(s) for s in stage3_steps], t2_delta, color=colors_t2, edgecolor="black", linewidth=0.6)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    impl.annotate_bars(axes[0], bars, "{:+.2f}")
    axes[0].set_title("Tier2 Delta vs Stage2 Bridge", fontsize=12, weight="bold")
    axes[0].set_ylabel("Delta PPL (negative is better)")
    axes[0].set_xlabel("Stage3 step")

    bars = axes[1].bar([str(s) for s in stage3_steps], t3_delta, color=colors_t3, edgecolor="black", linewidth=0.6)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    impl.annotate_bars(axes[1], bars, "{:+.3f}")
    axes[1].set_title("Tier3 Delta vs Stage2 Bridge", fontsize=12, weight="bold")
    axes[1].set_ylabel("Delta accuracy (positive is better)")
    axes[1].set_xlabel("Stage3 step")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    impl.save_figure(
        fig,
        output_root / "stage_compare" / "bridge" / "stage3_minus_stage2_bridge_macro_delta_by_step.png",
        generated,
    )


def plot_three_stage_compare(
    impl: ModuleType,
    stage1_steps: List[int],
    stage1_data: Dict[int, dict],
    stage2_steps: List[int],
    stage2_data: Dict[int, dict],
    stage3_steps: List[int],
    stage3_data: Dict[int, dict],
    preferred_compare_steps: List[int],
    output_root: Path,
    generated: List[Path],
) -> None:
    stage2_steps = [s for s in stage2_steps if s != STAGE2_BRIDGE_STEP]
    stage3_steps = [s for s in stage3_steps if s != STAGE3_BRIDGE_STEP]
    if not stage2_steps or not stage3_steps:
        return

    shared_slices = sorted(
        {k for s in stage1_steps for k in stage1_data[s]["tier2"]["slices"].keys()}
        & {k for s in stage2_steps for k in stage2_data[s]["tier2"]["slices"].keys()}
        & {k for s in stage3_steps for k in stage3_data[s]["tier2"]["slices"].keys()},
        key=lambda k: (TIER2_SLICE_ORDER.index(k) if k in TIER2_SLICE_ORDER else 999, k),
    )
    shared_tasks = sorted(
        {k for s in stage1_steps for k in stage1_data[s]["tier3"]["tasks"].keys()}
        & {k for s in stage2_steps for k in stage2_data[s]["tier3"]["tasks"].keys()}
        & {k for s in stage3_steps for k in stage3_data[s]["tier3"]["tasks"].keys()},
        key=lambda k: (TIER3_TASK_ORDER.index(k) if k in TIER3_TASK_ORDER else 999, k),
    )
    overlap_steps = sorted(set(stage1_steps) & set(stage2_steps) & set(stage3_steps))

    if not shared_slices or not shared_tasks:
        return

    # 1) Three-stage Tier2 macro on shared slices.
    s1_t2_shared = compute_group_macro_ppl(stage1_data, stage1_steps, shared_slices)
    s2_t2_shared = compute_group_macro_ppl(stage2_data, stage2_steps, shared_slices)
    s3_t2_shared = compute_group_macro_ppl(stage3_data, stage3_steps, shared_slices)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(stage1_steps, s1_t2_shared, marker="o", linewidth=2.3, color="#1f77b4", label="Stage1")
    ax.plot(stage2_steps, s2_t2_shared, marker="o", linewidth=2.3, color="#ff7f0e", label="Stage2")
    ax.plot(stage3_steps, s3_t2_shared, marker="o", linewidth=2.3, color="#2ca02c", label="Stage3")
    ax.set_title("Stage1 vs Stage2 vs Stage3: Tier2 Shared-Slice Macro PPL", fontsize=12, weight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Macro PPL on shared slices (lower is better)")
    ax.legend()
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "three_stage" / "tier2" / "stage1_vs_stage2_vs_stage3_shared_macro_ppl_vs_step.png",
        generated,
    )

    # 2) Three-stage Tier3 macro ACC.
    s1_t3_macro = [finite_or_nan(stage1_data[s]["tier3"]["macro"]) for s in stage1_steps]
    s2_t3_macro = [finite_or_nan(stage2_data[s]["tier3"]["macro"]) for s in stage2_steps]
    s3_t3_macro = [finite_or_nan(stage3_data[s]["tier3"]["macro"]) for s in stage3_steps]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(stage1_steps, s1_t3_macro, marker="o", linewidth=2.3, color="#1f77b4", label="Stage1")
    ax.plot(stage2_steps, s2_t3_macro, marker="o", linewidth=2.3, color="#ff7f0e", label="Stage2")
    ax.plot(stage3_steps, s3_t3_macro, marker="o", linewidth=2.3, color="#2ca02c", label="Stage3")
    ax.set_title("Stage1 vs Stage2 vs Stage3: Tier3 Macro Accuracy", fontsize=12, weight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Macro accuracy (higher is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "three_stage" / "tier3" / "stage1_vs_stage2_vs_stage3_macro_acc_vs_step.png",
        generated,
    )

    if not overlap_steps:
        return

    # 3) Overlap-steps summary bars for macros.
    s1_t2_ov = [compute_group_macro_ppl(stage1_data, [s], shared_slices)[0] for s in overlap_steps]
    s2_t2_ov = [compute_group_macro_ppl(stage2_data, [s], shared_slices)[0] for s in overlap_steps]
    s3_t2_ov = [compute_group_macro_ppl(stage3_data, [s], shared_slices)[0] for s in overlap_steps]

    s1_t3_ov = [finite_or_nan(stage1_data[s]["tier3"]["macro"]) for s in overlap_steps]
    s2_t3_ov = [finite_or_nan(stage2_data[s]["tier3"]["macro"]) for s in overlap_steps]
    s3_t3_ov = [finite_or_nan(stage3_data[s]["tier3"]["macro"]) for s in overlap_steps]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    grouped_bar_by_stage(
        impl,
        axes[0],
        [str(s) for s in overlap_steps],
        {"Stage1": s1_t2_ov, "Stage2": s2_t2_ov, "Stage3": s3_t2_ov},
        "Macro PPL on shared slices (lower is better)",
        "Tier2 Macro at Overlap Steps",
        percent_axis=False,
    )
    grouped_bar_by_stage(
        impl,
        axes[1],
        [str(s) for s in overlap_steps],
        {"Stage1": s1_t3_ov, "Stage2": s2_t3_ov, "Stage3": s3_t3_ov},
        "Macro accuracy (higher is better)",
        "Tier3 Macro at Overlap Steps",
        percent_axis=True,
    )
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "three_stage" / "summary" / "stage123_macro_overlap_steps.png",
        generated,
    )

    # 4) Side-by-side absolute values at selected overlap step.
    preferred_overlap = [s for s in preferred_compare_steps if s in overlap_steps]
    compare_step = preferred_overlap[-1] if preferred_overlap else max(overlap_steps)

    # Tier2 per-slice faceoff at compare step.
    x = list(range(len(shared_slices)))
    width = 0.25
    y1 = [finite_or_nan(stage1_data[compare_step]["tier2"]["slices"].get(k, {}).get("ppl")) for k in shared_slices]
    y2 = [finite_or_nan(stage2_data[compare_step]["tier2"]["slices"].get(k, {}).get("ppl")) for k in shared_slices]
    y3 = [finite_or_nan(stage3_data[compare_step]["tier2"]["slices"].get(k, {}).get("ppl")) for k in shared_slices]

    fig, ax = plt.subplots(figsize=(16, 7))
    bars1 = ax.bar([i - width for i in x], y1, width=width, label="Stage1", color="#1f77b4", edgecolor="black")
    bars2 = ax.bar([i for i in x], y2, width=width, label="Stage2", color="#ff7f0e", edgecolor="black")
    bars3 = ax.bar([i + width for i in x], y3, width=width, label="Stage3", color="#2ca02c", edgecolor="black")
    impl.annotate_bars(ax, bars1, "{:.2f}")
    impl.annotate_bars(ax, bars2, "{:.2f}")
    impl.annotate_bars(ax, bars3, "{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(shared_slices, rotation=30)
    ax.set_title(f"Tier2 Shared Slices: Stage1 vs Stage2 vs Stage3 at Step {compare_step}", fontsize=12, weight="bold")
    ax.set_ylabel("PPL (lower is better)")
    ax.legend()
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "three_stage" / "tier2" / f"stage123_shared_slice_ppl_step_{compare_step}.png",
        generated,
    )

    # Tier3 per-task faceoff at compare step.
    x = list(range(len(shared_tasks)))
    y1 = [finite_or_nan(stage1_data[compare_step]["tier3"]["tasks"].get(k, {}).get("acc")) for k in shared_tasks]
    y2 = [finite_or_nan(stage2_data[compare_step]["tier3"]["tasks"].get(k, {}).get("acc")) for k in shared_tasks]
    y3 = [finite_or_nan(stage3_data[compare_step]["tier3"]["tasks"].get(k, {}).get("acc")) for k in shared_tasks]

    fig, ax = plt.subplots(figsize=(16, 7))
    bars1 = ax.bar([i - width for i in x], y1, width=width, label="Stage1", color="#1f77b4", edgecolor="black")
    bars2 = ax.bar([i for i in x], y2, width=width, label="Stage2", color="#ff7f0e", edgecolor="black")
    bars3 = ax.bar([i + width for i in x], y3, width=width, label="Stage3", color="#2ca02c", edgecolor="black")
    impl.annotate_bars(ax, bars1, "{:.3f}")
    impl.annotate_bars(ax, bars2, "{:.3f}")
    impl.annotate_bars(ax, bars3, "{:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(shared_tasks, rotation=30)
    ax.set_title(f"Tier3 Shared Tasks: Stage1 vs Stage2 vs Stage3 at Step {compare_step}", fontsize=12, weight="bold")
    ax.set_ylabel("Accuracy (higher is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "three_stage" / "tier3" / f"stage123_task_acc_step_{compare_step}.png",
        generated,
    )

    # 5) Stage3-minus-(Stage2, Stage1) macro deltas at overlap steps.
    t2_d_s3_s2 = [s3 - s2 for s3, s2 in zip(s3_t2_ov, s2_t2_ov)]  # negative better
    t2_d_s3_s1 = [s3 - s1 for s3, s1 in zip(s3_t2_ov, s1_t2_ov)]  # negative better
    t3_d_s3_s2 = [s3 - s2 for s3, s2 in zip(s3_t3_ov, s2_t3_ov)]  # positive better
    t3_d_s3_s1 = [s3 - s1 for s3, s1 in zip(s3_t3_ov, s1_t3_ov)]  # positive better

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    grouped_bar_by_stage(
        impl,
        axes[0],
        [str(s) for s in overlap_steps],
        {"Stage3-Stage2": t2_d_s3_s2, "Stage3-Stage1": t2_d_s3_s1},
        "Delta PPL (negative is better)",
        "Tier2 Macro Delta at Overlap Steps",
        percent_axis=False,
    )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)

    grouped_bar_by_stage(
        impl,
        axes[1],
        [str(s) for s in overlap_steps],
        {"Stage3-Stage2": t3_d_s3_s2, "Stage3-Stage1": t3_d_s3_s1},
        "Delta accuracy (positive is better)",
        "Tier3 Macro Delta at Overlap Steps",
        percent_axis=True,
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)

    impl.save_figure(
        fig,
        output_root / "stage_compare" / "three_stage" / "summary" / "stage3_minus_stage2_and_stage1_macro_delta_overlap.png",
        generated,
    )

    # 5b) Objective quadrants versus Stage2 and Stage1 at overlap steps.
    # Objective is met when Tier3 gain >= 0 and Tier2 loss <= 0.
    t2_loss_vs_s2 = [((s3 - s2) / s2 * 100.0) if math.isfinite(s2) and s2 != 0 else float("nan") for s3, s2 in zip(s3_t2_ov, s2_t2_ov)]
    t3_gain_vs_s2 = [((s3 - s2) / s2 * 100.0) if math.isfinite(s2) and s2 != 0 else float("nan") for s3, s2 in zip(s3_t3_ov, s2_t3_ov)]
    t2_loss_vs_s1 = [((s3 - s1) / s1 * 100.0) if math.isfinite(s1) and s1 != 0 else float("nan") for s3, s1 in zip(s3_t2_ov, s1_t2_ov)]
    t3_gain_vs_s1 = [((s3 - s1) / s1 * 100.0) if math.isfinite(s1) and s1 != 0 else float("nan") for s3, s1 in zip(s3_t3_ov, s1_t3_ov)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)
    for ax, xvals, yvals, title in [
        (axes[0], t2_loss_vs_s2, t3_gain_vs_s2, "Stage3 vs Stage2 Objective"),
        (axes[1], t2_loss_vs_s1, t3_gain_vs_s1, "Stage3 vs Stage1 Objective"),
    ]:
        finite_pairs = [(x, y) for x, y in zip(xvals, yvals) if math.isfinite(x) and math.isfinite(y)]
        if finite_pairs:
            xs = [x for x, _ in finite_pairs]
            ys = [y for _, y in finite_pairs]
            x_min = min(min(xs), 0.0)
            x_max = max(max(xs), 0.0)
            y_min = min(min(ys), 0.0)
            y_max = max(max(ys), 0.0)
            x_pad = max(0.5, (x_max - x_min) * 0.15)
            y_pad = max(0.5, (y_max - y_min) * 0.15)
            x_min -= x_pad
            x_max += x_pad
            y_min -= y_pad
            y_max += y_pad
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            if x_min < 0 and y_max > 0:
                ax.add_patch(
                    Rectangle(
                        (x_min, 0.0),
                        0.0 - x_min,
                        y_max - 0.0,
                        facecolor="#d8f3dc",
                        edgecolor="none",
                        alpha=0.25,
                        zorder=0,
                    )
                )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        sc = ax.scatter(xvals, yvals, c=range(len(overlap_steps)), cmap="viridis", s=95)
        for step, x, y in zip(overlap_steps, xvals, yvals):
            if math.isfinite(x) and math.isfinite(y):
                ax.annotate(str(step), (x, y), textcoords="offset points", xytext=(6, 5), fontsize=9)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("Overlap step index")

    axes[0].set_ylabel("Tier3 ACC change (%) [>= 0 means gain]")
    axes[0].set_xlabel("Tier2 PPL change (%) [<= 0 means no loss]")
    axes[1].set_xlabel("Tier2 PPL change (%) [<= 0 means no loss]")
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "three_stage" / "summary" / "stage3_objective_quadrants_vs_stage1_stage2_overlap.png",
        generated,
    )

    # 5c) Pass/fail chart for objective versus Stage2 and Stage1 on overlap steps.
    pass_vs_s2 = [1 if math.isfinite(x) and math.isfinite(y) and x <= 0 and y >= 0 else 0 for x, y in zip(t2_loss_vs_s2, t3_gain_vs_s2)]
    pass_vs_s1 = [1 if math.isfinite(x) and math.isfinite(y) and x <= 0 and y >= 0 else 0 for x, y in zip(t2_loss_vs_s1, t3_gain_vs_s1)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = list(range(len(overlap_steps)))
    width = 0.36
    bars1 = ax.bar([i - width / 2 for i in x], pass_vs_s2, width=width, label="vs Stage2", color="#2ca02c", edgecolor="black")
    bars2 = ax.bar([i + width / 2 for i in x], pass_vs_s1, width=width, label="vs Stage1", color="#9467bd", edgecolor="black")
    impl.annotate_bars(ax, bars1, "{:.0f}")
    impl.annotate_bars(ax, bars2, "{:.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in overlap_steps])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Pass"])
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel("Overlap step")
    ax.set_ylabel("Objective status")
    ax.set_title("Stage3 Objective Pass/Fail on Overlap Steps\n(pass = Tier3 gain and Tier2 no-loss)", fontsize=12, weight="bold")
    ax.legend()
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "three_stage" / "summary" / "stage3_objective_pass_fail_overlap.png",
        generated,
    )

    # 6) Per-slice and per-task overlap deltas for Stage3-Stage2 and Stage3-Stage1.
    series_t2_s3_s2: Dict[int, List[float]] = {}
    series_t2_s3_s1: Dict[int, List[float]] = {}
    series_t3_s3_s2: Dict[int, List[float]] = {}
    series_t3_s3_s1: Dict[int, List[float]] = {}

    for step in overlap_steps:
        v_t2_s3_s2: List[float] = []
        v_t2_s3_s1: List[float] = []
        for slc in shared_slices:
            s3 = finite_or_nan(stage3_data[step]["tier2"]["slices"].get(slc, {}).get("ppl"))
            s2 = finite_or_nan(stage2_data[step]["tier2"]["slices"].get(slc, {}).get("ppl"))
            s1 = finite_or_nan(stage1_data[step]["tier2"]["slices"].get(slc, {}).get("ppl"))
            v_t2_s3_s2.append(s3 - s2)
            v_t2_s3_s1.append(s3 - s1)
        series_t2_s3_s2[step] = v_t2_s3_s2
        series_t2_s3_s1[step] = v_t2_s3_s1

        v_t3_s3_s2: List[float] = []
        v_t3_s3_s1: List[float] = []
        for task in shared_tasks:
            s3 = finite_or_nan(stage3_data[step]["tier3"]["tasks"].get(task, {}).get("acc"))
            s2 = finite_or_nan(stage2_data[step]["tier3"]["tasks"].get(task, {}).get("acc"))
            s1 = finite_or_nan(stage1_data[step]["tier3"]["tasks"].get(task, {}).get("acc"))
            v_t3_s3_s2.append(s3 - s2)
            v_t3_s3_s1.append(s3 - s1)
        series_t3_s3_s2[step] = v_t3_s3_s2
        series_t3_s3_s1[step] = v_t3_s3_s1

    fig, ax = plt.subplots(figsize=(16, 7))
    impl.grouped_bar(
        ax=ax,
        categories=shared_slices,
        series=series_t2_s3_s2,
        y_label="Delta PPL (Stage3 - Stage2, negative is better)",
        title="Tier2 Shared-Slice Delta on Overlap Steps: Stage3 - Stage2",
        percent_axis=False,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", labelrotation=30)
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "three_stage" / "tier2" / "stage3_minus_stage2_shared_slice_ppl_delta_overlap.png",
        generated,
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    impl.grouped_bar(
        ax=ax,
        categories=shared_slices,
        series=series_t2_s3_s1,
        y_label="Delta PPL (Stage3 - Stage1, negative is better)",
        title="Tier2 Shared-Slice Delta on Overlap Steps: Stage3 - Stage1",
        percent_axis=False,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", labelrotation=30)
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "three_stage" / "tier2" / "stage3_minus_stage1_shared_slice_ppl_delta_overlap.png",
        generated,
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    impl.grouped_bar(
        ax=ax,
        categories=shared_tasks,
        series=series_t3_s3_s2,
        y_label="Delta accuracy (Stage3 - Stage2, positive is better)",
        title="Tier3 Shared-Task Delta on Overlap Steps: Stage3 - Stage2",
        percent_axis=True,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", labelrotation=30)
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "three_stage" / "tier3" / "stage3_minus_stage2_task_acc_delta_overlap.png",
        generated,
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    impl.grouped_bar(
        ax=ax,
        categories=shared_tasks,
        series=series_t3_s3_s1,
        y_label="Delta accuracy (Stage3 - Stage1, positive is better)",
        title="Tier3 Shared-Task Delta on Overlap Steps: Stage3 - Stage1",
        percent_axis=True,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", labelrotation=30)
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "three_stage" / "tier3" / "stage3_minus_stage1_task_acc_delta_overlap.png",
        generated,
    )


def main() -> None:
    args = parse_args()
    snapshots = parse_int_csv(args.snapshots)
    compare_steps = parse_int_csv(args.compare_steps)

    repo_root = Path(__file__).resolve().parents[3]
    impl = load_stage1_plot_impl(repo_root)
    configure_impl_for_stage3(impl)
    impl.setup_style()

    stage3_steps_all, stage3_data = impl.load_results(args.stage3_results_root)
    baseline_step = choose_stage3_baseline_step(stage3_steps_all)
    stage3_steps = [s for s in stage3_steps_all if s != baseline_step]
    if not stage3_steps:
        stage3_steps = list(stage3_steps_all)

    snapshots = [s for s in snapshots if s in stage3_steps]
    if not snapshots and stage3_steps:
        snapshots = [stage3_steps[0], stage3_steps[-1]]

    generated: List[Path] = []

    # Stage-3 charts with same filesystem/logic as Stage-1/Stage-2.
    impl.plot_tier2_series(stage3_steps, stage3_data, args.output_root, generated)
    impl.plot_tier3_series(stage3_steps, stage3_data, args.output_root, generated)
    impl.plot_macro_trends(stage3_steps, stage3_data, args.output_root, generated)
    impl.plot_relative_vs_baseline(stage3_steps, stage3_data, args.output_root, generated)
    impl.plot_tier3_support(stage3_steps, stage3_data, args.output_root, generated)
    impl.plot_snapshots(stage3_steps, stage3_data, snapshots, args.output_root, generated)
    impl.plot_domain_views(stage3_steps, stage3_data, snapshots, args.output_root, generated)
    impl.plot_early_late_delta(stage3_steps, stage3_data, args.output_root, generated)
    impl.plot_tradeoff(stage3_steps, stage3_data, args.output_root, generated)

    # Additional Stage-3 objective/focus charts.
    plot_stage3_focus(impl, stage3_steps, stage3_data, baseline_step, args.output_root, generated)
    plot_stage3_objective(impl, stage3_steps, stage3_data, baseline_step, args.output_root, generated)
    plot_stage3_bridge_focus(impl, stage3_steps, stage3_data, baseline_step, args.output_root, generated)

    # Stage1-vs-Stage2-vs-Stage3 comparisons.
    stage_compare_enabled = False
    stage_compare_error = ""
    stage1_steps: List[int] = []
    stage2_steps_all: List[int] = []
    stage2_steps: List[int] = []
    if not args.skip_stage_compare:
        try:
            stage1_steps, stage1_data = impl.load_results(args.stage1_results_root)
            stage2_steps_all, stage2_data = impl.load_results(args.stage2_results_root)
            stage2_steps = [s for s in stage2_steps_all if s != STAGE2_BRIDGE_STEP]
            if not stage2_steps:
                stage2_steps = list(stage2_steps_all)
            plot_three_stage_compare(
                impl,
                stage1_steps,
                stage1_data,
                stage2_steps,
                stage2_data,
                stage3_steps,
                stage3_data,
                compare_steps,
                args.output_root,
                generated,
            )
            stage_compare_enabled = True
        except Exception as exc:  # pragma: no cover
            stage_compare_error = str(exc)

    manifest = {
        "stage3_results_root": str(args.stage3_results_root),
        "stage2_results_root": str(args.stage2_results_root),
        "stage1_results_root": str(args.stage1_results_root),
        "output_root": str(args.output_root),
        "stage3_steps_all": stage3_steps_all,
        "stage3_steps_plotted": stage3_steps,
        "baseline_step": baseline_step,
        "stage2_steps_all": stage2_steps_all,
        "stage2_steps": stage2_steps,
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
