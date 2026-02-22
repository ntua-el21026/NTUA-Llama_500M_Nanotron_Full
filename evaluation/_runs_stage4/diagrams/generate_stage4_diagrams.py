#!/usr/bin/env python3
"""Generate Stage-4 evaluation diagrams (Tier2, Tier3, Tier4 + bridge views).

Reads Stage-4 results:
  evaluation/_runs_stage4/eval_results/by_checkpoint/step_<STEP>/
    - tier2_ppl.json
    - tier3_cf.json
    - tier4_sft_native.json

Optionally reads Stage-1/2/3 results for stage-history comparison charts.

Writes diagrams under:
  evaluation/_runs_stage4/diagrams/
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
import numpy as np


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

TIER4_METRIC_ORDER = [
    "instruction_match_rate",
    "all_checks_pass_rate",
    "format_valid_rate",
    "chat_clean_rate",
    "code_parse_rate",
    "json_valid_rate",
]

TIER4_METRIC_LABELS = {
    "instruction_match_rate": "Instruction match",
    "all_checks_pass_rate": "All checks pass",
    "format_valid_rate": "Format valid",
    "chat_clean_rate": "Chat clean",
    "code_parse_rate": "Code parse",
    "json_valid_rate": "JSON valid",
}

TIER4_CATEGORY_ORDER = [
    "format_following",
    "structured_output",
    "code_generation",
    "instruction_priority",
    "instruction_following",
    "reasoning_short",
    "safety_behavior",
    "chat_cleanliness",
]

TIER4_CATEGORY_COLORS = {
    "format_following": "#1f77b4",
    "structured_output": "#17becf",
    "code_generation": "#ff7f0e",
    "instruction_priority": "#9467bd",
    "instruction_following": "#bcbd22",
    "reasoning_short": "#2ca02c",
    "safety_behavior": "#d62728",
    "chat_cleanliness": "#7f7f7f",
}

DEFAULT_SNAPSHOTS = [500, 3500, 5000]
STAGE4_BRIDGE_STEP = 12000
STAGE2_BRIDGE_STEP = 50000
STAGE3_BRIDGE_STEP = 22000


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

    parser = argparse.ArgumentParser(description="Generate Stage-4 evaluation diagrams")
    parser.add_argument(
        "--stage4-results-root",
        type=Path,
        default=repo_root / "evaluation" / "_runs_stage4" / "eval_results" / "by_checkpoint",
        help="Stage-4 by_checkpoint root",
    )
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
        default=repo_root / "evaluation" / "_runs_stage4" / "diagrams",
        help="Output diagrams root",
    )
    parser.add_argument(
        "--snapshots",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SNAPSHOTS),
        help="Comma-separated snapshot steps for Stage-4 snapshot/domain charts",
    )
    parser.add_argument(
        "--skip-stage-history",
        action="store_true",
        help="Skip Stage1/2/3/4 stage-history comparison charts",
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


def configure_impl_for_stage4(impl: ModuleType) -> None:
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


def choose_stage4_baseline_step(all_steps: Sequence[int]) -> int:
    if STAGE4_BRIDGE_STEP in all_steps:
        return STAGE4_BRIDGE_STEP
    return min(all_steps)


def load_stage4_results(results_root: Path) -> Tuple[List[int], Dict[int, dict]]:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    data: Dict[int, dict] = {}
    for step_dir in sorted(results_root.glob("step_*")):
        if not step_dir.is_dir():
            continue
        try:
            step = int(step_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue

        tier2_path = step_dir / "tier2_ppl.json"
        tier3_path = step_dir / "tier3_cf.json"
        tier4_path = step_dir / "tier4_sft_native.json"
        if not (tier2_path.exists() and tier3_path.exists() and tier4_path.exists()):
            continue

        tier2 = json.loads(tier2_path.read_text(encoding="utf-8"))
        tier3 = json.loads(tier3_path.read_text(encoding="utf-8"))
        tier4 = json.loads(tier4_path.read_text(encoding="utf-8"))

        prompt_map = {
            str(p.get("id")): p
            for p in tier4.get("prompts", [])
            if isinstance(p, dict) and p.get("id") is not None
        }

        data[step] = {
            "tier2": {
                "slices": tier2.get("slices", {}),
                "macro": tier2.get("ppl_macro_avg"),
            },
            "tier3": {
                "tasks": tier3.get("tasks", {}),
                "macro": tier3.get("cf_macro_avg"),
            },
            "tier4": {
                "summary": tier4.get("summary", {}),
                "by_category": tier4.get("by_category", {}),
                "prompts": tier4.get("prompts", []),
                "prompt_map": prompt_map,
                "suite_meta": tier4.get("suite_meta", {}),
            },
        }

    steps = sorted(data.keys())
    if not steps:
        raise RuntimeError(f"No valid Stage-4 step results found under: {results_root}")
    return steps, data


def compute_group_macro_ppl(data: Dict[int, dict], steps: Sequence[int], slice_names: Sequence[str]) -> List[float]:
    out: List[float] = []
    for step in steps:
        vals: List[float] = []
        for name in slice_names:
            ppl = finite_or_nan(data[step]["tier2"]["slices"].get(name, {}).get("ppl"))
            if math.isfinite(ppl):
                vals.append(ppl)
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def compute_group_weighted_acc(data: Dict[int, dict], steps: Sequence[int], task_names: Sequence[str]) -> List[float]:
    out: List[float] = []
    for step in steps:
        weighted_num = 0.0
        weighted_den = 0.0
        for name in task_names:
            entry = data[step]["tier3"]["tasks"].get(name, {})
            acc = finite_or_nan(entry.get("acc"))
            n = finite_or_nan(entry.get("n"))
            if math.isfinite(acc) and math.isfinite(n) and n > 0:
                weighted_num += acc * n
                weighted_den += n
        out.append(weighted_num / weighted_den if weighted_den > 0 else float("nan"))
    return out


def tier4_summary_metric(data: Dict[int, dict], step: int, metric: str) -> float:
    return finite_or_nan(data[step]["tier4"]["summary"].get(metric))


def tier4_category_metric(data: Dict[int, dict], step: int, category: str, metric: str = "all_checks_pass_rate") -> float:
    return finite_or_nan(data[step]["tier4"]["by_category"].get(category, {}).get(metric))


def stage_order_with_baseline(baseline_step: int, steps: Sequence[int]) -> List[int]:
    ordered = [baseline_step]
    ordered.extend([s for s in sorted(steps) if s != baseline_step])
    return ordered


def plot_stage4_focus(
    impl: ModuleType,
    steps: Sequence[int],
    data: Dict[int, dict],
    baseline_step: int,
    output_root: Path,
    generated: List[Path],
) -> None:
    available_reasoning = [x for x in TIER2_REASONING_SLICES if any(x in data[s]["tier2"]["slices"] for s in steps)]

    if available_reasoning:
        fig, ax = plt.subplots(figsize=(12, 6))
        cmap = plt.get_cmap("tab10")
        for idx, slice_name in enumerate(available_reasoning):
            ys = [finite_or_nan(data[s]["tier2"]["slices"].get(slice_name, {}).get("ppl")) for s in steps]
            ax.plot(steps, ys, marker="o", linewidth=2.0, label=slice_name, color=cmap(idx % 10))
        ax.set_title("Stage-4 Tier2 Reasoning Slices: PPL vs Step", fontsize=12, weight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("PPL (lower is better)")
        ax.set_xticks(list(steps))
        ax.legend(title="Reasoning slice", loc="best")
        impl.save_figure(
            fig,
            output_root / "tier2" / "reasoning" / "tier2_reasoning_ppl_by_slice_vs_step.png",
            generated,
        )

    anchors = [x for x in TIER2_ANCHOR_SLICES if any(x in data[s]["tier2"]["slices"] for s in steps)]
    hq = [x for x in TIER2_HQ_SLICES if any(x in data[s]["tier2"]["slices"] for s in steps)]
    reasoning = [x for x in TIER2_REASONING_SLICES if any(x in data[s]["tier2"]["slices"] for s in steps)]

    series: Dict[str, List[float]] = {}
    if anchors:
        series["Anchor slices"] = compute_group_macro_ppl(data, steps, anchors)
    if hq:
        series["HQ slices"] = compute_group_macro_ppl(data, steps, hq)
    if reasoning:
        series["Reasoning slices"] = compute_group_macro_ppl(data, steps, reasoning)

    if series:
        fig, ax = plt.subplots(figsize=(12, 6))
        palette = {
            "Anchor slices": "#1f77b4",
            "HQ slices": "#ff7f0e",
            "Reasoning slices": "#2ca02c",
        }
        for label, ys in series.items():
            ax.plot(steps, ys, marker="o", linewidth=2.2, label=label, color=palette.get(label))
        ax.set_title("Stage-4 Tier2 Macro by Slice Group", fontsize=12, weight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Macro PPL (lower is better)")
        ax.set_xticks(list(steps))
        ax.legend(loc="best")
        impl.save_figure(
            fig,
            output_root / "tier2" / "reasoning" / "tier2_anchor_hq_reasoning_macro_ppl_vs_step.png",
            generated,
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        for label, ys in series.items():
            base_idx = steps.index(baseline_step) if baseline_step in steps else 0
            base = ys[base_idx]
            rel = []
            for y in ys:
                if not math.isfinite(base) or base == 0 or not math.isfinite(y):
                    rel.append(float("nan"))
                else:
                    rel.append((base - y) / base * 100.0)
            ax.plot(steps, rel, marker="o", linewidth=2.2, label=label, color=palette.get(label))
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Stage-4 Tier2 Group Relative vs Bridge (step {baseline_step})", fontsize=12, weight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Relative improvement (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xticks(list(steps))
        ax.legend(loc="best")
        impl.save_figure(
            fig,
            output_root / "tier2" / "reasoning" / "tier2_anchor_hq_reasoning_relative_vs_bridge.png",
            generated,
        )

    # Tier3 grouped macro by semantic group.
    group_order = ["commonsense", "science_qa", "reasoning_physical", "knowledge_reasoning"]
    group_series: Dict[str, List[float]] = {}
    for group in group_order:
        tasks = [k for k, g in TIER3_GROUP_MAP.items() if g == group and any(k in data[s]["tier3"]["tasks"] for s in steps)]
        if not tasks:
            continue
        group_series[group] = compute_group_weighted_acc(data, steps, tasks)

    if group_series:
        fig, ax = plt.subplots(figsize=(12, 6))
        for group, ys in group_series.items():
            ax.plot(
                steps,
                ys,
                marker="o",
                linewidth=2.0,
                label=group,
                color=TIER3_GROUP_COLORS.get(group, "#7f7f7f"),
            )
        ax.set_title("Stage-4 Tier3 Grouped Accuracy vs Step", fontsize=12, weight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Weighted accuracy (higher is better)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xticks(list(steps))
        ax.legend(loc="best")
        impl.save_figure(
            fig,
            output_root / "tier3" / "objective" / "tier3_group_macro_acc_vs_step.png",
            generated,
        )


def plot_tier4_summary_and_categories(
    impl: ModuleType,
    steps: Sequence[int],
    data: Dict[int, dict],
    baseline_step: int,
    snapshots: Sequence[int],
    output_root: Path,
    generated: List[Path],
) -> None:
    # Summary metric trends.
    fig, ax = plt.subplots(figsize=(13, 7))
    cmap = plt.get_cmap("tab10")
    for idx, metric in enumerate(TIER4_METRIC_ORDER):
        ys = [tier4_summary_metric(data, s, metric) for s in steps]
        ax.plot(steps, ys, marker="o", linewidth=2.0, label=TIER4_METRIC_LABELS.get(metric, metric), color=cmap(idx % 10))
    ax.set_title("Stage-4 Tier4 Summary Metrics vs Step", fontsize=12, weight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate (higher is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(list(steps))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, title="Metric")
    impl.save_figure(
        fig,
        output_root / "tier4" / "summary" / "tier4_summary_metrics_vs_step.png",
        generated,
        right_legend=True,
    )

    # Delta vs bridge for summary metrics.
    fig, ax = plt.subplots(figsize=(13, 7))
    for idx, metric in enumerate(TIER4_METRIC_ORDER):
        base = tier4_summary_metric(data, baseline_step, metric)
        ys = []
        for s in steps:
            curr = tier4_summary_metric(data, s, metric)
            ys.append(curr - base if math.isfinite(curr) and math.isfinite(base) else float("nan"))
        ax.plot(steps, ys, marker="o", linewidth=2.0, label=TIER4_METRIC_LABELS.get(metric, metric), color=cmap(idx % 10))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Stage-4 Tier4 Metric Delta vs Bridge (step {baseline_step})", fontsize=12, weight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Delta rate (positive is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(list(steps))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, title="Metric")
    impl.save_figure(
        fig,
        output_root / "tier4" / "objective" / "tier4_metric_delta_vs_stage3_bridge.png",
        generated,
        right_legend=True,
    )

    # Category pass-rate lines.
    categories = [
        c
        for c in TIER4_CATEGORY_ORDER
        if any(c in data[s]["tier4"]["by_category"] for s in steps)
    ]
    fig, ax = plt.subplots(figsize=(13, 7))
    for cat in categories:
        ys = [tier4_category_metric(data, s, cat, "all_checks_pass_rate") for s in steps]
        ax.plot(
            steps,
            ys,
            marker="o",
            linewidth=2.0,
            label=cat,
            color=TIER4_CATEGORY_COLORS.get(cat, "#7f7f7f"),
        )
    ax.set_title("Stage-4 Tier4 Category Pass Rate vs Step", fontsize=12, weight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("All-checks pass rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(list(steps))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, title="Category")
    impl.save_figure(
        fig,
        output_root / "tier4" / "categories" / "tier4_category_pass_rate_vs_step.png",
        generated,
        right_legend=True,
    )

    snap_steps = [s for s in snapshots if s in steps]
    if not snap_steps and steps:
        snap_steps = [steps[0], steps[-1]]

    if categories and snap_steps:
        series = {
            s: [tier4_category_metric(data, s, c, "all_checks_pass_rate") for c in categories]
            for s in snap_steps
        }
        fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.1), 6))
        impl.grouped_bar(
            ax=ax,
            categories=categories,
            series=series,
            y_label="All-checks pass rate",
            title="Stage-4 Tier4 Category Pass Rate (Early/Mid/Late)",
            percent_axis=True,
        )
        ax.tick_params(axis="x", labelrotation=30)
        impl.save_figure(
            fig,
            output_root / "tier4" / "categories" / "tier4_category_pass_rate_early_mid_late.png",
            generated,
        )

    # Pass/fail counts.
    fig, ax = plt.subplots(figsize=(11, 6))
    pass_counts = []
    fail_counts = []
    for s in steps:
        n_prompts = int(data[s]["tier4"]["summary"].get("n_prompts", 0) or 0)
        pass_rate = tier4_summary_metric(data, s, "all_checks_pass_rate")
        pass_n = int(round(pass_rate * n_prompts)) if math.isfinite(pass_rate) else 0
        fail_n = max(0, n_prompts - pass_n)
        pass_counts.append(pass_n)
        fail_counts.append(fail_n)

    x = np.arange(len(steps))
    bars1 = ax.bar(x, pass_counts, label="Pass", color="#2ca02c", edgecolor="black")
    bars2 = ax.bar(x, fail_counts, bottom=pass_counts, label="Fail", color="#d62728", edgecolor="black")
    impl.annotate_bars(ax, bars1, "{:.0f}")
    for i, b in enumerate(bars2):
        h = b.get_height()
        if h <= 0:
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            pass_counts[i] + h,
            f"{h:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_title("Stage-4 Tier4 Prompt Pass/Fail Counts", fontsize=12, weight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Prompt count")
    ax.legend(loc="best")
    impl.save_figure(
        fig,
        output_root / "tier4" / "summary" / "tier4_pass_fail_counts_vs_step.png",
        generated,
    )


def plot_tier4_prompt_views(
    impl: ModuleType,
    steps: Sequence[int],
    data: Dict[int, dict],
    output_root: Path,
    generated: List[Path],
) -> None:
    prompt_ids: List[str] = []
    for step in steps:
        prompt_ids.extend(list(data[step]["tier4"]["prompt_map"].keys()))

    if not prompt_ids:
        return

    seen = set()
    ordered_prompt_ids: List[str] = []
    for pid in sorted(prompt_ids):
        if pid in seen:
            continue
        seen.add(pid)
        ordered_prompt_ids.append(pid)

    matrix = []
    for pid in ordered_prompt_ids:
        row = []
        for step in steps:
            entry = data[step]["tier4"]["prompt_map"].get(pid, {})
            val = entry.get("all_checks_pass")
            row.append(1.0 if val is True else 0.0 if val is False else float("nan"))
        matrix.append(row)

    fig_h = max(8, len(ordered_prompt_ids) * 0.26)
    fig_w = max(7, len(steps) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(np.array(matrix), aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_title("Stage-4 Tier4 Prompt-Level All-Checks Pass Heatmap", fontsize=12, weight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Prompt ID")
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels([str(s) for s in steps])
    # Keep labels readable with many prompts.
    if len(ordered_prompt_ids) <= 60:
        ax.set_yticks(np.arange(len(ordered_prompt_ids)))
        ax.set_yticklabels(ordered_prompt_ids, fontsize=7)
    else:
        every = max(1, len(ordered_prompt_ids) // 40)
        yticks = np.arange(0, len(ordered_prompt_ids), every)
        ax.set_yticks(yticks)
        ax.set_yticklabels([ordered_prompt_ids[i] for i in yticks], fontsize=7)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Pass (1) / Fail (0)")
    impl.save_figure(
        fig,
        output_root / "tier4" / "prompts" / "tier4_prompt_pass_heatmap.png",
        generated,
    )

    # Prompt pass rate averaged across SFT checkpoints.
    prompt_pass = []
    for pid in ordered_prompt_ids:
        vals = []
        for step in steps:
            entry = data[step]["tier4"]["prompt_map"].get(pid, {})
            val = entry.get("all_checks_pass")
            if isinstance(val, bool):
                vals.append(1.0 if val else 0.0)
        score = sum(vals) / len(vals) if vals else float("nan")
        prompt_pass.append((pid, score))

    prompt_pass.sort(key=lambda x: (x[1], x[0]), reverse=True)
    labels = [x[0] for x in prompt_pass]
    vals = [x[1] for x in prompt_pass]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.24), 6))
    bars = ax.bar(labels, vals, color="#2ca02c", edgecolor="black", linewidth=0.5)
    for b, val in zip(bars, vals):
        if not math.isfinite(val):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            val,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    ax.set_title("Stage-4 Tier4 Prompt Pass Rate (Sorted)", fontsize=12, weight="bold")
    ax.set_xlabel("Prompt ID")
    ax.set_ylabel("Average all-checks pass rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="x", labelrotation=90)
    impl.save_figure(
        fig,
        output_root / "tier4" / "prompts" / "tier4_prompt_pass_rate_sorted.png",
        generated,
    )


def plot_stage4_objective(
    impl: ModuleType,
    steps: Sequence[int],
    data: Dict[int, dict],
    baseline_step: int,
    output_root: Path,
    generated: List[Path],
) -> None:
    if baseline_step not in data or not steps:
        return

    base_t2 = finite_or_nan(data[baseline_step]["tier2"]["macro"])
    base_t3 = finite_or_nan(data[baseline_step]["tier3"]["macro"])
    base_t4 = tier4_summary_metric(data, baseline_step, "all_checks_pass_rate")

    delta_t2 = []
    delta_t3 = []
    delta_t4 = []
    for step in steps:
        d2 = finite_or_nan(data[step]["tier2"]["macro"]) - base_t2
        d3 = finite_or_nan(data[step]["tier3"]["macro"]) - base_t3
        d4 = tier4_summary_metric(data, step, "all_checks_pass_rate") - base_t4
        delta_t2.append(d2)
        delta_t3.append(d3)
        delta_t4.append(d4)

    # Quadrant: Tier2 no-loss vs Tier3 gain, colored by Tier4 delta.
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(delta_t2, delta_t3, c=delta_t4, cmap="coolwarm", s=90, edgecolor="black")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    for step, x, y in zip(steps, delta_t2, delta_t3):
        ax.annotate(str(step), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_title(f"Stage-4 Objective (vs bridge {baseline_step}): Tier2 vs Tier3", fontsize=12, weight="bold")
    ax.set_xlabel("Tier2 macro delta (PPL step - bridge, <= 0 is better)")
    ax.set_ylabel("Tier3 macro delta (acc step - bridge, >= 0 is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Tier4 all-checks delta")
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    impl.save_figure(
        fig,
        output_root / "cross_tier" / "objective" / "stage4_objective_quadrant_tier2_tier3_vs_bridge.png",
        generated,
    )

    # Quadrant: Tier2 no-loss vs Tier4 gain, colored by Tier3 delta.
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(delta_t2, delta_t4, c=delta_t3, cmap="viridis", s=90, edgecolor="black")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    for step, x, y in zip(steps, delta_t2, delta_t4):
        ax.annotate(str(step), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_title(f"Stage-4 Objective (vs bridge {baseline_step}): Tier2 vs Tier4", fontsize=12, weight="bold")
    ax.set_xlabel("Tier2 macro delta (PPL step - bridge, <= 0 is better)")
    ax.set_ylabel("Tier4 all-checks delta (>= 0 is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Tier3 macro delta")
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    impl.save_figure(
        fig,
        output_root / "cross_tier" / "objective" / "stage4_objective_quadrant_tier2_tier4_vs_bridge.png",
        generated,
    )

    # Per-step pass/fail flags for objective constraints.
    tier2_pass = [1.0 if d <= 0 else 0.0 for d in delta_t2]
    tier3_pass = [1.0 if d >= 0 else 0.0 for d in delta_t3]
    tier4_pass = [1.0 if d >= 0 else 0.0 for d in delta_t4]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(steps))
    width = 0.24
    b1 = ax.bar(x - width, tier2_pass, width=width, label="Tier2 no-loss", color="#1f77b4", edgecolor="black")
    b2 = ax.bar(x, tier3_pass, width=width, label="Tier3 gain", color="#d62728", edgecolor="black")
    b3 = ax.bar(x + width, tier4_pass, width=width, label="Tier4 gain", color="#2ca02c", edgecolor="black")
    impl.annotate_bars(ax, b1, "{:.0f}")
    impl.annotate_bars(ax, b2, "{:.0f}")
    impl.annotate_bars(ax, b3, "{:.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Stage-4 Objective Pass Flags vs Bridge {baseline_step}", fontsize=12, weight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Pass flag (1=yes, 0=no)")
    ax.legend(loc="best")
    impl.save_figure(
        fig,
        output_root / "cross_tier" / "objective" / "stage4_objective_pass_flags_vs_bridge.png",
        generated,
    )


def plot_stage4_bridge_focus(
    impl: ModuleType,
    steps: Sequence[int],
    data: Dict[int, dict],
    baseline_step: int,
    output_root: Path,
    generated: List[Path],
) -> None:
    ordered = stage_order_with_baseline(baseline_step, steps)
    labels = [f"bridge:{baseline_step}" if s == baseline_step else str(s) for s in ordered]

    t2_vals = [finite_or_nan(data[s]["tier2"]["macro"]) for s in ordered]
    t3_vals = [finite_or_nan(data[s]["tier3"]["macro"]) for s in ordered]
    t4_vals = [tier4_summary_metric(data, s, "all_checks_pass_rate") for s in ordered]

    x = np.arange(len(ordered))
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(x, t2_vals, marker="o", linewidth=2.2, color="#1f77b4")
    impl.add_line_point_labels(axs[0], list(x), t2_vals, "{:.2f}")
    axs[0].set_title("Stage-4 vs Stage-3 Bridge: Tier2/Tier3/Tier4 Macro Trends", fontsize=12, weight="bold")
    axs[0].set_ylabel("Tier2 macro PPL")

    axs[1].plot(x, t3_vals, marker="o", linewidth=2.2, color="#d62728")
    impl.add_line_point_labels(axs[1], list(x), t3_vals, "{:.3f}")
    axs[1].set_ylabel("Tier3 macro acc")
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    axs[2].plot(x, t4_vals, marker="o", linewidth=2.2, color="#2ca02c")
    impl.add_line_point_labels(axs[2], list(x), t4_vals, "{:.3f}")
    axs[2].set_ylabel("Tier4 all-checks")
    axs[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axs[2].set_xlabel("Checkpoint")

    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels)

    impl.save_figure(
        fig,
        output_root / "stage_compare" / "bridge" / "stage4_vs_stage3_bridge_macro_trends.png",
        generated,
    )

    base_t2 = t2_vals[0]
    base_t3 = t3_vals[0]
    base_t4 = t4_vals[0]

    d_t2 = [v - base_t2 for v in t2_vals[1:]]
    d_t3 = [v - base_t3 for v in t3_vals[1:]]
    d_t4 = [v - base_t4 for v in t4_vals[1:]]
    step_labels = [str(s) for s in ordered[1:]]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)

    colors = ["#2ca02c" if x <= 0 else "#d62728" for x in d_t2]
    bars = axs[0].bar(step_labels, d_t2, color=colors, edgecolor="black")
    axs[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    impl.annotate_bars(axs[0], bars, "{:+.2f}")
    axs[0].set_title("Tier2 delta")
    axs[0].set_ylabel("PPL(step - bridge)")

    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in d_t3]
    bars = axs[1].bar(step_labels, d_t3, color=colors, edgecolor="black")
    axs[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    impl.annotate_bars(axs[1], bars, "{:+.3f}")
    axs[1].set_title("Tier3 delta")
    axs[1].set_ylabel("Acc(step - bridge)")
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in d_t4]
    bars = axs[2].bar(step_labels, d_t4, color=colors, edgecolor="black")
    axs[2].axhline(0.0, color="black", linestyle="--", linewidth=1)
    impl.annotate_bars(axs[2], bars, "{:+.3f}")
    axs[2].set_title("Tier4 delta")
    axs[2].set_ylabel("Pass(step - bridge)")
    axs[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.suptitle(f"Stage-4 Macro Delta vs Stage-3 Bridge (step {baseline_step})", fontsize=12, weight="bold")
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "bridge" / "stage4_minus_stage3_bridge_macro_delta_by_step.png",
        generated,
    )


def choose_stage_ref(steps: Sequence[int], bridge_step: int | None = None) -> int:
    vals = sorted(steps)
    if bridge_step is None:
        return vals[-1]
    without_bridge = [s for s in vals if s != bridge_step]
    return without_bridge[-1] if without_bridge else vals[-1]


def plot_stage_history_compare(
    impl: ModuleType,
    stage1_steps: Sequence[int],
    stage1_data: Dict[int, dict],
    stage2_steps: Sequence[int],
    stage2_data: Dict[int, dict],
    stage3_steps: Sequence[int],
    stage3_data: Dict[int, dict],
    stage4_steps: Sequence[int],
    stage4_data: Dict[int, dict],
    baseline_step: int,
    output_root: Path,
    generated: List[Path],
) -> None:
    s1_ref = choose_stage_ref(stage1_steps)
    s2_ref = choose_stage_ref(stage2_steps, bridge_step=STAGE2_BRIDGE_STEP)
    s3_ref = choose_stage_ref(stage3_steps, bridge_step=STAGE3_BRIDGE_STEP)

    labels = [
        f"S1:{s1_ref}",
        f"S2:{s2_ref}",
        f"S3:{s3_ref}",
        f"S4-bridge:{baseline_step}",
    ] + [f"S4:{s}" for s in sorted(stage4_steps)]

    t2_vals = [
        finite_or_nan(stage1_data[s1_ref]["tier2"]["macro"]),
        finite_or_nan(stage2_data[s2_ref]["tier2"]["macro"]),
        finite_or_nan(stage3_data[s3_ref]["tier2"]["macro"]),
        finite_or_nan(stage4_data[baseline_step]["tier2"]["macro"]),
    ] + [finite_or_nan(stage4_data[s]["tier2"]["macro"]) for s in sorted(stage4_steps)]

    t3_vals = [
        finite_or_nan(stage1_data[s1_ref]["tier3"]["macro"]),
        finite_or_nan(stage2_data[s2_ref]["tier3"]["macro"]),
        finite_or_nan(stage3_data[s3_ref]["tier3"]["macro"]),
        finite_or_nan(stage4_data[baseline_step]["tier3"]["macro"]),
    ] + [finite_or_nan(stage4_data[s]["tier3"]["macro"]) for s in sorted(stage4_steps)]

    x = np.arange(len(labels))

    fig, axs = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axs[0].plot(x, t2_vals, marker="o", linewidth=2.2, color="#1f77b4")
    impl.add_line_point_labels(axs[0], list(x), t2_vals, "{:.2f}")
    axs[0].set_ylabel("Tier2 macro PPL")
    axs[0].set_title("Stage History: Macro Progression (Stage1/2/3 + Stage4)", fontsize=12, weight="bold")

    axs[1].plot(x, t3_vals, marker="o", linewidth=2.2, color="#d62728")
    impl.add_line_point_labels(axs[1], list(x), t3_vals, "{:.3f}")
    axs[1].set_ylabel("Tier3 macro acc")
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axs[1].set_xlabel("Stage / checkpoint")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=25)

    impl.save_figure(
        fig,
        output_root / "stage_compare" / "history" / "stage1234_macro_progression.png",
        generated,
    )

    # Delta vs previous point.
    d_t2 = [t2_vals[i] - t2_vals[i - 1] for i in range(1, len(t2_vals))]
    d_t3 = [t3_vals[i] - t3_vals[i - 1] for i in range(1, len(t3_vals))]
    d_labels = [f"{labels[i-1]}->{labels[i]}" for i in range(1, len(labels))]

    fig, axs = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    colors_t2 = ["#2ca02c" if x <= 0 else "#d62728" for x in d_t2]
    bars = axs[0].bar(d_labels, d_t2, color=colors_t2, edgecolor="black")
    axs[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    impl.annotate_bars(axs[0], bars, "{:+.2f}")
    axs[0].set_ylabel("Tier2 delta PPL")

    colors_t3 = ["#2ca02c" if x >= 0 else "#d62728" for x in d_t3]
    bars = axs[1].bar(d_labels, d_t3, color=colors_t3, edgecolor="black")
    axs[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    impl.annotate_bars(axs[1], bars, "{:+.3f}")
    axs[1].set_ylabel("Tier3 delta acc")
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axs[1].tick_params(axis="x", labelrotation=30)

    fig.suptitle("Stage History: Macro Delta vs Previous Point", fontsize=12, weight="bold")
    impl.save_figure(
        fig,
        output_root / "stage_compare" / "history" / "stage1234_macro_delta_vs_previous.png",
        generated,
    )


def main() -> None:
    args = parse_args()
    snapshots = parse_int_csv(args.snapshots)

    repo_root = Path(__file__).resolve().parents[3]
    impl = load_stage1_plot_impl(repo_root)
    configure_impl_for_stage4(impl)
    impl.setup_style()

    stage4_steps_all, stage4_data = load_stage4_results(args.stage4_results_root)
    baseline_step = choose_stage4_baseline_step(stage4_steps_all)
    stage4_steps = [s for s in stage4_steps_all if s != baseline_step]
    if not stage4_steps:
        stage4_steps = list(stage4_steps_all)

    snapshots = [s for s in snapshots if s in stage4_steps]
    if not snapshots and stage4_steps:
        snapshots = [stage4_steps[0], stage4_steps[-1]]

    generated: List[Path] = []

    # Stage-4 Tier2/Tier3 charts with same filesystem/logic as previous stages.
    impl.plot_tier2_series(stage4_steps, stage4_data, args.output_root, generated)
    impl.plot_tier3_series(stage4_steps, stage4_data, args.output_root, generated)
    impl.plot_macro_trends(stage4_steps, stage4_data, args.output_root, generated)
    impl.plot_relative_vs_baseline(stage4_steps, stage4_data, args.output_root, generated)
    impl.plot_tier3_support(stage4_steps, stage4_data, args.output_root, generated)
    impl.plot_snapshots(stage4_steps, stage4_data, snapshots, args.output_root, generated)
    impl.plot_domain_views(stage4_steps, stage4_data, snapshots, args.output_root, generated)
    impl.plot_early_late_delta(stage4_steps, stage4_data, args.output_root, generated)
    impl.plot_tradeoff(stage4_steps, stage4_data, args.output_root, generated)

    # Additional Stage-4 charts.
    plot_stage4_focus(impl, stage4_steps, stage4_data, baseline_step, args.output_root, generated)
    plot_tier4_summary_and_categories(
        impl,
        stage4_steps,
        stage4_data,
        baseline_step,
        snapshots,
        args.output_root,
        generated,
    )
    plot_tier4_prompt_views(impl, stage4_steps, stage4_data, args.output_root, generated)
    plot_stage4_objective(impl, stage4_steps, stage4_data, baseline_step, args.output_root, generated)
    plot_stage4_bridge_focus(impl, stage4_steps, stage4_data, baseline_step, args.output_root, generated)

    # Optional stage-history comparison.
    stage_history_enabled = False
    stage_history_error = ""
    stage1_steps: List[int] = []
    stage2_steps: List[int] = []
    stage3_steps: List[int] = []
    if not args.skip_stage_history:
        try:
            stage1_steps, stage1_data = impl.load_results(args.stage1_results_root)
            stage2_steps, stage2_data = impl.load_results(args.stage2_results_root)
            stage3_steps, stage3_data = impl.load_results(args.stage3_results_root)
            plot_stage_history_compare(
                impl,
                stage1_steps,
                stage1_data,
                stage2_steps,
                stage2_data,
                stage3_steps,
                stage3_data,
                stage4_steps,
                stage4_data,
                baseline_step,
                args.output_root,
                generated,
            )
            stage_history_enabled = True
        except Exception as exc:  # pragma: no cover
            stage_history_error = str(exc)

    manifest = {
        "stage4_results_root": str(args.stage4_results_root),
        "stage3_results_root": str(args.stage3_results_root),
        "stage2_results_root": str(args.stage2_results_root),
        "stage1_results_root": str(args.stage1_results_root),
        "output_root": str(args.output_root),
        "stage4_steps_all": stage4_steps_all,
        "stage4_steps_plotted": stage4_steps,
        "baseline_step": baseline_step,
        "snapshots_requested": snapshots,
        "tier4_metrics": TIER4_METRIC_ORDER,
        "tier4_categories": TIER4_CATEGORY_ORDER,
        "stage_history_enabled": stage_history_enabled,
        "stage_history_error": stage_history_error,
        "stage1_steps": stage1_steps,
        "stage2_steps": stage2_steps,
        "stage3_steps": stage3_steps,
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
