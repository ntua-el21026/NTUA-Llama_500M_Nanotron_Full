#!/usr/bin/env python3
"""Generate paper figures with token-based x-axis.

Outputs are split into:
- pretrain_stages123/
- sft_stage4/

Requested figures:
Pretrain (Stages 1-3)
1) Tier2 perplexity per slice across tokens (with stage regions)
2) Tier3 semantic-category accuracy across tokens (with stage regions)
3) Tier2 macro perplexity across tokens (with stage regions)
4) Tier3 macro CF across tokens (with stage regions)

SFT (Stage 4)
1) Macro perplexity across SFT tokens (bridge + SFT checkpoints)
2) Macro CF across SFT tokens (bridge + SFT checkpoints)
3) Prompt-category pass rates (bridge vs final checkpoint)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import yaml
from matplotlib.patches import Patch


STAGE_ORDER = ["stage1", "stage2", "stage3"]
STAGE_COLORS = {
    "stage1": "#4e79a7",
    "stage2": "#f28e2b",
    "stage3": "#59a14f",
}
STAGE_NAMES = {
    "stage1": "Stage 1",
    "stage2": "Stage 2",
    "stage3": "Stage 3",
}

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

TIER2_DOMAIN_ORDER = ["web", "code", "math"]
TIER2_DOMAIN_COLORS = {
    "web": "#1f77b4",
    "code": "#ff7f0e",
    "math": "#2ca02c",
}

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

TIER3_GROUP_ORDER = [
    "commonsense",
    "science_qa",
    "reasoning_physical",
    "knowledge_reasoning",
]

TIER3_GROUP_COLORS = {
    "commonsense": "#9467bd",
    "science_qa": "#17becf",
    "reasoning_physical": "#d62728",
    "knowledge_reasoning": "#8c564b",
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

STAGE2_BRIDGE_STEP = 50000
STAGE3_BRIDGE_STEP = 22000
STAGE4_BRIDGE_STEP = 12000


@dataclass
class TokenSpec:
    tokens_per_step: int
    train_steps: int

    @property
    def stage_total_tokens(self) -> int:
        return self.tokens_per_step * self.train_steps


@dataclass
class StepEntry:
    tier2_macro: float
    tier3_macro: float
    tier2_slices: Dict[str, dict]
    tier3_tasks: Dict[str, dict]
    tier4_summary: Dict[str, float]
    tier4_by_category: Dict[str, dict]


@dataclass
class EvalPoint:
    origin_stage: str
    suite_stage: str
    step: int
    tokens: int
    entry: StepEntry


def finite_or_nan(x: Optional[float]) -> float:
    if x is None:
        return float("nan")
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(v):
        return float("nan")
    return v


def mean_finite(values: Sequence[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def setup_style() -> None:
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 140,
        }
    )


def save_figure(fig: plt.Figure, path: Path, generated: List[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)


def parse_token_spec(config_path: Path) -> TokenSpec:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tokens_cfg = cfg.get("tokens", {})
    parallel_cfg = cfg.get("parallelism", {})

    seq = int(tokens_cfg["sequence_length"])
    micro = int(tokens_cfg["micro_batch_size"])
    accum = int(tokens_cfg["batch_accumulation_per_replica"])
    dp = int(parallel_cfg["dp"])
    train_steps = int(tokens_cfg["train_steps"])

    tps = seq * micro * accum * dp
    return TokenSpec(tokens_per_step=tps, train_steps=train_steps)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage_entries(root: Path) -> Tuple[List[int], Dict[int, StepEntry]]:
    if not root.exists():
        raise FileNotFoundError(f"Results root not found: {root}")

    entries: Dict[int, StepEntry] = {}
    for step_dir in sorted(root.glob("step_*")):
        if not step_dir.is_dir():
            continue

        try:
            step = int(step_dir.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue

        t2_path = step_dir / "tier2_ppl.json"
        t3_path = step_dir / "tier3_cf.json"
        if not (t2_path.exists() and t3_path.exists()):
            continue

        t2 = load_json(t2_path)
        t3 = load_json(t3_path)

        t4_path = step_dir / "tier4_sft_native.json"
        if t4_path.exists():
            t4 = load_json(t4_path)
            t4_summary = t4.get("summary", {})
            t4_by_cat = t4.get("by_category", {})
        else:
            t4_summary = {}
            t4_by_cat = {}

        entries[step] = StepEntry(
            tier2_macro=finite_or_nan(t2.get("ppl_macro_avg")),
            tier3_macro=finite_or_nan(t3.get("cf_macro_avg")),
            tier2_slices=t2.get("slices", {}),
            tier3_tasks=t3.get("tasks", {}),
            tier4_summary=t4_summary,
            tier4_by_category=t4_by_cat,
        )

    steps = sorted(entries.keys())
    if not steps:
        raise RuntimeError(f"No valid step results under: {root}")
    return steps, entries


def ordered_names(names: Sequence[str], preferred: Sequence[str]) -> List[str]:
    idx = {n: i for i, n in enumerate(preferred)}
    return sorted(names, key=lambda n: (idx.get(n, 10_000), n))


def pretrain_tokens(stage: str, step: int, t1: TokenSpec, t2: TokenSpec, t3: TokenSpec) -> int:
    if stage == "stage1":
        return step * t1.tokens_per_step
    if stage == "stage2":
        return t1.stage_total_tokens + step * t2.tokens_per_step
    if stage == "stage3":
        return t1.stage_total_tokens + t2.stage_total_tokens + step * t3.tokens_per_step
    raise ValueError(stage)


def build_pretrain_points(
    s1_entries: Dict[int, StepEntry],
    s2_entries: Dict[int, StepEntry],
    s3_entries: Dict[int, StepEntry],
    t1: TokenSpec,
    t2: TokenSpec,
    t3: TokenSpec,
) -> List[EvalPoint]:
    # Keyed by checkpoint origin so later-stage suite can overwrite bridge duplicates.
    by_key: Dict[Tuple[str, int], EvalPoint] = {}

    for step, entry in sorted(s1_entries.items()):
        key = ("stage1", step)
        by_key[key] = EvalPoint(
            origin_stage="stage1",
            suite_stage="stage1",
            step=step,
            tokens=pretrain_tokens("stage1", step, t1, t2, t3),
            entry=entry,
        )

    for step, entry in sorted(s2_entries.items()):
        origin = "stage1" if step == STAGE2_BRIDGE_STEP else "stage2"
        key = (origin, step)
        by_key[key] = EvalPoint(
            origin_stage=origin,
            suite_stage="stage2",
            step=step,
            tokens=pretrain_tokens(origin, step, t1, t2, t3),
            entry=entry,
        )

    for step, entry in sorted(s3_entries.items()):
        origin = "stage2" if step == STAGE3_BRIDGE_STEP else "stage3"
        key = (origin, step)
        by_key[key] = EvalPoint(
            origin_stage=origin,
            suite_stage="stage3",
            step=step,
            tokens=pretrain_tokens(origin, step, t1, t2, t3),
            entry=entry,
        )

    points = list(by_key.values())
    points.sort(key=lambda p: (p.tokens, STAGE_ORDER.index(p.origin_stage), p.step))
    return points


def shade_pretrain_stage_regions(ax: plt.Axes, t1: TokenSpec, t2: TokenSpec, t3: TokenSpec) -> None:
    b0 = 0
    b1 = t1.stage_total_tokens
    b2 = t1.stage_total_tokens + t2.stage_total_tokens
    b3 = t1.stage_total_tokens + t2.stage_total_tokens + t3.stage_total_tokens

    spans = [
        (b0, b1, "stage1"),
        (b1, b2, "stage2"),
        (b2, b3, "stage3"),
    ]

    for left, right, stage in spans:
        ax.axvspan(left, right, color=STAGE_COLORS[stage], alpha=0.06)

    ax.axvline(b1, color=STAGE_COLORS["stage2"], linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(b2, color=STAGE_COLORS["stage3"], linestyle="--", linewidth=1.0, alpha=0.8)


def stage_area_legend_handles() -> List[Patch]:
    return [
        Patch(facecolor=STAGE_COLORS["stage1"], edgecolor="none", alpha=0.12, label="Stage 1 area"),
        Patch(facecolor=STAGE_COLORS["stage2"], edgecolor="none", alpha=0.12, label="Stage 2 area"),
        Patch(facecolor=STAGE_COLORS["stage3"], edgecolor="none", alpha=0.12, label="Stage 3 area"),
    ]


def apply_token_axis_billions(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _pos: f"{x/1e9:.1f}"))
    ax.set_xlabel("Cumulative training tokens (billions)")


def apply_token_axis_millions(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _pos: f"{x/1e6:.1f}"))
    ax.set_xlabel("SFT training tokens (millions)")


def fmt_tokens_m(tokens: float) -> str:
    return f"{tokens / 1e6:.1f}M tok"


def weighted_group_accuracy(entry: StepEntry, group: str) -> float:
    num = 0.0
    den = 0.0
    for task, grp in TIER3_GROUP_MAP.items():
        if grp != group:
            continue
        t = entry.tier3_tasks.get(task, {})
        acc = finite_or_nan(t.get("acc"))
        n = finite_or_nan(t.get("n"))
        if math.isfinite(acc) and math.isfinite(n) and n > 0:
            num += acc * n
            den += n
    if den <= 0:
        return float("nan")
    return num / den


def plot_pretrain_tier2_per_slice(
    points: Sequence[EvalPoint],
    t1: TokenSpec,
    t2: TokenSpec,
    t3: TokenSpec,
    out_dir: Path,
    generated: List[Path],
) -> None:
    xs = [p.tokens for p in points]

    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    line_handles = []
    for domain in TIER2_DOMAIN_ORDER:
        ys = []
        for p in points:
            vals = []
            for slc, d in TIER2_DOMAIN_MAP.items():
                if d != domain:
                    continue
                ppl = finite_or_nan(p.entry.tier2_slices.get(slc, {}).get("ppl"))
                if math.isfinite(ppl):
                    vals.append(ppl)
            ys.append(mean_finite(vals))
        (line,) = ax.plot(
            xs,
            ys,
            linewidth=2.3,
            color=TIER2_DOMAIN_COLORS[domain],
            label=domain,
        )
        ax.scatter(
            xs,
            ys,
            color=TIER2_DOMAIN_COLORS[domain],
            edgecolor="black",
            linewidth=0.6,
            s=36,
            zorder=3,
        )
        line_handles.append(line)

    # Overlay macro curve on the same chart for direct context.
    macro_ys = [p.entry.tier2_macro for p in points]
    (macro_line,) = ax.plot(
        xs,
        macro_ys,
        linewidth=2.4,
        linestyle="--",
        color="#111111",
        label="macro_ppl",
    )
    ax.scatter(
        xs,
        macro_ys,
        color="#111111",
        edgecolor="black",
        linewidth=0.6,
        s=38,
        zorder=3,
    )
    line_handles.append(macro_line)

    shade_pretrain_stage_regions(ax, t1, t2, t3)
    apply_token_axis_billions(ax)
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Tier 2 Domain Perplexity (Web/Code/Math) Across Pretraining Tokens", fontweight="bold")
    stage_handles = stage_area_legend_handles()
    handles = line_handles + stage_handles
    labels = [h.get_label() for h in line_handles] + [h.get_label() for h in stage_handles]
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=True,
        ncol=4,
        title="Legend",
    )

    save_figure(fig, out_dir / "pretrain_tier2_per_slice_vs_tokens.png", generated)


def plot_pretrain_tier3_semantic(
    points: Sequence[EvalPoint],
    t1: TokenSpec,
    t2: TokenSpec,
    t3: TokenSpec,
    out_dir: Path,
    generated: List[Path],
) -> None:
    xs = [p.tokens for p in points]

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    line_handles = []
    for group in TIER3_GROUP_ORDER:
        ys = [weighted_group_accuracy(p.entry, group) for p in points]
        (line,) = ax.plot(
            xs,
            ys,
            linewidth=2.0,
            color=TIER3_GROUP_COLORS.get(group, "#333333"),
            label=group,
        )
        ax.scatter(
            xs,
            ys,
            color=TIER3_GROUP_COLORS.get(group, "#333333"),
            edgecolor="black",
            linewidth=0.6,
            s=36,
            zorder=3,
        )
        line_handles.append(line)

    # Overlay macro CF curve on the same chart for direct context.
    macro_ys = [p.entry.tier3_macro for p in points]
    (macro_line,) = ax.plot(
        xs,
        macro_ys,
        linewidth=2.4,
        linestyle="--",
        color="#111111",
        label="macro_cf",
    )
    ax.scatter(
        xs,
        macro_ys,
        color="#111111",
        edgecolor="black",
        linewidth=0.6,
        s=38,
        zorder=3,
    )
    line_handles.append(macro_line)

    shade_pretrain_stage_regions(ax, t1, t2, t3)
    apply_token_axis_billions(ax)
    ax.set_ylabel("Weighted accuracy (higher is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Tier 3 Semantic-Category Accuracy Across Pretraining Tokens", fontweight="bold")
    stage_handles = stage_area_legend_handles()
    handles = line_handles + stage_handles
    labels = [h.get_label() for h in line_handles] + [h.get_label() for h in stage_handles]
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=True,
        ncol=4,
        title="Legend",
    )

    save_figure(fig, out_dir / "pretrain_tier3_semantic_categories_vs_tokens.png", generated)


def plot_pretrain_macro(
    points: Sequence[EvalPoint],
    t1: TokenSpec,
    t2: TokenSpec,
    t3: TokenSpec,
    out_dir: Path,
    generated: List[Path],
) -> None:
    xs = np.array([p.tokens for p in points])
    ys_t2 = np.array([p.entry.tier2_macro for p in points], dtype=float)
    ys_t3 = np.array([p.entry.tier3_macro for p in points], dtype=float)

    # Macro PPL
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.plot(xs, ys_t2, color="#1f77b4", linewidth=2.2, marker="o", markersize=4.4, label="Macro trend")
    for stage in STAGE_ORDER:
        sxs = [p.tokens for p in points if p.origin_stage == stage]
        sys = [p.entry.tier2_macro for p in points if p.origin_stage == stage]
        if sxs:
            ax.scatter(sxs, sys, color=STAGE_COLORS[stage], edgecolor="black", s=38, label=STAGE_NAMES[stage], zorder=3)

    shade_pretrain_stage_regions(ax, t1, t2, t3)
    apply_token_axis_billions(ax)
    ax.set_ylabel("Macro perplexity (lower is better)")
    ax.set_title("Tier 2 Macro Perplexity Across Pretraining Tokens", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=True, ncol=4, title="Legend")
    save_figure(fig, out_dir / "pretrain_tier2_macro_ppl_vs_tokens.png", generated)

    # Macro CF
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.plot(xs, ys_t3, color="#d62728", linewidth=2.2, marker="o", markersize=4.4, label="Macro trend")
    for stage in STAGE_ORDER:
        sxs = [p.tokens for p in points if p.origin_stage == stage]
        sys = [p.entry.tier3_macro for p in points if p.origin_stage == stage]
        if sxs:
            ax.scatter(sxs, sys, color=STAGE_COLORS[stage], edgecolor="black", s=38, label=STAGE_NAMES[stage], zorder=3)

    shade_pretrain_stage_regions(ax, t1, t2, t3)
    apply_token_axis_billions(ax)
    ax.set_ylabel("Macro CF accuracy (higher is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Tier 3 Macro CF Accuracy Across Pretraining Tokens", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=True, ncol=4, title="Legend")
    save_figure(fig, out_dir / "pretrain_tier3_macro_cf_vs_tokens.png", generated)


def build_sft_points(stage4_entries: Dict[int, StepEntry], sft_spec: TokenSpec) -> Tuple[int, List[Tuple[int, StepEntry]]]:
    if STAGE4_BRIDGE_STEP in stage4_entries:
        bridge_step = STAGE4_BRIDGE_STEP
    else:
        bridge_step = min(stage4_entries.keys())

    ckpts = [(s, e) for s, e in sorted(stage4_entries.items()) if s != bridge_step]
    return bridge_step, ckpts


def plot_sft_macro(stage4_entries: Dict[int, StepEntry], sft_spec: TokenSpec, out_dir: Path, generated: List[Path]) -> None:
    bridge_step, ckpts = build_sft_points(stage4_entries, sft_spec)
    bridge_entry = stage4_entries[bridge_step]

    xs = [0] + [step * sft_spec.tokens_per_step for step, _ in ckpts]
    ys_t2 = [bridge_entry.tier2_macro] + [entry.tier2_macro for _, entry in ckpts]
    ys_t3 = [bridge_entry.tier3_macro] + [entry.tier3_macro for _, entry in ckpts]

    labels = [f"Bridge ({fmt_tokens_m(0)})"] + [
        f"SFT ({fmt_tokens_m(step * sft_spec.tokens_per_step)})" for step, _ in ckpts
    ]

    # SFT macro PPL
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    ax.plot(xs, ys_t2, linewidth=2.4, marker="o", markersize=5, color="#1f77b4")
    ax.scatter([xs[0]], [ys_t2[0]], marker="D", s=70, color="#7f7f7f", edgecolor="black", label="Stage3 bridge")
    ax.scatter(xs[1:], ys_t2[1:], marker="o", s=52, color="#2ca02c", edgecolor="black", label="Stage4 checkpoints")

    for x, y, lbl in zip(xs, ys_t2, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    apply_token_axis_millions(ax)
    ax.set_ylabel("Macro perplexity (lower is better)")
    ax.set_title("SFT Final Results: Tier 2 Macro Perplexity vs SFT Tokens", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=True, ncol=3, title="Legend")
    save_figure(fig, out_dir / "sft_tier2_macro_ppl_vs_tokens.png", generated)

    # SFT macro CF
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    ax.plot(xs, ys_t3, linewidth=2.4, marker="o", markersize=5, color="#d62728")
    ax.scatter([xs[0]], [ys_t3[0]], marker="D", s=70, color="#7f7f7f", edgecolor="black", label="Stage3 bridge")
    ax.scatter(xs[1:], ys_t3[1:], marker="o", s=52, color="#2ca02c", edgecolor="black", label="Stage4 checkpoints")

    for x, y, lbl in zip(xs, ys_t3, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    apply_token_axis_millions(ax)
    ax.set_ylabel("Macro CF accuracy (higher is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("SFT Final Results: Tier 3 Macro CF Accuracy vs SFT Tokens", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=True, ncol=3, title="Legend")
    save_figure(fig, out_dir / "sft_tier3_macro_cf_vs_tokens.png", generated)


def plot_sft_prompt_category(
    stage4_entries: Dict[int, StepEntry],
    sft_spec: TokenSpec,
    out_dir: Path,
    generated: List[Path],
) -> None:
    bridge_step, ckpts = build_sft_points(stage4_entries, TokenSpec(tokens_per_step=1, train_steps=1))
    if not ckpts:
        return

    final_step, final_entry = ckpts[-1]
    bridge_entry = stage4_entries[bridge_step]
    bridge_tokens = 0.0
    final_tokens = final_step * sft_spec.tokens_per_step

    # Panel A: key summary prompt metrics (bridge vs final)
    summary_metrics = [
        ("instruction_match_rate", "Instruction match"),
        ("all_checks_pass_rate", "Strict all-checks"),
        ("format_valid_rate", "Format valid"),
        ("code_parse_rate", "Code parse"),
    ]
    summary_labels = [label for _key, label in summary_metrics]
    bridge_summary = [finite_or_nan(bridge_entry.tier4_summary.get(key)) for key, _label in summary_metrics]
    final_summary = [finite_or_nan(final_entry.tier4_summary.get(key)) for key, _label in summary_metrics]

    # Panel B: include every category with improvement in at least one check.
    metric_candidates = [
        ("all_checks_pass_rate", "strict"),
        ("format_valid_rate", "format"),
        ("chat_clean_rate", "chat"),
        ("instruction_match_rate", "instr"),
        ("code_parse_rate", "code"),
        ("json_valid_rate", "json"),
    ]
    categories = []
    for c in TIER4_CATEGORY_ORDER:
        if c in bridge_entry.tier4_by_category or c in final_entry.tier4_by_category:
            best_delta = float("-inf")
            best_metric = ""
            for metric_key, metric_label in metric_candidates:
                b = finite_or_nan(bridge_entry.tier4_by_category.get(c, {}).get(metric_key))
                f = finite_or_nan(final_entry.tier4_by_category.get(c, {}).get(metric_key))
                d = f - b if math.isfinite(b) and math.isfinite(f) else float("nan")
                if math.isfinite(d) and d > best_delta:
                    best_delta = d
                    best_metric = metric_label
            if math.isfinite(best_delta) and best_delta > 0:
                categories.append((c, best_delta, best_metric))
    categories.sort(key=lambda x: x[1], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), gridspec_kw={"width_ratios": [1.5, 1.0]})

    # Left panel
    ax0 = axes[0]
    x0 = np.arange(len(summary_labels))
    width = 0.38
    b1 = ax0.bar(
        x0 - width / 2,
        bridge_summary,
        width=width,
        color="#7f7f7f",
        edgecolor="black",
        label=f"Bridge ({fmt_tokens_m(bridge_tokens)})",
    )
    b2 = ax0.bar(
        x0 + width / 2,
        final_summary,
        width=width,
        color="#2ca02c",
        edgecolor="black",
        label=f"Final SFT ({fmt_tokens_m(final_tokens)})",
    )
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if not math.isfinite(h):
                continue
            ax0.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    ax0.set_xticks(x0)
    ax0.set_xticklabels(summary_labels, rotation=18, ha="right")
    ax0.set_ylabel("Pass rate")
    ax0.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax0.set_title("Prompt Summary Metrics", fontweight="bold")
    ax0.legend(loc="upper left", frameon=True)

    # Right panel
    ax1 = axes[1]
    if categories:
        y_names = [name for name, _d, _m in categories]
        deltas = [d for _name, d, _m in categories]
        metrics = [m for _name, _d, m in categories]
        bars = ax1.barh(y_names, deltas, color="#2ca02c", edgecolor="black")
        ax1.invert_yaxis()
        for bar, d, m in zip(bars, deltas, metrics):
            ax1.text(d + 0.005, bar.get_y() + bar.get_height() / 2, f"+{d:.2f} ({m})", va="center", fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No positive category deltas", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_yticks([])
    ax1.axvline(0.0, color="#666666", linewidth=1.0)
    ax1.set_xlabel("Best positive change across category checks (final - bridge)")
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_title("Prompt Categories with Positive Gains (Any Check)", fontweight="bold")

    fig.suptitle("SFT Prompt Evaluation: Bridge vs Final Checkpoint", fontweight="bold", y=1.02)

    save_figure(fig, out_dir / "sft_prompt_category_pass_bridge_vs_final.png", generated)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Generate token-based paper figures")
    p.add_argument("--stage1-root", type=Path, default=repo_root / "evaluation" / "_runs_stage1" / "eval_results" / "by_checkpoint")
    p.add_argument("--stage2-root", type=Path, default=repo_root / "evaluation" / "_runs_stage2" / "eval_results" / "by_checkpoint")
    p.add_argument("--stage3-root", type=Path, default=repo_root / "evaluation" / "_runs_stage3" / "eval_results" / "by_checkpoint")
    p.add_argument("--stage4-root", type=Path, default=repo_root / "evaluation" / "_runs_stage4" / "eval_results" / "by_checkpoint")

    p.add_argument("--config-stage1", type=Path, default=repo_root / "config" / "config_stage1.yaml")
    p.add_argument("--config-stage2", type=Path, default=repo_root / "config" / "config_stage2.yaml")
    p.add_argument("--config-stage3", type=Path, default=repo_root / "config" / "config_stage3.yaml")
    p.add_argument("--config-sft", type=Path, default=repo_root / "config" / "sft_stage3_smoltalk.yaml")

    p.add_argument("--output-root", type=Path, default=repo_root / "evaluation" / "paper" / "figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()

    t1_spec = parse_token_spec(args.config_stage1)
    t2_spec = parse_token_spec(args.config_stage2)
    t3_spec = parse_token_spec(args.config_stage3)
    sft_spec = parse_token_spec(args.config_sft)

    _, s1_entries = load_stage_entries(args.stage1_root)
    _, s2_entries = load_stage_entries(args.stage2_root)
    _, s3_entries = load_stage_entries(args.stage3_root)
    _, s4_entries = load_stage_entries(args.stage4_root)

    pretrain_points = build_pretrain_points(s1_entries, s2_entries, s3_entries, t1_spec, t2_spec, t3_spec)

    generated: List[Path] = []

    pretrain_out = args.output_root / "pretrain_stages123"
    sft_out = args.output_root / "sft_stage4"

    plot_pretrain_tier2_per_slice(pretrain_points, t1_spec, t2_spec, t3_spec, pretrain_out, generated)
    plot_pretrain_tier3_semantic(pretrain_points, t1_spec, t2_spec, t3_spec, pretrain_out, generated)
    plot_pretrain_macro(pretrain_points, t1_spec, t2_spec, t3_spec, pretrain_out, generated)

    plot_sft_macro(s4_entries, sft_spec, sft_out, generated)
    plot_sft_prompt_category(s4_entries, sft_spec, sft_out, generated)

    manifest = {
        "stage1_tokens_per_step": t1_spec.tokens_per_step,
        "stage2_tokens_per_step": t2_spec.tokens_per_step,
        "stage3_tokens_per_step": t3_spec.tokens_per_step,
        "sft_tokens_per_step": sft_spec.tokens_per_step,
        "stage1_total_tokens": t1_spec.stage_total_tokens,
        "stage2_total_tokens": t2_spec.stage_total_tokens,
        "stage3_total_tokens": t3_spec.stage_total_tokens,
        "sft_total_tokens": sft_spec.stage_total_tokens,
        "stage2_bridge_step": STAGE2_BRIDGE_STEP,
        "stage3_bridge_step": STAGE3_BRIDGE_STEP,
        "stage4_bridge_step": STAGE4_BRIDGE_STEP,
        "files_generated": [str(p) for p in generated],
    }

    manifest_path = args.output_root / "paper_figures_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    generated.append(manifest_path)

    print(f"Generated {len(generated)} files under {args.output_root}")
    for p in generated:
        print(p)


if __name__ == "__main__":
    main()
