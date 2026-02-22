#!/usr/bin/env python3
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _load_eval_module(repo_root: Path):
    eval_path = repo_root / "evaluation" / "run_eval_stage2.py"
    spec = importlib.util.spec_from_file_location("stage2_eval_impl", str(eval_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load evaluation module from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_nanotron_imports(nanotron_root: Path) -> None:
    for p in (nanotron_root, nanotron_root / "src", nanotron_root / "examples"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


class _NanotronCausalLMWrapper(torch.nn.Module):
    def __init__(self, nanotron_model: torch.nn.Module):
        super().__init__()
        self.nanotron_model = nanotron_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Nanotron Llama model returns logits as [seq, batch, vocab_shard] for TP.
        # For our TP=1 checkpoints this is full vocab; convert to [batch, seq, vocab].
        sharded_logits = self.nanotron_model.model(input_ids=input_ids, input_mask=attention_mask)
        logits = sharded_logits.transpose(0, 1).contiguous()
        return SimpleNamespace(logits=logits)


def _load_nanotron_model(checkpoint_path: Path, nanotron_root: Path):
    _prepare_nanotron_imports(nanotron_root)

    from examples.llama.convert_weights import load_nanotron_model

    # Nanotron model loader expects distributed env, even for single-process eval.
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
        if slurm_job_id.isdigit():
            # Keep port in user range and make it deterministic per Slurm job.
            os.environ["MASTER_PORT"] = str(15000 + (int(slurm_job_id) % 40000))
        else:
            os.environ["MASTER_PORT"] = "29500"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for direct Nanotron scoring.")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = load_nanotron_model(
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=dtype,
    )
    model.eval()
    return model, device


def _load_examples(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_checkpoint_layout(checkpoint_path: Path) -> None:
    metadata_path = checkpoint_path / "checkpoint_metadata.json"
    if not metadata_path.exists():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return
    tp = int(metadata.get("tp", 1))
    cp = int(metadata.get("cp", 1))
    if tp != 1:
        raise RuntimeError(
            f"Unsupported checkpoint tensor parallel size tp={tp}. "
            "Stage-2 Nanotron scorer currently supports tp=1 checkpoints only."
        )
    if cp != 1:
        raise RuntimeError(
            f"Unsupported checkpoint context parallel size cp={cp}. "
            "Stage-2 Nanotron scorer currently supports cp=1 checkpoints only."
        )


def main() -> None:
    repo_root = Path(_require_env("STAGE2_REPO_ROOT"))
    eval_data_dir = Path(_require_env("STAGE2_EVAL_DATA_DIR"))
    results_dir = Path(_require_env("STAGE2_RESULTS_DIR"))
    checkpoint_path = Path(_require_env("LIGHTEVAL_CHECKPOINT_PATH"))
    tokenizer_path = _require_env("LIGHTEVAL_TOKENIZER")
    seq_len = int(_require_env("LIGHTEVAL_SEQ_LEN"))
    batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "4"))
    nanotron_root = Path(
        os.environ.get(
            "NANOTRON_ROOT",
            f"/leonardo_scratch/large/userexternal/{os.environ.get('USER', '')}/nanotron_smollm3",
        )
    )

    eval_impl = _load_eval_module(repo_root)
    _validate_checkpoint_layout(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    nt_model, device = _load_nanotron_model(checkpoint_path=checkpoint_path, nanotron_root=nanotron_root)
    model = _NanotronCausalLMWrapper(nt_model)
    model.eval()

    nan_tracker = {"nan_inf_count": 0}

    tier2 = {"slices": {}, "ppl_macro_avg": None}
    ppl_root = eval_data_dir / "ppl_slices"
    for slice_dir in sorted(ppl_root.glob("*")):
        tokens_path = slice_dir / "packed_tokens.pt"
        if not tokens_path.exists():
            continue
        tokens = torch.load(tokens_path, map_location="cpu")
        nll, token_count = eval_impl.compute_ppl(
            model=model,
            tokens=tokens,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=batch_size,
            device=device,
            nan_tracker=nan_tracker,
        )
        ppl = math.exp(nll) if math.isfinite(nll) else float("inf")
        tier2["slices"][slice_dir.name] = {"nll": nll, "ppl": ppl, "tokens": token_count}

    if tier2["slices"]:
        tier2["ppl_macro_avg"] = sum(v["ppl"] for v in tier2["slices"].values()) / len(tier2["slices"])

    tier3 = {"tasks": {}, "cf_macro_avg": None}
    cf_root = eval_data_dir / "cf_tasks"
    for task_dir in sorted(cf_root.glob("*")):
        examples_path = task_dir / "examples.jsonl"
        if not examples_path.exists():
            continue
        examples = _load_examples(examples_path)
        if not examples:
            continue
        correct = 0
        for ex in examples:
            scores = eval_impl.score_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt=ex["prompt"],
                candidates=ex["candidates"],
                seq_len=seq_len,
                device=device,
                nan_tracker=nan_tracker,
            )
            pred = int(max(range(len(scores)), key=lambda i: scores[i]))
            if pred == int(ex["label"]):
                correct += 1
        acc = correct / len(examples)
        tier3["tasks"][task_dir.name] = {"acc": acc, "n": len(examples)}

    if tier3["tasks"]:
        tier3["cf_macro_avg"] = sum(v["acc"] for v in tier3["tasks"].values()) / len(tier3["tasks"])

    results_dir.mkdir(parents=True, exist_ok=True)
    eval_impl.write_json(results_dir / "tier2_ppl.json", tier2)
    eval_impl.write_json(results_dir / "tier3_cf.json", tier3)

    print(f"[stage2-real] wrote {results_dir / 'tier2_ppl.json'}", flush=True)
    print(f"[stage2-real] wrote {results_dir / 'tier3_cf.json'}", flush=True)
    if nan_tracker["nan_inf_count"] > 0:
        print(f"[stage2-real] warning: nan/inf batches={nan_tracker['nan_inf_count']}", flush=True)


if __name__ == "__main__":
    main()
