#!/usr/bin/env python3
import ast
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _load_eval_module(repo_root: Path):
    eval_path = repo_root / "evaluation" / "run_eval_stage4.py"
    spec = importlib.util.spec_from_file_location("stage4_eval_impl", str(eval_path))
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
            "Stage-4 Nanotron scorer currently supports tp=1 checkpoints only."
        )
    if cp != 1:
        raise RuntimeError(
            f"Unsupported checkpoint context parallel size cp={cp}. "
            "Stage-4 Nanotron scorer currently supports cp=1 checkpoints only."
        )


def _build_chat_prompt(tokenizer: AutoTokenizer, system_text: str, user_text: str) -> str:
    system_text = (system_text or "").strip()
    user_text = (user_text or "").strip()
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if system_text:
        return f"System: {system_text}\nUser: {user_text}\nAssistant:"
    return f"User: {user_text}\nAssistant:"


def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: List[int], penalty: float) -> torch.Tensor:
    if penalty == 1.0 or not generated_ids:
        return logits
    out = logits.clone()
    for token_id in set(generated_ids):
        value = out[token_id]
        if value < 0:
            out[token_id] = value * penalty
        else:
            out[token_id] = value / penalty
    return out


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    scaled = logits / max(temperature, 1e-6)
    probs = torch.softmax(scaled, dim=-1)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        if cutoff.any():
            cutoff[1:] = cutoff[:-1].clone()
            cutoff[0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            denom = sorted_probs.sum()
            if float(denom.item()) > 0:
                sorted_probs = sorted_probs / denom
                sampled = int(torch.multinomial(sorted_probs, num_samples=1).item())
                return int(sorted_indices[sampled].item())

    denom = probs.sum()
    if float(denom.item()) <= 0:
        return int(torch.argmax(logits).item())
    sampled = int(torch.multinomial(probs, num_samples=1).item())
    return sampled


def _generate_completion(
    model: _NanotronCausalLMWrapper,
    tokenizer: AutoTokenizer,
    prompt: str,
    seq_len: int,
    generation_cfg: Dict[str, float],
    device: torch.device,
) -> str:
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 160))
    temperature = float(generation_cfg.get("temperature", 0.0))
    top_p = float(generation_cfg.get("top_p", 1.0))
    repetition_penalty = float(generation_cfg.get("repetition_penalty", 1.0))

    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    if input_ids.shape[1] > seq_len:
        input_ids = input_ids[:, -seq_len:]

    eos_token_id = tokenizer.eos_token_id
    generated_ids: List[int] = []

    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        next_logits = logits[0, -1, :].float()
        next_logits = _apply_repetition_penalty(next_logits, generated_ids, repetition_penalty)
        token_id = _sample_next_token(next_logits, temperature, top_p)
        generated_ids.append(token_id)

        next_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if input_ids.shape[1] > seq_len:
            input_ids = input_ids[:, -seq_len:]
        if eos_token_id is not None and token_id == eos_token_id:
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _normalize_list_field(checks: dict, key: str) -> List[str]:
    value = checks.get(key, [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    raise RuntimeError(f"Tier4 check '{key}' must be a string or list, got {type(value)}")


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _bullet_count(text: str) -> int:
    return len([line for line in text.splitlines() if re.match(r"^\s*[-*+]\s+", line)])


def _numbered_count(text: str) -> int:
    return len([line for line in text.splitlines() if re.match(r"^\s*\d+[.)]\s+", line)])


def _is_chat_clean(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    leak_patterns = [
        r"(^|\n)\s*(user|assistant|system)\s*:",
        r"<\|start_header_id\|>",
        r"<\|end_header_id\|>",
        r"\[/?INST\]",
    ]
    return not any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in leak_patterns)


def _evaluate_checks(response: str, checks: dict) -> Tuple[Dict[str, bool], bool, bool, bool, bool]:
    response_lower = response.lower()
    check_results: Dict[str, bool] = {}

    must_include = _normalize_list_field(checks, "must_include")
    if must_include:
        check_results["must_include"] = all(token.lower() in response_lower for token in must_include)

    must_not_include = _normalize_list_field(checks, "must_not_include")
    if must_not_include:
        check_results["must_not_include"] = all(token.lower() not in response_lower for token in must_not_include)

    contains_any = _normalize_list_field(checks, "contains_any")
    if contains_any:
        check_results["contains_any"] = any(token.lower() in response_lower for token in contains_any)

    one_of_outputs = _normalize_list_field(checks, "one_of_outputs")
    if one_of_outputs:
        normalized = response.strip().lower()
        allowed = {option.strip().lower() for option in one_of_outputs}
        check_results["one_of_outputs"] = normalized in allowed

    max_words = checks.get("max_words")
    if max_words is not None:
        check_results["max_words"] = _word_count(response) <= int(max_words)

    min_bullets = checks.get("min_bullets")
    if min_bullets is not None:
        check_results["min_bullets"] = _bullet_count(response) >= int(min_bullets)

    min_numbered_items = checks.get("min_numbered_items")
    if min_numbered_items is not None:
        check_results["min_numbered_items"] = _numbered_count(response) >= int(min_numbered_items)

    json_requested = bool(checks.get("expect_json", False))
    if json_requested:
        try:
            json.loads(response)
            check_results["expect_json"] = True
        except Exception:
            check_results["expect_json"] = False

    python_requested = bool(checks.get("python_parse", False))
    if python_requested:
        try:
            ast.parse(response)
            check_results["python_parse"] = True
        except Exception:
            check_results["python_parse"] = False

    if not check_results:
        check_results["non_empty"] = bool(response.strip())

    format_keys = {"max_words", "min_bullets", "min_numbered_items", "expect_json", "python_parse", "one_of_outputs"}
    format_checks = [result for key, result in check_results.items() if key in format_keys]
    format_valid = all(format_checks) if format_checks else bool(response.strip())

    pass_all = all(check_results.values())
    return check_results, pass_all, format_valid, python_requested, json_requested


def _compute_tier4(
    model: _NanotronCausalLMWrapper,
    tokenizer: AutoTokenizer,
    eval_data_dir: Path,
    results_dir: Path,
    seq_len: int,
    device: torch.device,
) -> dict:
    prompts_path = eval_data_dir / "sft_native" / "prompts.jsonl"
    meta_path = eval_data_dir / "sft_native" / "meta.json"
    if not prompts_path.exists() or not meta_path.exists():
        raise RuntimeError("Missing Tier4 prompt cache under eval_data_dir/sft_native")
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_examples(prompts_path)
    suite_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    generation_cfg = suite_meta.get("generation", {}) or {}

    prompt_results: List[dict] = []
    category_stats: Dict[str, Dict[str, float]] = {}

    pass_count = 0
    format_count = 0
    chat_clean_count = 0
    python_candidates = 0
    python_pass = 0
    json_candidates = 0
    json_pass = 0

    for prompt_cfg in prompts:
        prompt_id = str(prompt_cfg.get("id", "unknown"))
        category = str(prompt_cfg.get("category", "general"))
        system_text = str(prompt_cfg.get("system", "You are a helpful assistant."))
        user_text = str(prompt_cfg.get("user", ""))
        checks = prompt_cfg.get("checks", {}) or {}

        prompt_text = _build_chat_prompt(tokenizer, system_text, user_text)
        response = _generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            seq_len=seq_len,
            generation_cfg=generation_cfg,
            device=device,
        )

        check_results, pass_all, format_valid, python_requested, json_requested = _evaluate_checks(response, checks)
        chat_clean = _is_chat_clean(response)

        pass_count += int(pass_all)
        format_count += int(format_valid)
        chat_clean_count += int(chat_clean)

        if python_requested:
            python_candidates += 1
            python_pass += int(check_results.get("python_parse", False))
        if json_requested:
            json_candidates += 1
            json_pass += int(check_results.get("expect_json", False))

        if category not in category_stats:
            category_stats[category] = {"n": 0, "pass": 0, "format": 0, "clean": 0}
        category_stats[category]["n"] += 1
        category_stats[category]["pass"] += int(pass_all)
        category_stats[category]["format"] += int(format_valid)
        category_stats[category]["clean"] += int(chat_clean)

        prompt_results.append(
            {
                "id": prompt_id,
                "category": category,
                "response": response,
                "checks": check_results,
                "all_checks_pass": pass_all,
                "format_valid": format_valid,
                "chat_clean": chat_clean,
                "word_count": _word_count(response),
                "bullet_count": _bullet_count(response),
                "numbered_count": _numbered_count(response),
            }
        )

    n_prompts = len(prompt_results)
    by_category = {}
    for category, stats in category_stats.items():
        n_cat = int(stats["n"])
        by_category[category] = {
            "n": n_cat,
            "all_checks_pass_rate": stats["pass"] / n_cat if n_cat else None,
            "format_valid_rate": stats["format"] / n_cat if n_cat else None,
            "chat_clean_rate": stats["clean"] / n_cat if n_cat else None,
        }

    tier4 = {
        "summary": {
            "n_prompts": n_prompts,
            "instruction_match_rate": pass_count / n_prompts if n_prompts else None,
            "all_checks_pass_rate": pass_count / n_prompts if n_prompts else None,
            "format_valid_rate": format_count / n_prompts if n_prompts else None,
            "chat_clean_rate": chat_clean_count / n_prompts if n_prompts else None,
            "code_parse_rate": (python_pass / python_candidates) if python_candidates else None,
            "json_valid_rate": (json_pass / json_candidates) if json_candidates else None,
        },
        "suite_meta": {
            "version": suite_meta.get("version"),
            "prompt_count": suite_meta.get("prompt_count"),
            "generation": generation_cfg,
        },
        "by_category": by_category,
        "prompts": prompt_results,
    }

    generations_jsonl = results_dir / "tier4_sft_native_generations.jsonl"
    generations_jsonl.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in prompt_results),
        encoding="utf-8",
    )
    return tier4


def main() -> None:
    repo_root = Path(_require_env("STAGE4_REPO_ROOT"))
    eval_data_dir = Path(_require_env("STAGE4_EVAL_DATA_DIR"))
    results_dir = Path(_require_env("STAGE4_RESULTS_DIR"))
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

    tier4 = _compute_tier4(
        model=model,
        tokenizer=tokenizer,
        eval_data_dir=eval_data_dir,
        results_dir=results_dir,
        seq_len=seq_len,
        device=device,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    eval_impl.write_json(results_dir / "tier2_ppl.json", tier2)
    eval_impl.write_json(results_dir / "tier3_cf.json", tier3)
    eval_impl.write_json(results_dir / "tier4_sft_native.json", tier4)

    print(f"[stage4-real] wrote {results_dir / 'tier2_ppl.json'}", flush=True)
    print(f"[stage4-real] wrote {results_dir / 'tier3_cf.json'}", flush=True)
    print(f"[stage4-real] wrote {results_dir / 'tier4_sft_native.json'}", flush=True)
    if nan_tracker["nan_inf_count"] > 0:
        print(f"[stage4-real] warning: nan/inf batches={nan_tracker['nan_inf_count']}", flush=True)


if __name__ == "__main__":
    main()
