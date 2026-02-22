#!/usr/bin/env python3
# vibe_test_sft_chat_best.py
#
# Vibe test for your SFT-exported HF model using your Llama-3 chat-template tokenizer.
# - Robust EOS/EOT stopping
# - "Best vibe" sampling defaults (can switch to greedy with --mode greedy)
# - Optional debug prints (prompt tail + last token ids + stop reason)
#
# Usage (example):
#   python -u vibe_test_sft_chat_best.py
#   python -u vibe_test_sft_chat_best.py --mode greedy --debug
#   python -u vibe_test_sft_chat_best.py --max_new_tokens 220 --temperature 0.7 --top_p 0.92

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---- defaults (paths) ----
DEFAULT_EXPORT = "/leonardo_scratch/large/userexternal/mpeppas0/sft_project/hf_exports/sft_smoltalk_5000"
DEFAULT_TOKDIR = "/leonardo_scratch/large/userexternal/mpeppas0/sft_cache/tokenizers/llama3_with_chat_template"

DEFAULT_PROMPTS = [
    "If i have 9 apples and i eat 3, how many do i have left",
    "What is best for your health: (a)cigarettes (b) fruits , (c) burgers ",
    "Explain quantum computing to a five-year-old",
    "What is the correct answer to 25/5: (a)5 (b) 3 , (c) 0 ",
    "Which number is bigger: (a)2 (b) -3 , (c) 50 ",
    "Purple is : (a)a colour (b) a continent , (c) a human being",
    "If I have 3 apples and I eat 2, how many apples do I have? Now, if I buy 5 more, but one is a pear, how many apples do I have?",
    "You have a 4-hour study session. Create a plan with 6 time blocks.Include: (a) 2 short breaks, (b) one block for “active recall”, (c) one block for “practice problems”.Explain why the order is optimal in 3 sentences",
    "Write a simple Python script to check if a number is prime",
    "Who is the current President of the United States?",
    "Expand the following point : Kids dont like styding because its hard."
]

SYSTEM = "You are a helpful assistant."


def build_text(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt.strip()},
    ]
    # add_generation_prompt=True lets the template add the assistant prefix properly
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_eos_ids(tokenizer):
    """
    Collect possible stop token ids: eos + (eot/end_of_turn/endoftext if present).
    Returns unique list, keeping order.
    """
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))

    candidates = ["<|eot_id|>", "<|end_of_turn|>", "<|endoftext|>"]
    for tok in candidates:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None:
            continue
        # some tokenizers return unk for unknown tokens
        if tokenizer.unk_token_id is not None and tid == tokenizer.unk_token_id:
            continue
        eos_ids.append(int(tid))

    # unique, preserve order
    return list(dict.fromkeys(eos_ids))


def safe_generate(model, inputs, gen_kwargs):
    """
    Some HF versions may not support certain kwargs (e.g. typical_p).
    Try once; if TypeError, remove common offenders and retry.
    """
    try:
        return model.generate(**inputs, **gen_kwargs)
    except TypeError as e:
        msg = str(e)
        # Try removing "typical_p" if unsupported
        if "typical_p" in msg and "typical_p" in gen_kwargs:
            gen_kwargs = dict(gen_kwargs)
            gen_kwargs.pop("typical_p", None)
            return model.generate(**inputs, **gen_kwargs)
        # Try removing "top_k" if unsupported (rare)
        if "top_k" in msg and "top_k" in gen_kwargs:
            gen_kwargs = dict(gen_kwargs)
            gen_kwargs.pop("top_k", None)
            return model.generate(**inputs, **gen_kwargs)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=DEFAULT_EXPORT, help="HF export folder (model.safetensors + config.json)")
    ap.add_argument("--tokdir", default=DEFAULT_TOKDIR, help="Tokenizer folder with chat_template")
    ap.add_argument("--mode", default="vibe", choices=["vibe", "greedy"], help="vibe=sampling, greedy=deterministic")
    ap.add_argument("--debug", action="store_true", help="Print prompt tail, token ids, stop reason")
    ap.add_argument("--seed", type=int, default=0, help="Seed (only meaningful for sampling)")
    ap.add_argument("--max_length", type=int, default=2048, help="Max input length (truncation)")
    ap.add_argument("--max_new_tokens", type=int, default=300, help="Generation length")

    # sampling knobs (used in vibe mode)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top_p", type=float, default=0.92)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--typical_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.18)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=8)

    args = ap.parse_args()

    # Offline safety
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    torch.set_grad_enabled(False)

    if args.seed:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print("Loading tokenizer:", args.tokdir, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokdir, local_files_only=True, use_fast=True)

    # padding safety
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    eos_ids = get_eos_ids(tokenizer)
    print("eos_ids:", eos_ids, flush=True)

    print("Loading model:", args.export, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.export,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    if args.mode == "greedy":
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=eos_ids,  # list is OK
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=max(args.repetition_penalty, 1.15),
            no_repeat_ngram_size=max(args.no_repeat_ngram_size, 6),
            use_cache=True,
        )
    else:
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            typical_p=args.typical_p,  # will be auto-removed if unsupported
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    for i, prompt in enumerate(DEFAULT_PROMPTS, 1):
        prompt = prompt.strip()
        if not prompt:
            continue

        text = build_text(tokenizer, prompt)

        if args.debug:
            print("\n--- PROMPT TAIL (last 300 chars) ---")
            print(text[-300:])
            tmp = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            print("--- PROMPT TOKENS (last 30 ids) ---")
            ids = tmp["input_ids"][0]
            print(ids[-30:].tolist(), flush=True)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = safe_generate(model, inputs, gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = out[0, prompt_len:]
        last_id = int(out[0, -1].item()) if out.numel() else -1
        gen_len = int(new_tokens.shape[-1])

        stopped_by_eos = last_id in set(eos_ids)
        hit_max = gen_len >= int(gen_kwargs["max_new_tokens"])

        completion_dbg = tokenizer.decode(new_tokens, skip_special_tokens=False)
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if args.debug:
            print(f"[debug] gen_len={gen_len} last_id={last_id} stopped_by_eos={stopped_by_eos} hit_max={hit_max}", flush=True)
            print("[debug] completion tail:", completion_dbg[-200:], flush=True)

        print("=" * 88)
        print(f"[{i}] PROMPT:\n{prompt}\n")
        print("COMPLETION:\n")
        print(completion.strip())
        print()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
