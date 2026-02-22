# vibe_test_sft_chat_best.py
#
# Vibe test for your SFT-exported HF model using your Llama-3 chat-template tokenizer.
# Changes (v2):
# - Robust stopping via StoppingCriteria (works even if eos_token_id=list isn't supported)
# - Stop tokens include: eos + <|eot_id|> + <|end_of_text|> (if present)
# - Safer "best vibe" defaults for weaker models (less over-constrained decoding)
# - Greedy mode is truly clean (no repetition constraints)
# - Seed=0 now actually sets the seed (previously it didn't, because `if args.seed:` was false)
#
# Usage examples:
#   python -u vibe_test_sft_chat_best.py
#   python -u vibe_test_sft_chat_best.py --mode greedy --debug
#   python -u vibe_test_sft_chat_best.py --temperature 0.6 --top_p 0.9 --top_k 50
#   python -u vibe_test_sft_chat_best.py --temperature 0.5 --top_p 0.85 --repetition_penalty 1.08 --no_repeat_ngram_size 4 --debug

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList


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
    "You have a 4-hour study session. Create a plan with 6 time blocks. Include: (a) 2 short breaks, (b) one block for “active recall”, (c) one block for “practice problems”. Explain why the order is optimal in 3 sentences",
    "Write a simple Python script to check if a number is prime",
    # NOTE: time-dependent; keep only if you want to see hallucination behavior
    "Who is the current President of the United States?",
    "Expand the following point : Kids dont like styding because its hard."
]

SYSTEM = (
    "You are a concise assistant. Answer directly.\n"
    "For multiple-choice questions, output only the letter and the final answer.\n"
    "For math, show 1-2 short steps then the final result.\n"
)


def build_text(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt.strip()},
    ]
    # add_generation_prompt=True lets the template add the assistant prefix properly
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _tok_id(tokenizer, tok: str):
    try:
        tid = tokenizer.convert_tokens_to_ids(tok)
    except Exception:
        return None
    if tid is None:
        return None
    # some tokenizers return unk for unknown tokens
    if tokenizer.unk_token_id is not None and tid == tokenizer.unk_token_id:
        return None
    return int(tid)


def get_stop_ids(tokenizer):
    """
    Collect possible stop token ids: eos + common llama3-ish end markers if present.
    Returns unique list, keeping order.
    """
    stop_ids = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))

    # Llama-3 style
    candidates = [
        "<|eot_id|>",
        "<|end_of_text|>",
        # keep these as optional fallbacks (often absent)
        "<|end_of_turn|>",
        "<|endoftext|>",
    ]
    for tok in candidates:
        tid = _tok_id(tokenizer, tok)
        if tid is not None:
            stop_ids.append(tid)

    # unique, preserve order
    return list(dict.fromkeys(stop_ids))


class StopOnTokens(StoppingCriteria):
    """Stop generation when the last generated token is in stop_ids."""
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = set(int(x) for x in stop_ids if x is not None)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids is None or input_ids.numel() == 0:
            return False
        last_id = int(input_ids[0, -1].item())
        return last_id in self.stop_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=DEFAULT_EXPORT, help="HF export folder (model.safetensors + config.json)")
    ap.add_argument("--tokdir", default=DEFAULT_TOKDIR, help="Tokenizer folder with chat_template")
    ap.add_argument("--mode", default="vibe", choices=["vibe", "greedy"], help="vibe=sampling, greedy=deterministic")
    ap.add_argument("--debug", action="store_true", help="Print prompt tail, token ids, stop reason")
    ap.add_argument("--seed", type=int, default=0, help="Seed (>=0 sets determinism; use -1 to disable seeding)")
    ap.add_argument("--max_length", type=int, default=2048, help="Max input length (truncation)")
    ap.add_argument("--max_new_tokens", type=int, default=220, help="Generation length")

    # sampling knobs (used in vibe mode) — tuned for weaker models
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)

    args = ap.parse_args()

    # Offline safety
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    torch.set_grad_enabled(False)

    # Seed handling: seed=0 is valid; -1 disables seeding
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print("Loading tokenizer:", args.tokdir, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokdir, local_files_only=True, use_fast=True)

    # padding safety
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    stop_ids = get_stop_ids(tokenizer)
    print("stop_ids:", stop_ids, flush=True)

    print("Loading model:", args.export, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.export,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    stopping = StoppingCriteriaList([StopOnTokens(stop_ids)])

    if args.mode == "greedy":
        # Clean greedy: no extra constraints that can distort weak models
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,  # keep default eos if available
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping,
            use_cache=True,
        )
    else:
        # Balanced vibe: avoid over-constraining (no typical_p; no huge no_repeat_ngram)
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,  # plus stopping_criteria for eot/end_of_text
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping,
            use_cache=True,
        )

    for i, prompt in enumerate(DEFAULT_PROMPTS, 1):
        prompt = prompt.strip()
        if not prompt:
            continue

        text = build_text(tokenizer, prompt)

        # Debug: show prompt tail and last prompt token ids
        if args.debug:
            print("\n--- PROMPT TAIL (last 350 chars) ---")
            print(text[-350:])
            tmp = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            ids = tmp["input_ids"][0]
            print("--- PROMPT TOKENS (last 30 ids) ---")
            print(ids[-30:].tolist(), flush=True)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = out[0, prompt_len:]
        gen_len = int(new_tokens.shape[-1]) if new_tokens is not None else 0
        last_id = int(out[0, -1].item()) if out is not None and out.numel() else -1

        stopped_by_stop = last_id in set(stop_ids)
        hit_max = gen_len >= int(gen_kwargs["max_new_tokens"])

        completion_dbg = tokenizer.decode(new_tokens, skip_special_tokens=False)
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if args.debug:
            print(f"[debug] gen_len={gen_len} last_id={last_id} stopped_by_stop={stopped_by_stop} hit_max={hit_max}", flush=True)
            print("[debug] completion tail:", completion_dbg[-220:], flush=True)

        print("=" * 88)
        print(f"[{i}] PROMPT:\n{prompt}\n")
        print("COMPLETION:\n")
        print(completion.strip())
        print()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
