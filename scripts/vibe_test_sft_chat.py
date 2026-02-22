#!/usr/bin/env python3
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- paths (βάλε τα σωστά σου) ----
EXPORT = "/leonardo_scratch/large/userexternal/mpeppas0/sft_project/hf_exports/sft_smoltalk_5000"
TOKDIR = "/leonardo_scratch/large/userexternal/mpeppas0/sft_cache/tokenizers/llama3_with_chat_template"

PROMPTS = [
    "A recipe mixes sugar and flour in a 3:5 ratio by weight. You have 2 kg of flour. How much sugar is needed, and what is the total mixture weight?",
    "Solve the equation: 3x+8 = 5x + 1",
    "Explain quantum computing to a five-year-old",
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
    # add_generation_prompt=True -> αφήνει το template να βάλει σωστά το assistant prefix
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def get_eos_ids(tokenizer):
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    # Llama-3 style end-of-turn tokens (ανάλογα με το tokenizer μπορεί να υπάρχει μόνο ένα)
    for tok in ["<|eot_id|>", "<|end_of_turn|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            eos_ids.append(tid)

    # unique, keep order
    return list(dict.fromkeys(eos_ids))

def main():
    # offline safety (να μην πάει hub)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    torch.set_grad_enabled(False)

    print("Loading tokenizer:", TOKDIR, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKDIR, local_files_only=True, use_fast=True)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id, eot_id]
    print("eos_ids:", eos_ids, flush=True)


    # padding safety
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading model:", EXPORT, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        EXPORT,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    # gen_kwargs = dict(
    #     max_new_tokens=180,
    #     do_sample=True,
    #     temperature=0.7,
    #     top_p=0.9,
    #     repetition_penalty=1.12,
    #     eos_token_id=tokenizer.eos_token_id,
    #     pad_token_id=tokenizer.pad_token_id,
    #     use_cache=True,
    # )
    # --- DIAGNOSTIC (greedy) ---
    gen_kwargs = dict(
        max_new_tokens=100,
        do_sample=False,                 # greedy
        eos_token_id=eos_ids,            # <-- list!
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.15,
        no_repeat_ngram_size=6,
        use_cache=True,
    )

    for i, p in enumerate(PROMPTS, 1):

        text = build_text(tokenizer, p)

        print("\n--- PROMPT TAIL (last 300 chars) ---")
        print(text[-300:])
        print("--- PROMPT TOKENS (last 30 ids) ---")

        tmp = tokenizer(text, return_tensors="pt")

        print(tmp["input_ids"][0, -30:].tolist(), flush=True)

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        gen_len = out.shape[-1] - inputs["input_ids"].shape[-1]
        last_id = out[0, -1].item()
        stopped_by_eos = last_id in eos_ids
        hit_max = gen_len >= gen_kwargs["max_new_tokens"]

        print(f"[debug] gen_len={gen_len} last_id={last_id} "f"stopped_by_eos={stopped_by_eos} hit_max={hit_max}", flush=True)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = out[0, prompt_len:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=False)
        completion_dbg = tokenizer.decode(new_tokens, skip_special_tokens=False)
        print("[debug] completion tail:", completion_dbg[-200:], flush=True)

        print(completion)
        print("=" * 88)
        print(f"[{i}] PROMPT:\n{p}\n")
        print("COMPLETION:\n")
        print(completion.strip())
        print()

if __name__ == "__main__":
    main()
