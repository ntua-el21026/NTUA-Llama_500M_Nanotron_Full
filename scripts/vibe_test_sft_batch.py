# /leonardo/home/userexternal/mpeppas0/Smol_Project/scripts/vibe_test_sft_batch.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

EXPORT = "/leonardo_scratch/large/userexternal/mpeppas0/sft_project/hf_export/sft_smoke_hf_200"

PROMPTS = [
    "Explain the difference between net metering and net billing, shortly.",
    "Give me 3 bullet tips for improving study focus.",
    "Write a short Python function that checks if a number is prime."
]

def format_prompt_plain(prompt: str) -> str:
    # Αυτό ταιριάζει με το “system/user/assistant” text template που έβλεπες
    return (
        "system\n\nYou are a helpful assistant.\n"
        "user\n\n" + prompt.strip() + "\n"
        "assistant\n"
    )

def main():
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(
        EXPORT,
        local_files_only=True,
        use_fast=True,
    )

    # padding safety
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        EXPORT,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # αφήνει το HF να βάλει το model στη GPU
        low_cpu_mem_usage=True,
    ).eval()

    gen_kwargs = dict(
        max_new_tokens=160,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    for i, p in enumerate(PROMPTS, 1):
        text = format_prompt_plain(p)

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)

        # IMPORTANT: κόβουμε το prompt, κρατάμε μόνο τα νέα tokens
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = out[0, prompt_len:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print("=" * 80)
        print(f"[{i}] PROMPT: {p}\n")
        print("COMPLETION:\n")
        print(completion.strip())
        print()

if __name__ == "__main__":
    main()
