import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# ---- CONFIG ----
HF_CACHE = "/leonardo_scratch/large/userexternal/mpeppas0/sft_cache"
DATASET_NAME = "HuggingFaceTB/smol-smoltalk"
SPLIT = "train"
MAX_SAMPLES = 20000  # smoke; βάλε None για full
OUT_DIR = "/leonardo_scratch/large/userexternal/mpeppas0/sft_cache/processed/smol_smoltalk_prompt_completion"
OUT_PARQUET = os.path.join(OUT_DIR, "train.parquet")

TOKENIZER_PATH = "/leonardo/home/userexternal/mpeppas0/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Force offline + cache
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE, "hub")
    os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("Loading dataset (offline cached)...", flush=True)
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    if MAX_SAMPLES is not None:
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

    print("Loading tokenizer (local path)...", flush=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

    # Convert each conversation to single (prompt, completion): take the LAST assistant message
    prompts, completions, sources = [], [], []
    for ex in ds:
        msgs = ex["messages"]

        # find last assistant
        last_a = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "assistant":
                last_a = i
                break

        if last_a is None:
            continue

        prompt_msgs = msgs[:last_a]
        completion = msgs[last_a].get("content", "")

        prompt_txt = tok.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        if not prompt_txt or not completion:
            continue

        prompts.append(prompt_txt)
        completions.append(completion)
        sources.append(ex.get("source", ""))

    out = Dataset.from_dict({"prompt": prompts, "completion": completions, "source": sources})
    print(out, flush=True)

    print(f"Saving to {OUT_PARQUET} ...", flush=True)
    out.to_parquet(OUT_PARQUET)
    print("Done.", flush=True)

if __name__ == "__main__":
    main()

