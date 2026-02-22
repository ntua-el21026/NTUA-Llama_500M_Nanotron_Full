import os, json
from datasets import load_dataset
from transformers import AutoTokenizer

CONFIG = os.environ.get("FWE_CONFIG_FINEWEB", "sample-10BT")
OUT_DIR_FINEWEB = os.environ["OUT_DIR_FINEWEB"]
TOKENIZER = os.environ["LLAMA_TOKENIZER_FINEWEB"]
TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET_FINEWEB", "1000000000"))
SHARD_DOCS = int(os.environ.get("SHARD_DOCS_FINEWEB", "200000"))
LANG = os.environ.get("LANG_FILTER_FINEWEB", "en")

def open_shard(out_dir, shard_id):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"finewebedu_int45_{shard_id:05d}.jsonl")
    return open(path, "w", encoding="utf-8"), path

def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=CONFIG, split="train", streaming=True)

    shard = 0
    f, shard_path = open_shard(OUT_DIR, shard)
    kept_docs = 0
    total_tokens = 0

    for ex in ds:
        s = ex.get("int_score", None)

        # Κρατάμε ΜΟΝΟ int_score 4 ή 5
        if s not in (4, 5):
            continue

        # Προαιρετικά: κρατάμε μόνο αγγλικά
        if LANG and ex.get("language", None) != LANG:
            continue

        text = ex.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        ids = tok(text, add_special_tokens=False)["input_ids"]
        n = len(ids)

        # Σταμάτα όταν πιάσουμε το token budget
        if total_tokens + n > TOKEN_BUDGET:
            break

        total_tokens += n
        kept_docs += 1
        f.write(json.dumps({"text": text, "int_score": s}, ensure_ascii=False) + "\n")

        # Κάνουμε shard rotation για να μην βγει 1 τεράστιο αρχείο
        if kept_docs % SHARD_DOCS == 0:
            f.close()
            shard += 1
            f, shard_path = open_shard(OUT_DIR, shard)

        if kept_docs % 50000 == 0:
            print(f"kept_docs={kept_docs:,} llama_tokens={total_tokens:,} last_shard={shard_path}")

    f.close()
    print("\nDONE")
    print(f"kept_docs={kept_docs:,}")
    print(f"llama_tokens={total_tokens:,}")
    print(f"out_dir={OUT_DIR}")

if __name__ == "__main__":
    main()


