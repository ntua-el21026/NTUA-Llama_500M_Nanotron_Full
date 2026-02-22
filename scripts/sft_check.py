# check_masking.py
import os, torch
from transformers import AutoTokenizer

TOKDIR="/leonardo_scratch/large/userexternal/mpeppas0/sft_cache/tokenizers/llama3_with_chat_template"
tok = AutoTokenizer.from_pretrained(TOKDIR, local_files_only=True, use_fast=True)

# ---- εδώ βάλε 1 παράδειγμα όπως *μπαίνει στο training* ----
SYSTEM="You are a helpful assistant."
USER="What is net metering?"
ASSISTANT="Net metering is ..."

# prompt (όπως στο training)
prompt = tok.apply_chat_template(
    [{"role":"system","content":SYSTEM},
     {"role":"user","content":USER}],
    tokenize=False,
    add_generation_prompt=True
)

# full text που τρέφεις στο model για SFT
full_text = prompt + ASSISTANT + tok.eos_token  # ή + "<|eot_id|>" ανάλογα με pipeline

enc = tok(full_text, return_tensors="pt")
input_ids = enc["input_ids"][0]

# labels: σωστό masking
labels = input_ids.clone()
prompt_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
labels[: prompt_ids.numel()] = -100   # mask ΟΛΟ το prompt (system+user+assistant header)
# (και mask padding αν έχεις)

masked = (labels == -100).sum().item()
total = labels.numel()
print("masked/total:", masked, "/", total, "(", masked/total, ")")

# Δείξε ένα window γύρω από το boundary
b = prompt_ids.numel()
print("Boundary token ids:", input_ids[b-10:b+10].tolist())
print("Boundary labels:", labels[b-10:b+10].tolist())
print("Decoded around boundary:\n", tok.decode(input_ids[b-50:b+50]))
