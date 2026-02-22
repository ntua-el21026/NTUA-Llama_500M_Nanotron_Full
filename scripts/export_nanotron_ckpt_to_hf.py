#!/usr/bin/env python3
# export_nanotron_ckpt_to_hf.py
#
# Convert a Nanotron LLaMA checkpoint folder (e.g. .../200/) to HuggingFace format.
#
# Expected Nanotron structure (like yours):
#   CKPT/
#     model_config.json
#     model/
#       model/
#         token_position_embeddings/...
#         decoder/<layer>/pp_block/attn/qkv_proj/...
#         decoder/<layer>/pp_block/attn/o_proj/...
#         decoder/<layer>/pp_block/mlp/gate_up_proj/...
#         decoder/<layer>/pp_block/mlp/down_proj/...
#         (norms)
#
# Output HF:
#   OUTDIR/
#     config.json
#     generation_config.json (optional)
#     tokenizer.json, tokenizer_config.json, special_tokens_map.json (copied/symlinked)
#     model.safetensors
#
# Usage:
#   python export_nanotron_ckpt_to_hf.py \
#       --ckpt /path/to/.../ckpts/.../200 \
#       --out  /path/to/export_hf \
#       --tokenizer /path/to/hf_tokenizer_snapshot
#
# You can point --tokenizer to your Llama-3 tokenizer snapshot:
#   /leonardo/home/userexternal/mpeppas0/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/<hash>

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors.torch import save_file
from safetensors import safe_open


# -------------------------
# Helpers
# -------------------------
def must_exist(p: str) -> str:
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def read_json(p: str) -> dict:
    return json.loads(Path(p).read_text())


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_first_tensor(path: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"No tensors in {path}")
        return f.get_tensor(keys[0])


def load_named_tensor(path: str, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def find_unique(pattern: str) -> str:
    hits = sorted(glob.glob(pattern, recursive=True))
    if len(hits) == 0:
        raise FileNotFoundError(f"Missing pattern: {pattern}")
    if len(hits) > 1:
        # often OK but safer to pick shortest
        hits.sort(key=lambda x: len(x))
    return hits[0]


def symlink_or_copy(src: str, dst: str):
    srcp = Path(src)
    dstp = Path(dst)
    if dstp.exists():
        return
    try:
        dstp.symlink_to(srcp)
    except Exception:
        # fallback to copy
        import shutil
        shutil.copy2(srcp, dstp)


def infer_model_dims(ckpt_root: str) -> Dict[str, int]:
    """
    Infer vocab_size, hidden, n_layers, num_heads/num_kv_heads, head_dim from checkpoint files.
    You already computed these; script re-derives robustly.
    """
    model_root = os.path.join(ckpt_root, "model", "model")

    # vocab_size & hidden from token embedding weight
    emb_path = find_unique(os.path.join(model_root, "**", "token_embedding", "*.safetensors"))
    emb = load_first_tensor(emb_path)
    vocab_size, hidden = int(emb.shape[0]), int(emb.shape[1])

    # n_layers from decoder dirs
    dec_root = os.path.join(model_root, "decoder")
    layer_dirs = sorted([p for p in glob.glob(os.path.join(dec_root, "*")) if os.path.isdir(p)])
    n_layers = len(layer_dirs)

    # qkv rows from layer0 qkv
    qkv_path = find_unique(os.path.join(model_root, "decoder", "0", "pp_block", "attn", "qkv_proj", "*.safetensors"))
    qkv = load_first_tensor(qkv_path)
    qkv_rows = int(qkv.shape[0])
    # LLaMA GQA: qkv_rows = (hidden + 2 * kv_dim), kv_dim = num_kv_heads * head_dim, hidden = num_heads*head_dim
    # We can infer head_dim by trying divisors
    head_dim = None
    num_heads = None
    num_kv_heads = None

    # common head_dim candidates
    for hd in [64, 80, 96, 128, 160, 192, 256]:
        if hidden % hd != 0:
            continue
        nh = hidden // hd
        rem = qkv_rows - hidden
        if rem <= 0 or rem % 2 != 0:
            continue
        kv_dim = rem // 2
        if kv_dim % hd != 0:
            continue
        nkv = kv_dim // hd
        # GQA constraint: nkv <= nh and nh % nkv == 0 in most LLaMA
        if nkv <= nh and (nh % nkv == 0):
            head_dim = hd
            num_heads = nh
            num_kv_heads = nkv
            break

    if head_dim is None:
        # fallback: assume no GQA
        head_dim = hidden // 16
        num_heads = 16
        num_kv_heads = num_heads

    return dict(
        vocab_size=vocab_size,
        hidden=hidden,
        n_layers=n_layers,
        qkv_rows=qkv_rows,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )


def find_final_norm(model_root: str, hidden: int) -> str:
    """
    Robustly find final RMSNorm weight outside decoder.
    """
    preferred_dirs = [
        os.path.join(model_root, "norm"),
        os.path.join(model_root, "final_norm"),
        os.path.join(model_root, "final_layernorm"),
        os.path.join(model_root, "output_layernorm"),
        os.path.join(model_root, "final_rmsnorm"),
    ]
    for d in preferred_dirs:
        if os.path.isdir(d):
            files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
            for p in files:
                t = load_first_tensor(p)
                if t.ndim == 1 and int(t.shape[0]) == hidden:
                    return p

    # fallback: scan model_root (exclude decoder)
    files = sorted(glob.glob(os.path.join(model_root, "**", "*.safetensors"), recursive=True))
    cands = []
    for p in files:
        p_norm = p.replace("\\", "/")
        if "/decoder/" in p_norm:
            continue
        if not re.search(r"(norm|layernorm|rmsnorm)", p, re.I):
            continue
        t = load_first_tensor(p)
        if t.ndim == 1 and int(t.shape[0]) == hidden:
            cands.append(p)

    if not cands:
        raise FileNotFoundError("Could not find final norm weight (1D hidden,) outside decoder.")
    cands.sort(key=lambda x: ("/norm/" not in x.replace("\\", "/").lower(), len(x)))
    return cands[0]


def maybe_find_lm_head(model_root: str, hidden: int, vocab_size: int) -> str:
    """
    Try to find LM head weight. Some Nanotron configs tie embeddings; if not present, we tie later.
    """
    files = sorted(glob.glob(os.path.join(model_root, "**", "*.safetensors"), recursive=True))
    cands = []
    for p in files:
        if re.search(r"(lm_head|output|logits|classifier)", p, re.I):
            t = load_first_tensor(p)
            if t.ndim == 2 and int(t.shape[0]) == vocab_size and int(t.shape[1]) == hidden:
                cands.append(p)
    if cands:
        cands.sort(key=len)
        return cands[0]
    return ""


def load_layer_weights(model_root: str, layer: int) -> Dict[str, torch.Tensor]:
    """
    Load nanotron layer files for one decoder layer into a dict.
    """
    base = os.path.join(model_root, "decoder", str(layer), "pp_block")

    def one(pat: str) -> torch.Tensor:
        p = find_unique(os.path.join(base, pat))
        return load_first_tensor(p)

    # Attention
    qkv = one("attn/qkv_proj/*.safetensors")  # (hidden + 2*kv_dim, hidden)
    oproj = one("attn/o_proj/*.safetensors")  # (hidden, hidden)

    # MLP
    gate_up = one("mlp/gate_up_proj/*.safetensors")  # (2*intermediate, hidden)
    down = one("mlp/down_proj/*.safetensors")        # (hidden, intermediate)

    # Norms (names differ slightly; try common ones)
    # pre-attn norm
    pre_ln = None
    pre_pats = [
        "input_layernorm/*.safetensors",
        "pre_attention_layernorm/*.safetensors",
        "attention_layernorm/*.safetensors",
        "pre_attn_layernorm/*.safetensors",
    ]
    for pat in pre_pats:
        hits = glob.glob(os.path.join(base, pat))
        if hits:
            pre_ln = load_first_tensor(sorted(hits)[0])
            break
    if pre_ln is None:
        raise FileNotFoundError(f"Missing pre-attn norm for layer {layer} under {base}")

    # post-attn norm
    post_ln = None
    post_pats = [
        "post_attention_layernorm/*.safetensors",
        "post_attn_layernorm/*.safetensors",
        "mlp_layernorm/*.safetensors",
        "post_mlp_layernorm/*.safetensors",
    ]
    for pat in post_pats:
        hits = glob.glob(os.path.join(base, pat))
        if hits:
            post_ln = load_first_tensor(sorted(hits)[0])
            break
    if post_ln is None:
        raise FileNotFoundError(f"Missing post-attn norm for layer {layer} under {base}")

    return dict(
        qkv=qkv, oproj=oproj,
        gate_up=gate_up, down=down,
        pre_ln=pre_ln, post_ln=post_ln,
    )


def split_qkv(qkv: torch.Tensor, hidden: int, kv_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Nanotron qkv is stacked row-wise: [Q; K; V] where
      Q: (hidden, hidden), K: (kv_dim, hidden), V: (kv_dim, hidden)
    """
    assert qkv.shape[1] == hidden
    assert qkv.shape[0] == hidden + 2 * kv_dim
    q = qkv[:hidden, :]
    k = qkv[hidden:hidden + kv_dim, :]
    v = qkv[hidden + kv_dim:, :]
    return q, k, v


# -------------------------
# Main export
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Nanotron checkpoint folder, e.g. .../200")
    ap.add_argument("--out", required=True, help="Output HF folder")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer snapshot folder (contains tokenizer.json etc.)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    ckpt = must_exist(args.ckpt)
    out = args.out
    tok = must_exist(args.tokenizer)

    ensure_dir(out)

    # Model root
    model_root = os.path.join(ckpt, "model", "model")
    must_exist(model_root)

    dims = infer_model_dims(ckpt)
    vocab_size = dims["vocab_size"]
    hidden = dims["hidden"]
    n_layers = dims["n_layers"]
    head_dim = dims["head_dim"]
    num_heads = dims["num_heads"]
    num_kv_heads = dims["num_kv_heads"]
    kv_dim = num_kv_heads * head_dim

    print("== inferred ==")
    print(f"vocab_size: {vocab_size}")
    print(f"hidden: {hidden}")
    print(f"n_layers: {n_layers}")
    print(f"num_heads: {num_heads} num_kv_heads: {num_kv_heads} head_dim: {head_dim} kv_dim: {kv_dim}")

    # Determine intermediate size from MLP gate_up rows: (2*intermediate, hidden)
    gate0 = find_unique(os.path.join(model_root, "decoder", "0", "pp_block", "mlp", "gate_up_proj", "*.safetensors"))
    gate0_t = load_first_tensor(gate0)
    intermediate = int(gate0_t.shape[0] // 2)
    print("intermediate_size:", intermediate)

    # Embeddings
    emb_path = find_unique(os.path.join(model_root, "**", "token_embedding", "*.safetensors"))
    embed_tokens = load_first_tensor(emb_path)

    # Final norm
    norm_path = find_final_norm(model_root, hidden)
    final_norm = load_first_tensor(norm_path)
    print("final_norm picked:", norm_path, tuple(final_norm.shape))

    # lm_head (optional)
    lm_head_path = maybe_find_lm_head(model_root, hidden, vocab_size)
    lm_head = None
    if lm_head_path:
        lm_head = load_first_tensor(lm_head_path)
        print("lm_head found:", lm_head_path, tuple(lm_head.shape))
    else:
        print("lm_head not found: will tie to embeddings")

    # Build HF state dict
    sd: Dict[str, torch.Tensor] = {}

    # HF key names for LLaMA
    sd["model.embed_tokens.weight"] = embed_tokens
    sd["model.norm.weight"] = final_norm

    # Layers
    for i in range(n_layers):
        lw = load_layer_weights(model_root, i)

        q, k, v = split_qkv(lw["qkv"], hidden=hidden, kv_dim=kv_dim)

        # Attention projections
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = q
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = k
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = v
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = lw["oproj"]

        # Norms
        sd[f"model.layers.{i}.input_layernorm.weight"] = lw["pre_ln"]
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = lw["post_ln"]

        # MLP projections
        gate_up = lw["gate_up"]  # (2*intermediate, hidden) = [gate; up]
        gate = gate_up[:intermediate, :]
        up = gate_up[intermediate:, :]
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = gate
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = up
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = lw["down"]

        if (i + 1) % 4 == 0:
            print(f"loaded layers: {i+1}/{n_layers}")

    # LM head
    # LM head
    if lm_head is not None:
        sd["lm_head.weight"] = lm_head
    else:
    # avoid shared storage (safetensors forbids shared tensors)
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()


    # dtype cast
    if args.dtype == "bfloat16":
        target_dtype = torch.bfloat16
    elif args.dtype == "float16":
        target_dtype = torch.float16
    else:
        target_dtype = torch.float32

    for k in list(sd.keys()):
        t = sd[k]
        if t.dtype != target_dtype:
            sd[k] = t.to(dtype=target_dtype)

    # Write HF config.json
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": vocab_size,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": n_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "hidden_act": "silu",
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "tie_word_embeddings": True,
        "torch_dtype": str(target_dtype).replace("torch.", ""),
        "bos_token_id": 128000,
        "eos_token_id": 128001,
    }
    Path(os.path.join(out, "config.json")).write_text(json.dumps(hf_config, indent=2))

    # Copy/symlink tokenizer files
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(tok, name)
        if os.path.exists(src):
            symlink_or_copy(src, os.path.join(out, name))
    # Some tokenizers also need added_tokens.json
    extra = os.path.join(tok, "added_tokens.json")
    if os.path.exists(extra):
        symlink_or_copy(extra, os.path.join(out, "added_tokens.json"))

    # Save model weights
    out_model = os.path.join(out, "model.safetensors")
    print("saving:", out_model)
    save_file(sd, out_model)

    print("\n✅ Export complete:")
    print(out)
    print("You can now load with:")
    print(f"  AutoModelForCausalLM.from_pretrained('{out}', local_files_only=True)")
    print(f"  AutoTokenizer.from_pretrained('{out}', local_files_only=True)")


if __name__ == "__main__":
    main()
