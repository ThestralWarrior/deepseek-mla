#!/usr/bin/env python3
# prefill N -> compressed cache -> decode 1 token, fp32 vs bf16
import argparse
import copy
import json
from pathlib import Path

import torch
from transformers import DynamicCache
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2RMSNorm,
    DeepseekV2RotaryEmbedding,
    apply_rotary_emb,
)

REAL = {
    "attention_bias": False, "attention_dropout": 0.0, "first_k_dense_replace": 1,
    "hidden_act": "silu", "hidden_size": 2048, "intermediate_size": 10944,
    "kv_lora_rank": 512, "max_position_embeddings": 163840,
    "moe_intermediate_size": 1408, "n_group": 1, "n_routed_experts": 64,
    "n_shared_experts": 2, "norm_topk_prob": False, "num_attention_heads": 16,
    "num_experts_per_tok": 6, "num_hidden_layers": 27, "num_key_value_heads": 16,
    "q_lora_rank": None, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-06, "rope_theta": 10000, "routed_scaling_factor": 1.0,
    "tie_word_embeddings": False, "topk_group": 1, "topk_method": "greedy",
    "v_head_dim": 128, "vocab_size": 102400,
    "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 40,
                     "mscale": 0.707, "mscale_all_dim": 0.707,
                     "original_max_position_embeddings": 4096, "type": "yarn"},
}

PAGE_SIZE = 128
CACHE_LENS = [16, 128, 1000, 1024]   # 1000 is not page-aligned


def build_config():
    k = dict(REAL)
    rope = k.pop("rope_scaling"); k.pop("rope_theta")
    k["rope_parameters"] = {**rope, "rope_type": rope["type"], "rope_theta": 10000}
    return DeepseekV2Config(**k)


def build_modules(cfg, dtype, seed=0):
    torch.manual_seed(seed)
    cfg = copy.deepcopy(cfg)
    cfg._attn_implementation = "eager"
    attn = DeepseekV2Attention(cfg, layer_idx=0).eval().to(dtype)
    norm = DeepseekV2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).eval().to(dtype)
    with torch.no_grad():  # inits to ones, which hides a dropped multiply
        norm.weight.copy_(torch.randn(cfg.hidden_size).mul_(0.1).add_(1.0).to(dtype))
    return attn, norm, DeepseekV2RotaryEmbedding(cfg)


@torch.no_grad()
def prefill_decode(attn, norm, rot, cfg, n_prefill, dtype, seed=1234):
    torch.manual_seed(seed)
    B = 1

    # bf16 round-trip first so both dtype runs get identical inputs
    x_all = torch.randn(B, n_prefill + 1, cfg.hidden_size)
    x_all = x_all.to(torch.bfloat16).to(dtype)

    cache = DynamicCache(config=cfg)

    x_pre = x_all[:, :n_prefill]
    h_pre = norm(x_pre)
    freqs_pre = rot(h_pre, torch.arange(n_prefill).unsqueeze(0))
    causal = torch.triu(
        torch.full((n_prefill, n_prefill), torch.finfo(torch.float32).min),
        diagonal=1,
    ).view(1, 1, n_prefill, n_prefill).to(dtype)
    attn(hidden_states=h_pre, attention_mask=causal,
         position_embeddings=freqs_pre, past_key_values=cache)

    latent_cache = cache.layers[0].keys.clone()   # [1,1,N,512]
    kpe_cache = cache.layers[0].values.clone()    # [1,1,N,64] post-RoPE

    x_dec = x_all[:, n_prefill:n_prefill + 1]
    h_dec = norm(x_dec)
    freqs_dec = rot(h_dec, torch.tensor([[n_prefill]]))

    # split q by hand only to capture both halves
    q = attn.q_proj(h_dec).view(B, 1, -1, attn.qk_head_dim).transpose(1, 2)
    q_nope, q_pe_raw = torch.split(
        q, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
    q_pe, _ = apply_rotary_emb(q_pe_raw, q_pe_raw, freqs_dec)

    # one token vs all of history, so no mask
    attn_out, _ = attn(hidden_states=h_dec, attention_mask=None,
                       position_embeddings=freqs_dec, past_key_values=cache)
    layer_out = x_dec + attn_out

    return {
        "x_prefill": x_pre,
        "x_decode": x_dec,
        "h_decode": h_dec,
        "latent_cache_prefill": latent_cache,
        "kpe_cache_prefill": kpe_cache,
        "latent_cache_full": cache.layers[0].keys.clone(),
        "kpe_cache_full": cache.layers[0].values.clone(),
        "q_nope": q_nope,
        "q_pe": q_pe,
        "query_states": torch.cat([q_nope, q_pe], dim=-1),
        "attn_out": attn_out,
        "layer_out": layer_out,
        "position": torch.tensor([n_prefill]),
        "scaling": torch.tensor(attn.scaling),
    }


@torch.no_grad()
def full_recompute_last_row(attn, norm, rot, cfg, n_prefill, dtype, seed=1234):
    # same token, no cache at all. catches bad position ids / cache ordering.
    torch.manual_seed(seed)
    x_all = torch.randn(1, n_prefill + 1, cfg.hidden_size)
    x_all = x_all.to(torch.bfloat16).to(dtype)

    h = norm(x_all)
    S = n_prefill + 1
    freqs = rot(h, torch.arange(S).unsqueeze(0))
    causal = torch.triu(
        torch.full((S, S), torch.finfo(torch.float32).min), diagonal=1
    ).view(1, 1, S, S).to(dtype)
    out, _ = attn(hidden_states=h, attention_mask=causal,
                  position_embeddings=freqs, past_key_values=None)
    return out[:, -1:], x_all[:, -1:] + out[:, -1:]


def compare(a, b):
    a32, b32 = a.float(), b.float()
    diff = (a32 - b32).abs()
    rms = a32.pow(2).mean().sqrt().item()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        # elemwise rel blows up near zero, /rms is the usable one
        "max_rel_elemwise": (diff / a32.abs().clamp_min(1e-12)).max().item(),
        "max_abs_over_rms": diff.max().item() / max(rms, 1e-12),
        "ref_absmax": a32.abs().max().item(),
        "rms": rms,
    }


def required_tolerance(ref, got, rtol):
    # solve |got-ref| <= atol + rtol*|ref| for atol
    ref32, got32 = ref.float(), got.float()
    slack = (got32 - ref32).abs() - rtol * ref32.abs()
    return max(slack.max().item(), 0.0)


TENSORS_OF_INTEREST = [
    ("latent_cache_full", "512-d latent cache"),
    ("kpe_cache_full", "64-d RoPE cache"),
    ("q_nope", "query (no-pos part)"),
    ("q_pe", "query (RoPE part)"),
    ("attn_out", "attention output"),
    ("layer_out", "layer output (attn + residual)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reference_data")
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cfg = build_config()
    outdir = Path(args.out)
    if not args.no_save:
        outdir.mkdir(parents=True, exist_ok=True)

    print("transformers", __import__("transformers").__version__)
    print(f"heads={cfg.num_attention_heads} kv_lora_rank={cfg.kv_lora_rank} "
          f"qk_rope={cfg.qk_rope_head_dim} v_head={cfg.v_head_dim}")

    # build in bf16, widen one copy. both sides then hold the same values.
    attn32, norm32, rot32 = build_modules(cfg, torch.bfloat16)
    attn32 = attn32.to(torch.float32)
    norm32 = norm32.to(torch.float32)
    attn16, norm16, rot16 = build_modules(cfg, torch.bfloat16)
    print(f"attention scaling = {attn32.scaling:.10f}  "
          f"(1/sqrt(192) would be {192 ** -0.5:.10f})")
    print()

    summary = {}
    worst = {name: 0.0 for name, _ in TENSORS_OF_INTEREST}

    for n in CACHE_LENS:
        aligned = "page-aligned" if n % PAGE_SIZE == 0 else "NOT page-aligned"
        print("=" * 74)
        print(f"cache length {n}  ({aligned}, {n / PAGE_SIZE:.2f} pages of {PAGE_SIZE})")
        print("=" * 74)

        ref = prefill_decode(attn32, norm32, rot32, cfg, n, torch.float32)
        got = prefill_decode(attn16, norm16, rot16, cfg, n, torch.bfloat16)

        rc_attn, rc_layer = full_recompute_last_row(
            attn32, norm32, rot32, cfg, n, torch.float32)
        d_attn = (ref["attn_out"] - rc_attn).abs().max().item()
        d_layer = (ref["layer_out"] - rc_layer).abs().max().item()
        print(f"cache-vs-recompute (fp32): attn {d_attn:.2e}  layer {d_layer:.2e}")
        assert d_attn < 1e-4, f"cached decode disagrees with full recompute: {d_attn}"

        assert ref["latent_cache_full"].shape == (1, 1, n + 1, cfg.kv_lora_rank)
        assert ref["kpe_cache_full"].shape == (1, 1, n + 1, cfg.qk_rope_head_dim)
        assert ref["q_nope"].shape == (1, cfg.num_attention_heads, 1, cfg.qk_nope_head_dim)
        assert ref["q_pe"].shape == (1, cfg.num_attention_heads, 1, cfg.qk_rope_head_dim)

        rows = {}
        print(f"{'tensor':<30} {'shape':<16} {'max|Δ|':>10} {'Δ/rms':>9} {'atol@rtol':>10}")
        print("-" * 78)
        for key, label in TENSORS_OF_INTEREST:
            stats = compare(ref[key], got[key])
            need = required_tolerance(ref[key], got[key], args.rtol)
            worst[key] = max(worst[key], need)
            shape = "x".join(str(d) for d in ref[key].shape)
            print(f"{label:<30} {shape:<16} {stats['max_abs']:>10.3e} "
                  f"{stats['max_abs_over_rms']:>9.2e} {need:>10.3e}")
            rows[key] = {**stats, "required_atol": need,
                         "shape": list(ref[key].shape)}
        summary[str(n)] = {"page_aligned": n % PAGE_SIZE == 0, "tensors": rows}
        print()

        if not args.no_save:
            torch.save(got, outdir / f"decode_bf16_n{n}.pt")
            torch.save(ref, outdir / f"decode_fp32_n{n}.pt")

    print("=" * 74)
    print(f"BF16 ERROR FLOOR  (worst across all cache lengths, rtol={args.rtol})")
    print("=" * 74)
    for key, label in TENSORS_OF_INTEREST:
        print(f"  {label:<30} atol >= {worst[key]:.3e}")
    head = max(worst["attn_out"], worst["layer_out"])
    print()
    print(f"  use for the mla_layer output: rtol={args.rtol}, atol={head:.1e} "
          f"(measured {head:.3e})")

    if not args.no_save:
        with open(outdir / "tolerance.json", "w") as f:
            json.dump({"rtol": args.rtol, "page_size": PAGE_SIZE,
                       "cache_lens": CACHE_LENS,
                       "worst_case_atol": worst, "per_length": summary}, f, indent=2)
        print(f"\nwrote tensors + tolerance.json to {outdir}/")


if __name__ == "__main__":
    main()
