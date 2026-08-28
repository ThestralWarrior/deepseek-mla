#!/usr/bin/env python3
# from-scratch MLA forward pass for DeepSeek-Coder-V2-Lite
import torch
import torch.nn.functional as F
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2ForCausalLM

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


def rms_norm(x, weight, eps=1e-6):
    in_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(in_dtype)


def build_freqs_cis(inv_freq, seq_len, attention_scaling=1.0):
    # one rotation angle per (position, freq-pair); polar() builds cos+i*sin directly
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq.to(torch.float32))
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    freqs_cis = freqs_cis * attention_scaling  # yarn amplitude correction
    return freqs_cis.unsqueeze(0)


def apply_rope(x, freqs_cis):
    # interleaved pairing: (0,1),(2,3),...
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)  # broadcast over heads
    return torch.view_as_real(x_c * freqs_cis).flatten(3).type_as(x)


def mla_forward(x, W, cfg, freqs_cis, attn_mask):
    # x already input_layernorm'd, shape (B, S, 2048)
    B, S, _ = x.shape
    H       = cfg["num_attention_heads"]   # 16
    d_nope  = cfg["qk_nope_head_dim"]      # 128
    d_rope  = cfg["qk_rope_head_dim"]      # 64
    d_qk    = d_nope + d_rope              # 192
    d_v     = cfg["v_head_dim"]            # 128
    d_c     = cfg["kv_lora_rank"]          # 512

    # query - q_lora_rank is null here so it is a single projection
    q = x @ W["q_proj"].T
    q = q.view(B, S, H, d_qk).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [d_nope, d_rope], dim=-1)

    # compress kv into one shared latent + one shared position lane
    ckv = x @ W["kv_a_proj"].T
    kv_nope, k_pe = torch.split(ckv, [d_c, d_rope], dim=-1)
    latent = rms_norm(kv_nope, W["kv_a_ln"], cfg["rms_norm_eps"])
    latent = latent.view(B, 1, S, d_c)
    k_pe = k_pe.view(B, 1, S, d_rope)

    # rope only touches the position lanes, never the latent itself
    q_pe = apply_rope(q_pe, freqs_cis)
    k_pe = apply_rope(k_pe, freqs_cis)

    # decompress the one latent back out to per-head k/v
    kv = latent @ W["kv_b_proj"].T
    kv = kv.view(B, S, H, d_nope + d_v).transpose(1, 2)
    k_nope, value_states = torch.split(kv, [d_nope, d_v], dim=-1)
    k_pe_all = k_pe.expand(-1, H, -1, -1)
    key_states = torch.cat([k_nope, k_pe_all], dim=-1)
    query_states = torch.cat([q_nope, q_pe], dim=-1)

    # ordinary scaled dot-product attention from here
    scores = (query_states @ key_states.transpose(2, 3)) * cfg["scaling"]
    scores = scores + attn_mask
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn = weights @ value_states

    attn = attn.transpose(1, 2).contiguous().reshape(B, S, H * d_v)
    return attn @ W["o_proj"].T


def main():
    torch.manual_seed(0)

    k = dict(REAL); k["num_hidden_layers"] = 2
    rope = k.pop("rope_scaling"); k.pop("rope_theta")
    k["rope_parameters"] = {**rope, "rope_type": rope["type"], "rope_theta": 10000}
    cfg_obj = DeepseekV2Config(**k)

    model = DeepseekV2ForCausalLM(cfg_obj).eval()
    model.config._attn_implementation = "eager"
    layer = model.model.layers[1]
    A = layer.self_attn

    cfg = {
        "num_attention_heads": cfg_obj.num_attention_heads,
        "qk_nope_head_dim": cfg_obj.qk_nope_head_dim,
        "qk_rope_head_dim": cfg_obj.qk_rope_head_dim,
        "v_head_dim": cfg_obj.v_head_dim,
        "kv_lora_rank": cfg_obj.kv_lora_rank,
        "rms_norm_eps": cfg_obj.rms_norm_eps,
        "scaling": A.scaling,
    }

    W = {
        "q_proj":    A.q_proj.weight.data,
        "kv_a_proj": A.kv_a_proj_with_mqa.weight.data,
        "kv_a_ln":   A.kv_a_layernorm.weight.data,
        "kv_b_proj": A.kv_b_proj.weight.data,
        "o_proj":    A.o_proj.weight.data,
    }

    B, S = 1, 6
    x_raw = torch.randn(B, S, cfg_obj.hidden_size)
    x = rms_norm(x_raw, layer.input_layernorm.weight.data, cfg_obj.rms_norm_eps)

    pos_ids = torch.arange(S).unsqueeze(0)
    rot = model.model.rotary_emb
    attention_scaling = rot.attention_scaling
    freqs_cis = build_freqs_cis(rot.inv_freq, S, attention_scaling)
    causal = torch.triu(torch.full((S, S), float("-inf")), diagonal=1).view(1, 1, S, S)

    ref_freqs = rot(x, pos_ids)
    assert torch.allclose(freqs_cis, ref_freqs, atol=1e-6)

    ref_norm = layer.input_layernorm(x_raw)
    assert torch.allclose(x, ref_norm, atol=1e-6)

    with torch.no_grad():
        mine = mla_forward(x, W, cfg, freqs_cis, causal)
        ref, _ = A(hidden_states=x,
                   attention_mask=causal,
                   position_embeddings=ref_freqs,
                   past_key_values=None)

    diff = (mine - ref).abs().max().item()
    print(f"prefill (6 tok):  max diff {diff:.2e}")
    assert torch.allclose(mine, ref, atol=1e-5)

    with torch.no_grad():
        x_all = torch.cat([x, rms_norm(torch.randn(B, 1, cfg_obj.hidden_size),
                                       layer.input_layernorm.weight.data,
                                       cfg_obj.rms_norm_eps)], dim=1)
        S2 = S + 1
        freqs2 = build_freqs_cis(rot.inv_freq, S2, attention_scaling)
        causal2 = torch.triu(torch.full((S2, S2), float("-inf")), diagonal=1).view(1, 1, S2, S2)
        mine_all = mla_forward(x_all, W, cfg, freqs2, causal2)
        ref_all, _ = A(hidden_states=x_all, attention_mask=causal2,
                       position_embeddings=rot(x_all, torch.arange(S2).unsqueeze(0)),
                       past_key_values=None)

    d2 = (mine_all[:, -1] - ref_all[:, -1]).abs().max().item()
    print(f"decode (tok 7):   max diff {d2:.2e}")
    assert torch.allclose(mine_all, ref_all, atol=1e-5)

    print("ok")


if __name__ == "__main__":
    main()
