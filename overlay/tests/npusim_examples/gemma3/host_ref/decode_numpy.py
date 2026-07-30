"""Phase 3a: numpy prototype of the full Gemma 3 270M decode step.

Validates the exact algorithm (sandwich RMSNorms, per-head QK-norm, dual-base
RoPE, MQA + causal/sliding attention, GeGLU, tied logits) and the int8 blob
layout against the host reference BEFORE writing bare-metal C++.

Two weight sources:
  --weights fp32 : dequantized fp32 from HF (isolates algorithm correctness)
  --weights int8 : dequantized from gemma3_weights.bin (adds quant error)

Gate: per-layer residual stream must match reference.hidden_states and the
step-0 argmax must be 818.
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "google/gemma-3-270m-it"


def rmsnorm(x, gamma, eps):
    # Gemma: x * rsqrt(mean(x^2)+eps) * (1 + gamma), computed in fp32.
    v = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(v + eps) * (1.0 + gamma)


def rope_cos_sin(seq, dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    pos = np.arange(seq, dtype=np.float64)[:, None] * inv[None, :]  # [seq, dim/2]
    cos = np.cos(pos)
    sin = np.sin(pos)
    return cos, sin


def apply_rope(x, cos, sin):
    # x: [seq, heads, dim]; HF Gemma uses rotate_half (first/second half split).
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return np.concatenate([o1, o2], axis=-1)


class Weights:
    """Provides fp32 weight matrices from either HF or the int8 blob."""

    def __init__(self, mode, out_dir):
        self.mode = mode
        self.cfg = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32).config
        if mode == "fp32":
            m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
            self.sd = {k: v.float().cpu().numpy() for k, v in m.state_dict().items()}
        else:
            self.man = json.load(open(os.path.join(out_dir, "gemma3_manifest.json")))
            self.blob = np.fromfile(os.path.join(out_dir, "gemma3_weights.bin"),
                                    dtype=np.uint8)

    def _from_blob(self, name):
        off, shape, dt = self.man["entries"][name]
        n = int(np.prod(shape))
        arr = self.blob[off: off + n * np.dtype(dt).itemsize].view(dt).reshape(shape)
        return arr

    def mat(self, layer, proj):
        """Return fp32 weight [out, in]."""
        if self.mode == "fp32":
            pre = "self_attn." if proj[0] in "qkvo" and proj.endswith("_proj") else "mlp."
            return self.sd[f"model.layers.{layer}.{pre}{proj}.weight"]
        q = self._from_blob(f"L{layer}.{proj}_q").astype(np.float32)
        s = self._from_blob(f"L{layer}.{proj}_scale")
        return q * s[:, None]

    def norm(self, layer, name):
        if self.mode == "fp32":
            key = (f"model.layers.{layer}.self_attn.{name}.weight"
                   if name.endswith("_norm") else
                   f"model.layers.{layer}.{name}.weight")
            return self.sd[key]
        return self._from_blob(f"L{layer}.{name}")

    def embed(self):
        if self.mode == "fp32":
            return self.sd["model.embed_tokens.weight"]
        q = self._from_blob("embed_q").astype(np.float32)
        s = self._from_blob("embed_scale")
        return q * s[:, None]

    def final_norm(self):
        return (self.sd["model.norm.weight"] if self.mode == "fp32"
                else self._from_blob("final_norm"))


def layer_forward(W, cfg, L, pos, hidden, kc, vc):
    """Single-position layer step with a KV cache (mirrors the C++ decoder).

    hidden: [hidden] residual for the current position (mutated in place).
    kc, vc: [layers, ctx, head_dim] caches; this position's K/V are written at
    [L, pos].
    """
    H, HD = cfg.hidden_size, cfg.head_dim
    NH = cfg.num_attention_heads
    eps = cfg.rms_norm_eps
    attn_scale = cfg.query_pre_attn_scalar ** -0.5
    sliding = cfg.layer_types[L] == "sliding_attention"
    theta = 10000.0 if sliding else 1000000.0
    cos, sin = rope_cos_sin(pos + 1, HD, theta)

    x = rmsnorm(hidden, W.norm(L, "input_layernorm"), eps)
    q = (x @ W.mat(L, "q_proj").T).reshape(NH, HD)
    k = (x @ W.mat(L, "k_proj").T).reshape(HD)
    v = (x @ W.mat(L, "v_proj").T).reshape(HD)
    q = rmsnorm(q, W.norm(L, "q_norm"), eps)
    k = rmsnorm(k, W.norm(L, "k_norm"), eps)
    q = apply_rope(q[None], cos[pos:pos + 1], sin[pos:pos + 1])[0]
    k = apply_rope(k[None, None], cos[pos:pos + 1], sin[pos:pos + 1])[0, 0]
    kc[L, pos] = k
    vc[L, pos] = v

    lo = 0
    if sliding and pos >= cfg.sliding_window:
        lo = pos - cfg.sliding_window + 1
    out = np.zeros((NH, HD), np.float32)
    for h in range(NH):
        sc = (kc[L, lo:pos + 1] @ q[h]) * attn_scale
        sc = sc - sc.max()
        p = np.exp(sc)
        p = p / p.sum()
        out[h] = p @ vc[L, lo:pos + 1]
    attn = out.reshape(NH * HD) @ W.mat(L, "o_proj").T
    attn = rmsnorm(attn, W.norm(L, "post_attention_layernorm"), eps)
    hidden = hidden + attn

    x = rmsnorm(hidden, W.norm(L, "pre_feedforward_layernorm"), eps)
    gate = x @ W.mat(L, "gate_proj").T
    up = x @ W.mat(L, "up_proj").T
    gelu = 0.5 * gate * (1.0 + np.tanh(
        0.7978845608 * (gate + 0.044715 * gate ** 3)))
    m = (gelu * up) @ W.mat(L, "down_proj").T
    m = rmsnorm(m, W.norm(L, "post_feedforward_layernorm"), eps)
    return hidden + m


def run_generate(W, cfg, ref, n_new):
    """Incremental greedy decode; compares to reference gen_ids."""
    H = cfg.hidden_size
    HD = cfg.head_dim
    NL = cfg.num_hidden_layers
    ids = ref["input_ids"].astype(np.int64)
    S = len(ids)
    ctx = S + n_new
    kc = np.zeros((NL, ctx, HD), np.float32)
    vc = np.zeros((NL, ctx, HD), np.float32)
    emb = W.embed()
    final_gamma = W.final_norm()

    def logits_argmax(hidden):
        hf = rmsnorm(hidden, final_gamma, cfg.rms_norm_eps)
        return int((hf @ emb.T).argmax())

    tok = None
    for pos in range(S):
        hidden = emb[ids[pos]].astype(np.float32) * (H ** 0.5)
        for L in range(NL):
            hidden = layer_forward(W, cfg, L, pos, hidden, kc, vc)
        if pos == S - 1:
            tok = logits_argmax(hidden)

    gen = [tok]
    for step in range(1, n_new):
        pos = S - 1 + step
        hidden = emb[tok].astype(np.float32) * (H ** 0.5)
        for L in range(NL):
            hidden = layer_forward(W, cfg, L, pos, hidden, kc, vc)
        tok = logits_argmax(hidden)
        gen.append(tok)

    ref_gen = ref["gen_ids"].astype(np.int64)[:n_new]
    gen = np.array(gen[:n_new], np.int64)
    match = int((gen == ref_gen).sum())
    print(f"generated : {gen.tolist()}")
    print(f"reference : {ref_gen.tolist()}")
    print(f"token match {match}/{len(ref_gen)}")
    print("GEN PASS" if match == len(ref_gen) else "GEN PARTIAL/FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--gen", type=int, default=0,
                    help="if >0, run incremental greedy decode for N tokens")
    ap.add_argument("--out", default=os.path.expanduser("~/gemma3_ref"))
    args = ap.parse_args()

    W = Weights(args.weights, args.out)
    cfg = W.cfg
    H, HD = cfg.hidden_size, cfg.head_dim
    NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads
    eps = cfg.rms_norm_eps
    attn_scale = cfg.query_pre_attn_scalar ** -0.5
    layer_types = cfg.layer_types

    ref = np.load(os.path.join(args.out, "reference.npz"))
    if args.gen > 0:
        run_generate(W, cfg, ref, args.gen)
        return
    ids = ref["input_ids"].astype(np.int64)
    seq = len(ids)
    ref_hs = ref["hidden_states"]  # [L+1, hidden] at last position

    emb = W.embed()
    h = emb[ids].astype(np.float32) * (H ** 0.5)   # [seq, hidden]

    print(f"weights={args.weights} seq={seq}")
    print(f"  L00 embed   last-pos maxdiff {np.abs(h[-1]-ref_hs[0]).max():.3e}")

    for i in range(cfg.num_hidden_layers):
        theta = (10000.0 if layer_types[i] == "sliding_attention" else 1000000.0)
        cos, sin = rope_cos_sin(seq, HD, theta)

        res = h
        x = rmsnorm(h, W.norm(i, "input_layernorm"), eps)

        q = (x @ W.mat(i, "q_proj").T).reshape(seq, NH, HD)
        k = (x @ W.mat(i, "k_proj").T).reshape(seq, NKV, HD)
        v = (x @ W.mat(i, "v_proj").T).reshape(seq, NKV, HD)
        q = rmsnorm(q, W.norm(i, "q_norm"), eps)
        k = rmsnorm(k, W.norm(i, "k_norm"), eps)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # MQA: broadcast the single KV head across all query heads.
        group = NH // NKV
        out = np.zeros((seq, NH, HD), np.float32)
        causal = np.triu(np.full((seq, seq), -np.inf, np.float32), 1)
        if layer_types[i] == "sliding_attention":
            sw = cfg.sliding_window
            band = np.tril(np.full((seq, seq), -np.inf, np.float32), -sw)
            causal = causal + band
        for hh in range(NH):
            kv = hh // group
            sc = (q[:, hh, :] @ k[:, kv, :].T) * attn_scale + causal
            sc = sc - sc.max(axis=-1, keepdims=True)
            p = np.exp(sc)
            p = p / p.sum(axis=-1, keepdims=True)
            out[:, hh, :] = p @ v[:, kv, :]
        attn = out.reshape(seq, NH * HD) @ W.mat(i, "o_proj").T
        attn = rmsnorm(attn, W.norm(i, "post_attention_layernorm"), eps)
        h = res + attn

        res = h
        x = rmsnorm(h, W.norm(i, "pre_feedforward_layernorm"), eps)
        gate = x @ W.mat(i, "gate_proj").T
        up = x @ W.mat(i, "up_proj").T
        gelu = 0.5 * gate * (1.0 + np.tanh(
            0.7978845608 * (gate + 0.044715 * gate ** 3)))
        m = (gelu * up) @ W.mat(i, "down_proj").T
        m = rmsnorm(m, W.norm(i, "post_feedforward_layernorm"), eps)
        h = res + m

        md = np.abs(h[-1] - ref_hs[i + 1]).max()
        print(f"  L{i:02d} residual last-pos maxdiff {md:.3e}")

    hf = rmsnorm(h, W.final_norm(), eps)
    logits = hf[-1] @ W.embed().T
    am = int(logits.argmax())
    print(f"argmax {am} (ref {int(ref['argmax0'])})  "
          f"logits maxdiff {np.abs(logits - ref['logits0']).max():.3e}")
    print("PASS" if am == int(ref["argmax0"]) else "FAIL")


if __name__ == "__main__":
    main()
