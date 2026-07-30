"""Phase 2: pack Gemma 3 270M into an int8 weight blob for the simulator.

Layout (all sections 4-byte aligned; blob is written verbatim to the DDR
weight base at runtime):

  matmul weights : int8, HF-native row-major [out, in], per-output-channel
                   symmetric scale (fp32 [out]). This matches the existing int8
                   reduction-MatMul layout (rhs [cols, inner] row-major) so no
                   transpose is needed, and scale[out] lines up with requant.
  norm gammas    : fp32, stored as (1 + gamma) already folded? NO - stored raw;
                   the kernel applies x*rsqrt(...)*(1+gamma) like the tree's
                   RmsNormF. (Gemma stores gamma; +1 happens in the kernel.)
  embedding      : int8 [vocab, 640] + per-row scale, serves BOTH token lookup
                   and the tied logits head.

Emits:
  <out>/gemma3_weights.bin  - the packed blob (huge, scratch, not committed)
  gemma3_layout.h           - dims + byte offsets (small, committed)
  <out>/gemma3_manifest.json- offsets for the Python driver
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "google/gemma-3-270m-it"


def quant_per_row(w):
    """Per-row symmetric int8. w: [out, in] fp32 -> (int8 [out,in], scale[out])."""
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.round(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


class Blob:
    def __init__(self):
        self.parts = []
        self.off = 0
        self.entries = {}

    def _pad4(self):
        pad = (-self.off) % 4
        if pad:
            self.parts.append(np.zeros(pad, np.uint8))
            self.off += pad

    def add(self, name, arr):
        self._pad4()
        raw = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        self.entries[name] = (self.off, arr.shape, str(arr.dtype))
        self.parts.append(raw)
        self.off += raw.size
        return self.off

    def bytes(self):
        return np.concatenate(self.parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/gemma3_ref"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    sd = model.state_dict()

    def w(name):
        return sd[name].float().cpu().numpy()

    blob = Blob()
    n_layers = model.config.num_hidden_layers

    # --- Global: embedding (tied lm_head) + final norm ---
    emb = w("model.embed_tokens.weight")           # [vocab, hidden]
    emb_q, emb_s = quant_per_row(emb)
    blob.add("embed_q", emb_q)
    blob.add("embed_scale", emb_s)
    blob.add("final_norm", w("model.norm.weight").astype(np.float32))

    layer_off0 = None
    layer_stride = None
    mm = ["q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj"]
    norms = ["input_layernorm", "post_attention_layernorm",
             "pre_feedforward_layernorm", "post_feedforward_layernorm"]

    for i in range(n_layers):
        p = f"model.layers.{i}."
        start = blob.off + ((-blob.off) % 4)
        if i == 0:
            layer_off0 = start
        for nm in norms:
            blob.add(f"L{i}.{nm}", w(p + nm + ".weight").astype(np.float32))
        blob.add(f"L{i}.q_norm", w(p + "self_attn.q_norm.weight").astype(np.float32))
        blob.add(f"L{i}.k_norm", w(p + "self_attn.k_norm.weight").astype(np.float32))
        for nm in mm:
            src = p + ("self_attn." if nm.endswith("_proj") and nm[0] in "qkvo"
                       else "mlp.") + nm + ".weight"
            q, s = quant_per_row(w(src))
            blob.add(f"L{i}.{nm}_q", q)
            blob.add(f"L{i}.{nm}_scale", s)
        end = blob.off
        if i == 0:
            layer_stride = end - start

    data = blob.bytes()
    bin_path = os.path.join(args.out, "gemma3_weights.bin")
    data.tofile(bin_path)
    print(f"wrote {bin_path}: {data.size/1e6:.1f} MB, layer_stride "
          f"{layer_stride} B, {n_layers} layers")

    manifest = {
        "model_id": MODEL_ID,
        "n_layers": n_layers,
        "hidden": model.config.hidden_size,
        "intermediate": model.config.intermediate_size,
        "head_dim": model.config.head_dim,
        "n_heads": model.config.num_attention_heads,
        "n_kv_heads": model.config.num_key_value_heads,
        "vocab": model.config.vocab_size,
        "rms_eps": model.config.rms_norm_eps,
        "sliding_window": model.config.sliding_window,
        "layer_off0": layer_off0,
        "layer_stride": layer_stride,
        "total_bytes": int(data.size),
        "entries": blob.entries,
    }
    with open(os.path.join(args.out, "gemma3_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    emit_header(manifest, blob, layer_off0, layer_stride)

    # --- Sanity: dequant round-trip error on layer-0 q_proj against the
    #     reference input_norm activation, if available. ---
    ref = os.path.join(args.out, "reference.npz")
    if os.path.exists(ref):
        r = np.load(ref)
        x = r["l0_input_norm"].astype(np.float32)      # [hidden]
        wq = w("model.layers.0.self_attn.q_proj.weight")
        q, s = quant_per_row(wq)
        deq = q.astype(np.float32) * s[:, None]
        y_ref = wq @ x
        y_q = deq @ x
        rel = np.abs(y_q - y_ref).max() / (np.abs(y_ref).max() + 1e-9)
        print(f"q_proj dequant round-trip max rel err: {rel:.2e}")


def emit_header(m, blob, layer_off0, layer_stride):
    e = blob.entries
    L0 = {k[3:]: v[0] - layer_off0 for k, v in e.items() if k.startswith("L0.")}
    hdr = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "gemma3_layout.h")
    lines = [
        "// Generated by host_ref/pack_weights.py - do not edit by hand.",
        "// Byte offsets into the Gemma 3 270M int8 weight blob (DDR base).",
        "#ifndef GEMMA3_LAYOUT_H_",
        "#define GEMMA3_LAYOUT_H_",
        "",
        f"#define GEMMA3_N_LAYERS {m['n_layers']}",
        f"#define GEMMA3_HIDDEN {m['hidden']}",
        f"#define GEMMA3_INTERMEDIATE {m['intermediate']}",
        f"#define GEMMA3_HEAD_DIM {m['head_dim']}",
        f"#define GEMMA3_N_HEADS {m['n_heads']}",
        f"#define GEMMA3_N_KV_HEADS {m['n_kv_heads']}",
        f"#define GEMMA3_VOCAB {m['vocab']}",
        f"#define GEMMA3_SLIDING_WINDOW {m['sliding_window']}",
        f"#define GEMMA3_RMS_EPS {m['rms_eps']}f",
        "",
        f"#define GEMMA3_EMBED_Q_OFF {e['embed_q'][0]}",
        f"#define GEMMA3_EMBED_SCALE_OFF {e['embed_scale'][0]}",
        f"#define GEMMA3_FINAL_NORM_OFF {e['final_norm'][0]}",
        f"#define GEMMA3_LAYER0_OFF {layer_off0}",
        f"#define GEMMA3_LAYER_STRIDE {layer_stride}",
        f"#define GEMMA3_TOTAL_BYTES {m['total_bytes']}",
        "",
        "// Per-layer sub-offsets (add GEMMA3_LAYER0_OFF + i*GEMMA3_LAYER_STRIDE):",
    ]
    for k in sorted(L0):
        macro = "GEMMA3_L_" + k.upper().replace(".", "_")
        lines.append(f"#define {macro} {L0[k]}")
    lines += ["", "#endif  // GEMMA3_LAYOUT_H_", ""]
    with open(hdr, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {hdr}")


if __name__ == "__main__":
    main()
