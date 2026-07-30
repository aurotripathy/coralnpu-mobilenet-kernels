"""Generate the host golden reference for Gemma 3 270M bring-up.

Produces, for a fixed instruction-tuned prompt under greedy decoding:
  * the reference token IDs + decoded text (the token-exact gate target), and
  * step-0 (prefill) per-layer residual-stream activations plus layer-0
    submodule outputs (the per-layer shadow-compare signal).

The model runs in bfloat16 by default (--dtype), matching how Gemma 3 is
distributed and deployed; captured tensors are upcast to float32 before being
saved as numpy. Outputs go to --out (default ~/gemma3_ref), NOT into the repo
(logits alone are ~1 MB/step).
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-270m-it"
DEFAULT_PROMPT = "Give me a one-sentence fun fact about the moon."
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--out", default=os.path.expanduser("~/gemma3_ref"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPES[args.dtype])
    model.eval()
    print(f"model dtype: {args.dtype}")

    messages = [{"role": "user", "content": args.prompt}]
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    print(f"prompt tokens: {input_ids.shape[1]}")

    # --- Capture layer-0 submodule outputs on the prefill pass via hooks. ---
    caps = {}

    def grab(name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            caps[name] = t.detach()[0, -1].float().cpu().numpy()
        return hook

    l0 = model.model.layers[0]
    handles = [
        l0.input_layernorm.register_forward_hook(grab("l0.input_norm")),
        l0.self_attn.register_forward_hook(grab("l0.attn_out")),
        l0.post_attention_layernorm.register_forward_hook(grab("l0.post_attn_norm")),
        l0.pre_feedforward_layernorm.register_forward_hook(grab("l0.pre_ffw_norm")),
        l0.mlp.register_forward_hook(grab("l0.mlp_out")),
        l0.post_feedforward_layernorm.register_forward_hook(grab("l0.post_ffw_norm")),
    ]

    with torch.no_grad():
        prefill = model(input_ids, output_hidden_states=True, use_cache=True)
    for h in handles:
        h.remove()

    # Per-layer residual stream at the last prefill position: (L+1, hidden).
    hs = np.stack([h[0, -1].float().cpu().numpy()
                   for h in prefill.hidden_states])
    logits0 = prefill.logits[0, -1].float().cpu().numpy()
    argmax0 = int(logits0.argmax())
    print(f"step-0 argmax token: {argmax0} -> {tok.decode([argmax0])!r}")

    # --- Greedy decode for the token-exact gate. ---
    with torch.no_grad():
        gen = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
            num_beams=1)
    gen_ids = gen[0, input_ids.shape[1]:].cpu().numpy().astype(np.int64)
    text = tok.decode(gen_ids, skip_special_tokens=True)
    print(f"generated {len(gen_ids)} tokens:\n{text}")
    assert gen_ids[0] == argmax0, "greedy first token disagrees with argmax"

    np.savez(
        os.path.join(args.out, "reference.npz"),
        prompt=args.prompt,
        input_ids=input_ids[0].cpu().numpy().astype(np.int64),
        gen_ids=gen_ids,
        hidden_states=hs,
        logits0=logits0,
        argmax0=np.int64(argmax0),
        **{k.replace(".", "_"): v for k, v in caps.items()},
    )
    print(f"wrote {os.path.join(args.out, 'reference.npz')}")
    print(f"  hidden_states {hs.shape}, logits0 {logits0.shape}, "
          f"layer0 caps: {sorted(caps)}")


if __name__ == "__main__":
    main()
