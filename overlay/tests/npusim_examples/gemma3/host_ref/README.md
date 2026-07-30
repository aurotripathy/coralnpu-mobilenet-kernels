# Gemma 3 270M - host reference (Phase 1)

Host-side tooling that produces the golden reference the on-simulator decoder is
verified against. Runs on the host CPU in a venv, not on the simulator.

## Setup

```bash
python3 -m venv ~/gemmaenv
~/gemmaenv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/gemmaenv/bin/pip install transformers safetensors accelerate huggingface_hub sentencepiece numpy
```

`google/gemma-3-270m-it` is a **gated** model: accept the license at
https://huggingface.co/google/gemma-3-270m-it, then `~/gemmaenv/bin/hf auth login`
with a read token.

## Scripts

* `inspect_gemma3.py` - prints config + module tree (used to confirm the
  architecture and the exact submodule names hooked below).
* `gen_reference.py` - fixed-prompt greedy decode in bfloat16 (`--dtype`,
  default `bf16`, matching how Gemma 3 is deployed); writes `reference.npz` to
  `--out` (default `~/gemma3_ref`, kept out of the repo):
  * `input_ids`, `gen_ids` (+ decoded text) - the token-exact gate target.
  * `hidden_states` `(L+1, hidden)` - residual stream at the last prefill
    position entering each layer (per-layer shadow-compare signal).
  * `logits0` `(vocab,)`, `argmax0` - the step-0 logits and predicted token.
  * `l0_*` - layer-0 submodule outputs (input_norm, attn_out, post_attn_norm,
    pre_ffw_norm, mlp_out, post_ffw_norm) for fine-grained bring-up.

```bash
~/gemmaenv/bin/python gen_reference.py
```

## Reference architecture facts confirmed from the checkpoint

* 18 layers, hidden 640, intermediate 2048, head_dim 256.
* Attention: 4 query heads, 1 KV head (MQA); q_proj 1024x640, k/v_proj 256x640,
  o_proj 640x1024; per-head QK-RMSNorm (q_norm/k_norm are (256,)).
* Four RMSNorms per layer (sandwich): input, post_attention, pre_feedforward,
  post_feedforward; plus a final model norm. eps 1e-6.
* GeGLU MLP: gate/up 2048x640, down 640x2048, gelu_pytorch_tanh.
* Tied embeddings (262144x640) also serve as the logits head.
* RoPE dual base: 10000 (sliding) / 1000000 (full); sliding window 512;
  layer pattern (5 sliding + 1 full) x 3. query_pre_attn_scalar 256.
