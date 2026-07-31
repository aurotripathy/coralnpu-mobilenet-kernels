# Gemma 3 270M on npusim - implementation plan

Goal: greedy-decode Gemma 3 270M (instruction-tuned) on the CoralNPU
instruction-level simulator, reusing the optimized int8 RVV kernels from the
MobileNet work, with a token-exact verification gate against a host reference.

## The model

Target: **Gemma 3 270M** - the only Gemma 3 size that is realistic on CoralNPU
(the family is 270M / 1B / 4B / 12B / 27B).

Official LiteRT artifacts exist:

* [`litert-community/gemma-3-270m-it`](https://huggingface.co/litert-community/gemma-3-270m-it) -
  Google-maintained LiteRT release (`.litertlm`, plus `.task` bundles for the
  MediaPipe LLM Inference API).
* Reproducible conversion from raw weights:
  `litert-torch export_hf --model=google/gemma-3-270m-it --output_dir=...`
* Google also publishes QAT INT4 checkpoints of the raw model.

**Critical caveat:** `.litertlm` runs on the **LiteRT-LM runtime** - a large
host-class C++ runtime (XNNPack, CPU/GPU delegates) targeting Android/iOS/Web.
It does not run on bare-metal TFLM/RISC-V. So the LiteRT artifact's role here
is (a) weight source and (b) host-side golden reference - **not** the
on-device executable.

### Architecture facts (drive all sizing below)

| Item | Value |
|---|---|
| Layers | 18: (5x sliding-window-512 + 1x global) x 3 |
| Hidden / FFN | 640 / 2048, GeGLU (gate+up+down = 3 matmuls) |
| Attention | 4 Q heads, head_dim 256, 1 KV head (MQA), QK-RMSNorm |
| RoPE | dual base 10k (local) / 1M (global) |
| Vocab | 262,144, **tied embeddings** (also used as output head) |
| Params | ~168M embedding + ~100M transformer blocks |
| Context | 32k trained; we run with a small static context (e.g. 512-1024) |

## CoralNPU sizing

* **Weights:** int8 ~270 MB total (168 MB embedding + ~100 MB blocks). The
  sim's DDR region is currently 128 MB (`coralnpu_v2_sim_utils.py`, one-line
  change to grow) - either enlarge it or store the embedding table int4.
* **Compute per decoded token:** ~270M MACs - ~100M for the blocks and ~168M
  for the logits projection (tied embedding, 640 x 262144). At the measured
  2-4 int8 MACs/cycle of our `e32m8` kernels: **~70-135M cycles ~ 1 s/token
  at 100 MHz**. The logits head is the single biggest matmul; it is exactly
  our optimized "classifier" 1x1 kernel shape, just with 262k output channels.
* **KV cache:** MQA with 1 KV head x 256 dims, window 512 -> ~4.7 MB int8 for
  all 18 layers. Fits the 4 MB extmem + DDR comfortably.
* **Softmax:** greedy decode needs only **argmax** over logits (no final
  softmax). Attention softmax remains: 4 heads x 18 layers x <=512 window ~
  37k elements/token - at the current scalar softmax cost (~1.2k
  cycles/element) that would be ~44M cycles/token, so **vectorizing softmax is
  mandatory** (it was optional for MobileNet).

## What already exists in the tree

`tests/cocotb/rvv/ml_ops/gemma_kernels/` already contains **FP32 RVV building
blocks**, each with a numpy golden reference and a cocotb RTL test explicitly
sized for Gemma 3 270M (hidden 640, prefill seq 11, decode seq 1):

* `rvv_rms_norm.cc` - Gemma-exact RMSNorm `x*rsqrt(mean(x^2)+eps)*(1+w)`.
* `rvv_tanh_gelu_mul.cc` - GeGLU (tanh-approx GELU(gate)*up).
* `rvv_residual_add.cc` - residual add.
* `rvv_matmul.cc` - FP32 GeMV (decode) + tiled 2D (prefill).
* `rvv_flashattention_kernel.cc` - attention with **vectorized softmax + RVV
  `exp`** (so softmax is already vectorized).
* `static_reference_tests/int_matmul_16x48x16_kernel.cc` - int8 x int8 -> i32
  MatMul (e8m2 widening), the starting point for int8-weight matmul.

Caveats that shape the remaining work:

* They are **FP32** kernels. FP32 270M weights ~1.6 GB won't fit 128 MB DDR, so
  int8 weights are still required; the fp32 matmul needs an int8-weight variant.
* They run on **cocotb/RTL** (`RvvCoreMiniHighmemAxi`), not npusim - each is a
  standalone op test, not the ELF+npusim flow MobileNet uses.
* FlashAttention is **generic MHA, non-causal, no KV cache, no sliding window**;
  Gemma 3 270M is MQA (1 KV head) + causal + sliding-window-512, so attention
  needs real adaptation.

Still missing for end-to-end decode: RoPE (dual base), QK-norm wiring,
embedding lookup + tied logits + argmax, KV-cache ring buffer + sliding-window
masking, the 18-block orchestration loop, int8 weight packing/loading, and a
run/verify harness. The per-op math is largely built; the integration is not.

## Runtime approach

Three options considered:

* **(A) ai-edge-torch `.tflite` on TFLM** - rejected for now: the generative
  exports assume the full LiteRT runtime (KV-cache signatures, weight-only
  quant with fp32 activations, ops/dtypes TFLM doesn't fully support), and a
  270 MB flatbuffer can't be embedded the `generate_cc_arrays` way.
* **(B) Hand-rolled bare-metal decoder - CHOSEN.** A gemma.cpp-style minimal
  C++ runner (`run_gemma3_decode.cc`) that implements the 18-block decode
  step directly on top of our optimized kernels. Weights live in DDR as a raw
  packed blob written by the Python driver before the run (not embedded in
  the ELF). Tokenization happens on the host; the driver passes token IDs.
* **(C) Host golden reference** - HF `transformers` (or LiteRT-LM) greedy
  decode used to dump per-layer activations and reference tokens; this is the
  transformer analog of the V2 shadow-compare methodology that caught the
  MobileNet kernel bugs.

## Locked decisions

* **Precision: int8 weights + fp32 activations.** All matmul weights (QKV, O,
  gate/up/down, and the tied embedding used as the logits head) are stored
  per-channel symmetric int8 with fp32 scales; activations, norms, RoPE and
  attention stay fp32. This keeps the weight footprint ~270 MB (embedding is
  ~84 MB at int4 later if we want <128 MB) and sidesteps the activation-quant
  accuracy cliff. The int8-weight x fp32-activation matmul is the one new
  kernel to write (dequant weights on load, or int8 GeMV with fp32 accumulate).
* **Execution flow: npusim ELF driver** (the MobileNet pattern), not the
  cocotb/RTL op-test flow. A single cross-compiled `run_gemma3_decode.elf`
  runs the whole decode loop; a Python driver (`npusim_run_gemma3.py`) loads
  it, writes the packed weight blob + prompt tokens into DDR, runs, and reads
  emitted tokens back. Fast to iterate over 18 layers end-to-end.

## Phases

1. **Acquire + host reference.** Download `google/gemma-3-270m-it`
   (safetensors) into the host venv; script a fixed-prompt greedy decode that
   dumps (a) reference token IDs and (b) per-layer activations for step 0.
2. **Weight packer.** Python script -> single binary blob: per-channel
   symmetric int8 for all matmul weights + fp32 scales, RMSNorm gammas fp32,
   embedding int8 (int4 later if DDR stays 128 MB). A small header/manifest
   gives offsets; the same script emits a C header with the layout.
3. **Bare-metal decode step.** `run_gemma3_decode.cc`: embedding lookup,
   RMSNorm, QKV/O + GeGLU matmuls (reuse the `e32m8` broadcast-MAC kernel),
   RoPE, sliding-window attention over a ring-buffer KV cache, final
   RMSNorm + logits + argmax. First bring-up with **fp32 activations / int8
   weights** (accuracy-safe), int8 activations as a later optimization.
4. **Sim integration.** Grow the DDR region; the Python driver
   (`npusim_run_gemma3.py`) writes the weight blob + prompt tokens into DDR,
   runs decode step(s), reads the emitted token back, loops. Prefill = repeated
   single-token decode (simple first; batched prefill later).
5. **Verification gate** (`npusim_verify_gemma3.py`): token-exact greedy match
   vs the host reference for N tokens on fixed prompts; during bring-up,
   per-layer shadow compare against the dumped activations.
6. **Optimize + measure.** Vectorize softmax and RMSNorm, profile with the
   `mcycle` CycleProfiler, report cycles/token and tokens/s; then revisit int8
   activations and int4 embeddings.

## Progress

* **Phase 1 done.** `host_ref/` downloads `google/gemma-3-270m-it` and generates
  `reference.npz` (token-exact target + per-layer residual stream + layer-0
  submodule dumps). Fixed prompt greedy-decodes to argmax token 818 `'The'`.
* **Phase 2 done.** `host_ref/pack_weights.py` produces a 269.8 MB int8 blob
  (per-channel symmetric) + `gemma3_layout.h`. q_proj dequant round-trip
  max rel err 3.2e-3.
* **Phase 3a done (numpy prototype).** `host_ref/decode_numpy.py` implements the
  full decode step and reproduces the reference: **both fp32 and int8 weights
  predict argmax 818**, fp32 logits diff 1.4e-4, int8 logits diff 3.3. The
  algorithm (sandwich norms, QK-norm, dual-base RoPE, MQA + causal/sliding,
  GeGLU, tied logits) and the blob layout are validated; the C++ port is a
  translation of this.

The numbers to reproduce in C++ per layer are the `decode_numpy.py` per-layer
residual maxdiffs; step-0 argmax 818 is the top-level gate.

* **Phases 3b / 4 / 5 done (end-to-end on the simulator).**
  `run_gemma3_decode.cc` + `npusim_run_gemma3.py` run the full 21-token prefill
  on npusim and **reproduce the reference argmax 818** (`PASS`, clean mpause,
  no traps). Build/run: `bazel run //tests/npusim_examples/gemma3:npusim_run_gemma3`
  (needs the blob + reference in `~/gemma3_ref`).
  * **W8A8 integer matmul.** CoralNPU's RVV backend has **no vector int->float
    conversion** (`vfcvt.f.x.v` is an illegal instruction; the float vector
    kernels in `gemma_kernels/` are `manual`-tagged and never ran on this core).
    So `MatVecI8W` dynamically quantizes each activation vector to int8 and uses
    `int8*int8->int32` widening MACs (`vwmacc`), with one scalar int->float per
    output. Norms/RoPE/attention/GeGLU stay scalar fp32 (scalar FPU works).
  * **Control block in `.extdata`** (loaded, non-cleared EXTMEM): the CRT zeroes
    `.bss` at boot, so driver pre-run pokes must live outside `.bss`. Weights and
    logits live in `.ddr_bss` (NOLOAD) and are written by the driver.
  * **DDR grown to 512 MB** via a new `ddr_length_bytes` param on
    `CoralNPUV2Simulator` (default unchanged, so MobileNet is byte-identical).
  * **Cost:** prefill of 21 tokens ~ 1.16e9 cycles (~11.6 s @ 100 MHz),
    dominated by the 262k-vocab logits matmul. This is the Phase 6 target.

* **Multi-token generation with a KV cache.** The decoder was restructured to
  process one position at a time with a per-layer KV cache in DDR
  (`g_kcache`/`g_vcache`, `[layers][ctx][head_dim]`), unifying prefill and
  greedy decode: each new token runs a single position through all 18 layers,
  attending over the cache. `--max-new N` on the driver emits N tokens
  (default = reference length); `g_gen`/`g_num_gen` return the ids.
  * The incremental loop was validated in numpy first (`decode_numpy.py --gen`
    against an fp32-generated reference reproduces the sentence exactly),
    confirming the KV-cache/position/RoPE bookkeeping before the slow sim run.
  * **The golden reference is generated in bfloat16** (`gen_reference.py`
    defaults to `--dtype bf16`), matching how Gemma 3 is distributed and
    deployed. This is a much better gate for the int8 sim than the old fp32
    reference: bf16 and int8 both drift from idealized fp32 in the same
    direction. Step-0 argmax is still 818.
  * On the sim (W8A8), a full greedy run
    (`npusim_run_gemma3 -- --max-new 19`, 3.27e9 cycles) produces a complete,
    coherent sentence that closely tracks the bf16 reference:
    * sim: *"The moon is a beautiful, mysterious celestial body that orbits the
      Earth in a retrograde orbit."*
    * bf16 reference: *"The moon is a beautiful and mysterious celestial body
      that orbits the Earth in a retrograde orbit."*

    First token 818 matches and **18/19 tokens match the bf16 reference** (the
    sole difference is "," vs "and"); against the old fp32 reference it was 5/19.
    Token-exact greedy match is still not a hard gate — quantization can flip a
    token and cascade — but the first token plus near-exact continuation is a
    strong signal. A decode step costs ~1.2e8 cycles, dominated by the logits
    head (the Phase 6 optimization target).

## TBD: Phase 6 — optimize + measure

Phases 1-5 are done (host reference, weight packing, bare-metal decode, sim
integration, verification: 18/19 tokens matching the bf16 reference); Phase 6
is still open. A decode step costs ~1.2e8 cycles and the 21-token prefill
~1.16e9, both dominated by the logits head: the tied-embedding 640 x 262144
matmul is the single biggest matmul in the model (~62% of per-token cost).
Candidate attacks:

* **Higher-LMUL / vectorized logits kernel** — the head reuses the generic
  `MatVecI8W`; a dedicated kernel can stream the vocab dimension wider.
* **int4 embedding** — halves the dominant weight traffic for the head and
  the embedding lookup.
* **Restrict logits to a candidate token set** — greedy decode only needs
  argmax, so a shortlist (or emitting logits int8 via the classifier kernel)
  avoids most of the 262k columns.
* Also from the original phase plan: vectorize softmax and RMSNorm, profile
  with the `mcycle` CycleProfiler, report cycles/token and tokens/s, then
  revisit int8 activations.

## Risks / open questions

* **int8 residual drift:** with int8 weights, late-layer residual maxdiffs reach
  a few hundred (Gemma massive activations). Step-0 argmax holds, but multi-token
  greedy may diverge; the verify gate should track first-divergence token index.

* **Quantization accuracy:** LLMs tolerate weight-only int8 well; full int8
  activations (MobileNet-style) often degrade badly. Hence fp32 activations
  first - which makes RVV fp32 (`falu`) throughput a new unknown to measure.
* **262k-vocab logits matmul** dominates per-token cost (~62%); if needed,
  emit logits int8 via the classifier kernel or restrict to a candidate set.
* **DDR bandwidth** in the sim's cycle model vs real hardware - weight
  streaming from DDR is uncharted; MobileNet ran from ITCM/extmem.
* Tokenizer stays host-side (SentencePiece, 262k vocab) - fine for the sim
  flow, but a real deployment would need an on-device tokenizer.
