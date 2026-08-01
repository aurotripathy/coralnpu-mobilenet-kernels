# CoralNPU npusim examples: MobileNet kernels + Gemma 3

A minimal overlay on top of the upstream
[`google-coral/coralnpu`](https://github.com/google-coral/coralnpu) repo that
adds two things for the instruction-level simulator (`npusim`): optimized int8
convolution kernels with an end-to-end MobileNet V1 verification flow, and a
bare-metal Gemma 3 270M decoder that runs a full prefill + greedy generation.
It lets another developer reproduce the work against a pinned upstream commit
without forking the whole repo.

## What this adds

* **Optimized conv kernels** (`patches/0001-add-v2-conv-kernels.patch`,
  touching `conv.cc`, `accumulator_util.h`, and `memory_util.h`): vectorized
  int8 kernels for the MobileNet stem (`Conv_3_3_3_8`, a broadcast MAC at
  `e32m2` whose 8 lanes cover all 8 output channels in one tile) and the
  pointwise 1x1 layers (`Conv_1x1_Pointwise`, a broadcast MAC at `e32m8`
  covering 32 output channels per instruction), the dispatch that routes those
  shapes to them, a scalar `PrepareShiftParams` (fixes a requant corruption on
  the 1000-class classifier), and a one-field addition to `memory_util.h`. Both
  kernels seed the accumulator with `input_offset * sum(W)` rather than adding
  the input zero point to every input element; the stem takes that fast path
  for output pixels whose 3x3 window is fully in bounds and falls back to the
  bounds-checked path at the borders. This cuts CONV_2D from ~73.6M to ~12.7M
  cycles and the whole model from ~136M to ~26M cycles per image, bit-exact
  against the scalar reference. (The file also carries `*_V2` scalar variants
  as an unused bit-reference used during bring-up.)
* **npusim MobileNet examples** (`patches/0002-wire-npusim-examples.patch`
  plus `overlay/`): a real ImageNet-trained MobileNet V1 0.25 runner and a
  kernel verifier that runs the model over 10 random ImageNet validation
  images and checks top-1/top-5 against ground truth.
* **Gemma 3 270M decoder** (`patches/0003-gemma3-sim-ddr.patch` plus
  `overlay/tests/npusim_examples/gemma3/`): a hand-rolled bare-metal RVV decoder
  (`run_gemma3_decode.cc`) with int8 weights + fp32 activations and a W8A8
  integer matmul (CoralNPU's RVV has no vector int->float), driven by an npusim
  ELF driver (`npusim_run_gemma3.py`). It runs prefill + KV-cache greedy decode
  over all 18 layers and reproduces the bfloat16 host reference's first token
  (818) with 18/19 tokens matching. The patch only enlarges the simulated DDR
  region (a `ddr_length_bytes` param) so the 270 MB int8 weight blob fits;
  everything else is net-new. See
  `overlay/tests/npusim_examples/gemma3/README.md` for the design and the
  host-side prep (`host_ref/`).

## Layout

```
BASE_COMMIT        # upstream coralnpu commit these changes apply to
apply.sh           # applies patches + overlay onto a coralnpu checkout
patches/           # diffs for files that already exist upstream
overlay/           # net-new files, mirroring coralnpu's paths
```

The 10 validation image tensors (`images_224x224x3/val_*_224x224.npy`,
~1.5 MB) are
**not** shipped: they are regenerated deterministically by
`prepare_val_images.py` (fixed seed), so only the model, labels, manifest, and
the reference cat image are carried here.

The shipped `.tflite` is not from a model zoo — it is Keras Applications
MobileNet V1 (`alpha=0.25`, 224x224, `weights='imagenet'`) with the
`x/127.5 - 1` input preprocessing folded into `conv1`'s weights and BatchNorm,
then fully int8-quantized. The scripts that produced it are carried under
`mobilenet/make_models/` for provenance; see that folder's `README.md` for the
lineage and how to regenerate. Note the weights are Keras ImageNet weights, so
confirm redistribution rights before reusing the `.tflite` elsewhere.

## Reproduce

Prerequisites: Bazel 7.4.1 (via `bazelisk`, which reads `.bazelversion`) and
SRecord — see upstream coralnpu
[System Requirements](https://github.com/google-coral/coralnpu#system-requirements).
Image regeneration needs `numpy` + `pillow` and network access:
`pip install numpy pillow`.

Run every command below from **this** repo's root (a fresh checkout each time —
`apply.sh` expects an unpatched coralnpu):

```bash
# 0. Get this repo.
git clone <this-repo-url> && cd coralnpu-mobilenet-kernels

# 1. Get upstream at the exact pinned commit.
git clone https://github.com/google-coral/coralnpu.git
git -C coralnpu checkout "$(cat BASE_COMMIT)"

# 2. Apply the kernels + examples (also regenerates the 10 images).
./apply.sh coralnpu

# 3. Build and run the kernel verification.
(cd coralnpu && bazel run //tests/npusim_examples/mobilenet:npusim_verify_val10)
```

If you are offline or lack `numpy`/`pillow`, generate the images manually
before step 3: `python3 coralnpu/tests/npusim_examples/mobilenet/prepare_val_images.py
--seed 42 --count 10`.

**Time:** step 3's first run fetches Bazel deps (MPACT simulator, tflite_micro)
over the network, then simulates 10 images at ~26M cycles each — budget
~15-25 min total (dominated by the one-time dep fetch and build).

## Expected result

`--seed 42 --count 10` samples classes 25, 104, 114, 142, 228, 250, 281, 654,
754, 759. The verifier passes at top-1 >= 4/10 and top-5 >= 6/10; a
representative run scores 4/10 top-1 and 6/10 top-5 at ~26M cycles per image
(CONV_2D ~12.7M, ~65% of the total; the stem alone is ~4.3M). A broken kernel
collapses these to ~0, which is the failure signal. See
`overlay/tests/npusim_examples/mobilenet/README.md` for the full flow and
log-line reference.

## Gemma 3 270M (optional, extra host prep)

`apply.sh` copies the Gemma files and applies the DDR patch automatically, but
the example is **not** wired into the automated run: it needs the gated
`google/gemma-3-270m-it` checkpoint plus a ~270 MB int8 weight blob and a
bfloat16 reference that live outside the repo (in `~/gemma3_ref`), produced on
the host. In a Python env with `torch`, `transformers`, and Hugging Face access:

```bash
cd coralnpu/tests/npusim_examples/gemma3/host_ref
python gen_reference.py          # bf16 golden reference -> ~/gemma3_ref
python pack_weights.py           # int8 blob + gemma3_layout.h
(cd ../../../.. && bazel run //tests/npusim_examples/gemma3:npusim_run_gemma3)
```

Full design, precision rationale, and expected output are in
`overlay/tests/npusim_examples/gemma3/README.md` and `.../host_ref/README.md`.

## Applying to a different upstream commit

The patches are pinned to `BASE_COMMIT`. On a newer coralnpu, `apply.sh` warns
and `git apply --check` may fail if the upstream `conv.cc`,
`accumulator_util.h`, `memory_util.h`, or `coralnpu_v2_sim_utils.py` have
diverged; in that case rebase the patches in `patches/` by hand (the overlay
files are net-new and copy in regardless).
