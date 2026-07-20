# CoralNPU MobileNet kernel + npusim examples

A minimal overlay on top of the upstream
[`google-coral/coralnpu`](https://github.com/google-coral/coralnpu) repo that
adds two optimized int8 convolution kernels and an end-to-end MobileNet V1
verification flow for the instruction-level simulator (`npusim`). It lets
another developer reproduce the work against a pinned upstream commit without
forking the whole repo.

## What this adds

* **New conv kernels** (`patches/0001-add-v2-conv-kernels.patch`):
  `Conv_3_3_3_8_V2` and `Conv_1x1_Pointwise_V2` in
  `sw/opt/litert-micro/conv.cc`, plus the dispatch that routes the MobileNet
  stem and pointwise shapes to them, and a one-field addition to
  `sw/opt/litert-micro/memory_util.h`.
* **npusim MobileNet examples** (`patches/0002-wire-npusim-examples.patch`
  plus `overlay/`): a real ImageNet-trained MobileNet V1 0.25 runner and a
  kernel verifier that runs the model over 10 random ImageNet validation
  images and checks top-1/top-5 against ground truth.

## Layout

```
BASE_COMMIT        # upstream coralnpu commit these changes apply to
apply.sh           # applies patches + overlay onto a coralnpu checkout
patches/           # diffs for files that already exist upstream
overlay/           # net-new files, mirroring coralnpu's paths
```

The 10 validation image tensors (`images/val_*_224x224.npy`, ~1.5 MB) are
**not** shipped: they are regenerated deterministically by
`prepare_val_images.py` (fixed seed), so only the model, labels, manifest, and
the reference cat image are carried here.

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
(cd coralnpu && bazel run //tests/npusim_examples:npusim_verify_val10)
```

If you are offline or lack `numpy`/`pillow`, generate the images manually
before step 3: `python3 coralnpu/tests/npusim_examples/prepare_val_images.py
--seed 42 --count 10`.

**Time:** step 3's first run fetches Bazel deps (MPACT simulator, tflite_micro)
over the network, then simulates 10 images at ~136M cycles each — budget
~30-40 min total.

## Expected result

`--seed 42 --count 10` samples classes 25, 104, 114, 142, 228, 250, 281, 654,
754, 759. The verifier passes at top-1 >= 4/10 and top-5 >= 6/10; a
representative run scores 4/10 top-1 and 6/10 top-5 at ~136M cycles per image.
A broken kernel collapses these to ~0, which is the failure signal. See
`overlay/tests/npusim_examples/README.md` for the full flow and log-line
reference.

## Applying to a different upstream commit

The patches are pinned to `BASE_COMMIT`. On a newer coralnpu, `apply.sh` warns
and `git apply --check` may fail if the upstream `conv.cc` / `memory_util.h`
have diverged; in that case rebase the two patches in `patches/` by hand (the
overlay files are net-new and copy in regardless).
