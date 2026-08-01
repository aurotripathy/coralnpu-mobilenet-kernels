# make_models

Host-side scripts that generated `../models/mobilenet_v1_025_224_int8_real.tflite`,
the real ImageNet-trained MobileNet used by `npusim_run_real_mobilenet` and
`npusim_verify_val10`. They are kept for provenance and reproducibility; they
are **not** Bazel targets.

All four build the same base model — Keras Applications MobileNet V1
(`alpha=0.25`, 224x224, `weights='imagenet'`, 1000 classes) — and run full
int8 post-training quantization with the same representative dataset (the repo
cat image plus 63 random images, seed 42) and int8 input/output. They differ
only in **how the input preprocessing (`x/127.5 - 1`, mapping `[0,255]->[-1,1]`)
is handled**, so that the tflite accepts raw `[0,255]` pixels and quantizes the
input as `pixel - 128` (`zero_point=-128`) — the same regime as the repo's
dummy model and what the driver's `load_input_from_npy()` produces.

| Script | Preprocessing approach | Extra op in graph | Output |
|---|---|---|---|
| `make_real_model.py` | none; input stays in the `[-1, 1]` domain | n/a | `/tmp` (throwaway) |
| `make_real_model2.py` | adds a `Rescaling(1/127.5, offset=-1)` input layer | yes (rescale/quantize) | `../models/` |
| `make_real_model3.py` | folds rescale into `conv1` as a **bias**, rebuilding the model layer-by-layer | no | `../models/` |
| `make_real_model4.py` | folds rescale into `conv1` **weights** + the following BatchNorm `moving_mean`, in place | no | `../models/` |

## Lineage

The scripts are successive iterations on one goal (get raw-pixel int8 input
without an extra graph op):

1. **`make_real_model.py`** - baseline proof of concept. No fold, so the input
   domain is `[-1, 1]` and the repo's `pixel - 128` loader is only approximate.
   Writes to `/tmp` and includes extra debug output (label decode, both input
   regimes).
2. **`make_real_model2.py`** - fixes the input regime by wrapping the model in
   a `Rescaling` layer (input quant `scale=1, zp=-128`). Correct, but bakes an
   extra rescale op into the model that then runs on-chip.
3. **`make_real_model3.py`** - removes that op by folding the rescale into
   `conv1` via a bias term, using `conv(x/127.5 - 1, W) == conv(x, W/127.5) -
   sum(W)`. Rebuilds the model by cloning layers.
4. **`make_real_model4.py`** - same fold, done cleanly in place: scales `conv1`
   weights by `1/127.5` and absorbs the constant into `conv1_bn.moving_mean`
   (no bias, no rebuild). **This produced the shipped model.**

`make_real_model2/3/4.py` all write to the same `../models/` path, so the last
one run wins; `make_real_model4.py` is the current source of truth. The other
three are superseded and kept only to document how the model was derived.

## Reproduce

Needs `tensorflow`, `numpy`, and network access (Keras downloads the pretrained
MobileNet weights on first run into `~/.keras/models/`). The system `python3`
does **not** have these by default — install them into a virtualenv first:

```bash
python3 -m venv ~/tfenv
source ~/tfenv/bin/activate
pip install numpy tensorflow pillow
```

Then run (from anywhere; the scripts resolve `../images_224x224x3/` and
`../models/` relative to their own location):

```bash
python make_real_model4.py
```

> **Warning:** `make_real_model{2,3,4}.py` overwrite
> `../models/mobilenet_v1_025_224_int8_real.tflite` in place. Re-running
> reproduces an equivalent model but is not guaranteed byte-identical to the
> shipped file, so only run it if you intend to regenerate.

Each script prints the fold sanity-check diff, the tflite input quantization,
the op counts, and the cat image's top-5 so you can confirm the conversion.

## Note on origin / licensing

The model is a derivative of Keras Applications MobileNet ImageNet weights
(downloaded by `weights='imagenet'`), not an independently trained artifact.
Confirm redistribution rights for those weights before publishing the `.tflite`
outside this repo.

## TBD: remaining MobileNet optimization

Kernel-side work in `sw/opt/litert-micro/conv.cc`, not model generation, but
tracked here with the rest of the MobileNet provenance. Current state: ~26M
cycles/image after the `e32m2` stem tiling and the input-offset folds
(`input_offset * sum(W)` accumulator seeding) in both `Conv_3_3_3_8` and
`Conv_1x1_Pointwise`.

* **Restructure the stem to vectorize over output pixels instead of output
  channels.** The stem (`Conv_3_3_3_8`, node 0) is still the single most
  expensive node at ~4.3M cycles (16% of inference), spending ~12.6 cycles per
  tap iteration on what is only a vector load + MAC: with just 8 output
  channels it fills 8 lanes at best, and the same 864 bytes of gathered
  weights are re-fetched for all 12,544 output pixels. Vectorizing across
  `out_x` would broadcast weights as scalars and keep them out of the load
  stream, but input pixels for consecutive `out_x` are 6 bytes apart
  (stride 2 x 3 channels), so it needs strided loads (`vlse8`) or a gather —
  a rewrite, not a tweak.
* **MEAN and SOFTMAX** are the next-largest non-conv nodes (~1.6M and ~1.2M
  cycles, ~5% and ~4%); neither has been vectorized.
