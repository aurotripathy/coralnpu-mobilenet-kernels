# MobileNet V1 (alpha=0.25) in PyTorch, matched to the coralnpu int8 .tflite

Two PyTorch implementations of MobileNet V1 alpha=0.25 that load their weights
from the ImageNet model of the enclosing npusim example
(`../models/mobilenet_v1_025_224_int8_real.tflite`) and are checked against it:

| file | model | arithmetic | purpose |
|---|---|---|---|
| `mobilenet_v1_fp32.py` | `MobileNetV1` | float32, weights dequantized from the .tflite | readable reference of the architecture |
| `mobilenet_v1_int8.py` | `MobileNetV1Int8` | int8 weights/activations, int32 accumulation, the exact integer requantization of LiteRT-Micro | bit-exact reproduction of what the NPU executes |

`utils.py` holds the loaders (`load_tflite_weights`, `load_tflite_weights_int8`),
image/label loading and a runner for the .tflite reference. `main.py` ties it together.

## Running

Not part of the Bazel build; run it with a Python that has `torch`, `numpy`
and `ai-edge-litert` (e.g. `python3 -m venv ~/torchenv && ~/torchenv/bin/pip
install torch numpy ai-edge-litert`):

```bash
cd tests/npusim_examples/mobilenet/pytorch
~/torchenv/bin/python main.py                 # fp32 model (default)
~/torchenv/bin/python main.py --dtype int8    # int8 model, per-stage bit-exact check
~/torchenv/bin/python main.py --help          # --labels / --tflite / --image overrides
```

Both modes classify `../images_224x224x3/cat_224x224_real.npy`, print the top-5
next to the .tflite's top-5, and a per-stage table (params, shapes, dtype). In
int8 mode the table has an extra column comparing every stage's output with
the corresponding .tflite tensor:

```
layer        params   in shape         out shape        dtype            vs tflite
stem            224   3 x 224 x 224    8 x 112 x 112    int8 w, int32 b  bit-exact
block 1..13     ...                                      int8 w, int32 b  bit-exact
head        257,000   256 x 7 x 7      1000             int8 w, int32 b  152 of 1000 differ, max 1 LSB
```

## Architecture

Stem 3x3/s2 conv -> 13 depthwise-separable blocks (3x3 depthwise + 1x1
pointwise, every conv followed by BN + ReLU6) -> global average pool -> 1000-way
classifier. Channel counts are `BLOCK_SPECS` (the alpha=1.0 network) times
alpha, rounded to a multiple of 8. 3x3 convs use TensorFlow 'SAME' padding,
which for stride 2 pads bottom/right only (`SamePad3x3`); PyTorch's symmetric
`padding=1` does not match the .tflite. 470,072 parameters (257,000 in the
classifier). See the `MobileNetV1` docstring for the per-layer shape table.

The int8 model imports `make_divisible`, `SamePad3x3` and `BLOCK_SPECS` from
the fp32 file and mirrors its module layout (`stem`, `blocks[i].dw`,
`blocks[i].pw`, `head.avgpool`, `head.fc`, plus `softmax`).

## How the .tflite is mapped

* Fully int8 post-training quantized: per-output-channel symmetric int8
  weights, int32 biases, per-tensor int8 activations. BatchNorm is folded into
  the conv biases, so the .tflite has no BN tensors.
* The `x/127.5 - 1` input preprocessing is folded into the stem, so the model
  takes raw `[0, 255]` pixels; input quantization is `q = pixel - 128`.
* Every conv activation is quantized with scale 6/255, zero point -128: the
  fused ReLU6 is exactly the int8 range, so ReLU6 becomes a clamp.
* Ops are mapped by position in execution order: op 0 is the stem, ops 1-26
  alternate depthwise/pointwise for blocks 1-13, then MEAN, the classifier
  conv, RESHAPE, SOFTMAX. Layouts: conv `[Cout,kh,kw,Cin] -> [Cout,Cin,kh,kw]`,
  depthwise `[1,kh,kw,C] -> [C,1,kh,kw]`, classifier `[1000,1,1,256]`.

fp32 model: weights are dequantized, and each conv's bias is carried in the
following `BatchNorm2d` configured as identity + bias (gamma=1, beta=bias,
mean=0, var=1-eps), which keeps the standard Conv/BN/ReLU6 structure.

int8 model: nothing is dequantized. `Int8Conv2d` does
`acc = sum((x_q - x_zp) * w_q) + bias_q` in int32 (computed exactly in
float64), then `MultiplyByQuantizedMultiplier` with per-channel Q31
multiplier/shift (double rounding, as coralnpu's kernels require), adds the
output zero point and clamps. `Int8GlobalAvgPool` is TFLM's
`QuantizedMeanOrSum`; `Int8Softmax` is a port of the gemmlowp fixed-point
softmax. The fixed-point primitives were transcribed from the TFLM/gemmlowp
sources in coralnpu's bazel cache.

## Verification status (cat image)

* int8 model vs. desktop LiteRT (builtin kernels, no XNNPACK): all 27 conv
  activations bit-exact; `fc` and `softmax` bit-exact when given the
  interpreter's own inputs.
* The only difference is the MEAN: desktop LiteRT uses
  `optimized_integer_ops::Mean`, LiteRT-Micro (what coralnpu runs) uses
  `QuantizedMeanOrSum`. They round differently on 48 of 256 pooled channels
  (+-1 LSB), which propagates to +-1 LSB on 152 of 1000 logits and a few
  softmax entries. The model implements the TFLM formula on purpose.
* fp32 model vs. the same .tflite: top-5 classes agree; probabilities differ
  by up to ~0.08 because the float path never rounds activations to int8
  (tabby / Egyptian cat are a near-tie and swap).
* `npusim_run_real_mobilenet` (the NPU simulator) gives softmax scores 3-28
  LSB away from both LiteRT and this model, with the same top-4 classes. See
  "Quirks to be investigated" below.

## Quirks to be investigated

### NPU simulator scores differ from both LiteRT and this model

The prebuilt `npusim_run_real_mobilenet` (it finishes in ~4 s) was run on the
same cat image. The NPU's int8 softmax scores are `281: -61, 285: -61,
282: -61, 287: -75` (everything else -128); both the TFLM-semantics int8 model
here and desktop LiteRT give roughly `-49, -49, -58, -103`.

Whether the NPU's `vsmul`/`vssra` round-to-nearest-up requantization
(`sw/opt/litert-micro/accumulator_util.h`, `vxrm=0`) explains it was tested by
swapping gemmlowp's round-half-away-from-zero for round-half-up in the conv
requantization: it changes nothing. The ELF embeds the same model bytes and
uses the same input encoding (`pixel - 128`). MEAN and SOFTMAX on the NPU run
the stock TFLM reference kernels, which this model reproduces exactly.

So the 3-28 LSB gap is coming from the coralnpu conv kernels themselves
(`sw/opt/litert-micro/conv.cc`, `depthwise_conv.cc`), not from MEAN/SOFTMAX as
the coralnpu README suggests. That is a coralnpu-side question not pursued
here. The npusim only exposes the final output buffer, so pinning it down
would need per-layer dumps from the NPU side; the int8 model's stage-by-stage
outputs are ready to compare against them.
