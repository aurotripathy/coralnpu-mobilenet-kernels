# NPU Simulator MobileNet Examples

This folder contains end-to-end examples that run MobileNet V1 (alpha 0.25,
224x224 input) on the CoralNPU instruction-level simulator (`npusim`).

| Target | Model | Input | Output |
|---|---|---|---|
| `:npusim_run_mobilenet` | `mobilenet_v1_0.25_224_int8_dummy.tflite` (untrained weights, 5 classes) | random data | 5 scores |
| `:npusim_run_real_mobilenet` | `models/mobilenet_v1_025_224_int8_real.tflite` (ImageNet-trained, 1000 classes) | `images_224x224x3/cat_224x224_real.npy` | top-5 ImageNet labels |
| `:npusim_verify_val10` | same real model | 10 random ImageNet val images (`images_224x224x3/val_*_224x224.npy`) | pass/fail on top-1 & top-5 accuracy |

Run them from the repo root:

```bash
bazel run //tests/npusim_examples/mobilenet:npusim_run_real_mobilenet
```

## Verifying the kernels on real validation images

`:npusim_verify_val10` is an end-to-end kernel check that runs the real int8
MobileNet V1 0.25 over 10 images sampled at random (fixed seed) from the
ImageNet (ILSVRC-2012) validation set, one per class. Each image is
center-cropped to 224x224x3 and stored as `images_224x224x3/val_<class>_<label>_224x224.npy`;
`images_224x224x3/val10_manifest.json` records the ground-truth class index and label for
each. The driver runs the model on every image and compares the top-1/top-5
prediction against ground truth:

```bash
bazel run //tests/npusim_examples/mobilenet:npusim_verify_val10
```

MobileNet V1 0.25 is a small (~50% top-1) network, so the test does not demand
a perfect top-1 on every image. It passes when top-1 >= 4/10 and top-5 >= 6/10.
The point is to exercise the optimized conv and depthwise kernels on varied
real inputs: a broken kernel collapses these metrics to ~0 (bit-identical
garbage), whereas correct kernels produce sensible predictions and reasonable
near-misses. A representative run scores 4/10 top-1 and 6/10 top-5 (e.g. exact
hits on *European fire salamander*, *dowitcher*, *komondor*, *reflex camera*;
*tabby* landing just behind *Egyptian cat*), at ~26M cycles per image.

### Optional: CPU cross-check

`--cpu-check` additionally runs each image through a host TFLite interpreter
executing the *same* int8 `.tflite` and compares the raw int8 score vectors
(`utils/cpu_reference.py`). The CPU run is integer inference — int8 kernels,
int32 accumulators — not fp32/bf16; that is what makes the scores comparable
at all. The interpreter is not part of the Bazel environment, so pass a
Python that has one:

```bash
python3 -m venv ~/litertenv
~/litertenv/bin/pip install ai-edge-litert numpy
bazel run //tests/npusim_examples/mobilenet:npusim_verify_val10 -- \
    --cpu-check ~/litertenv/bin/python
```

Two comparisons run per image:

* **Scores** (`--cpu-tol`, default 48 LSB): max per-class |sim - cpu|. Do not
  tighten it expecting a bit-exact match: TFLM and TFLite implement some
  fixed-point ops (MEAN, SOFTMAX) differently, giving a benign 3-38 LSB
  spread on this set even though the conv kernels themselves track the TFLM
  reference to 1 LSB. The CPU path pins the `BUILTIN_REF` op resolver — the
  default XNNPack delegate drifts a few more LSB via its fp32
  requantization. A broken kernel misses by 100+ LSB, which is what the gate
  catches.
* **Labels**: does the CPU assign the same top-1 (tie-aware) and top-5 as the
  sim, right or wrong vs ground truth? Reported per image; gated on
  aggregates (`--cpu-top1-agree`, default 8/10 images; `--cpu-top5-overlap`,
  default mean 3.5/5) rather than per image, because on low-confidence
  images the score distribution is nearly flat — ranks 2-5 sit within a few
  LSB of the -128 floor, and the benign drift above legitimately reorders
  near-ties (a representative run agrees 9/10 on top-1 with mean overlap
  4.0; the one disagreement is *radio*, where both runs are wrong vs ground
  truth and the CPU's winner leads by ~4 LSB over a flat field). A broken
  kernel agrees on ~0/10.

### Why sim and CPU differ: TFLM vs TensorFlow Lite (LiteRT)

The two sides of the cross-check are **different runtimes** that happen to
execute the same `.tflite` flatbuffer:

* **TFLM** (TensorFlow Lite for Microcontrollers, the `tflite-micro`
  project) is what the simulator ELF runs: a bare-metal re-implementation
  with no OS, no heap (tensors live in a fixed pre-allocated arena), no
  delegates, and only explicitly registered kernels.
  `run_full_mobilenet_v1_real.cc` builds a TFLM `MicroInterpreter`, with the
  optimized RVV conv kernels from `sw/opt/litert-micro/conv.cc` registered
  in place of the stock convolutions.
* **TensorFlow Lite** (recently renamed **LiteRT**) is the full on-device
  runtime for phones/desktops — dynamic allocation, model loading from
  files, XNNPack and other delegates. This is what `--cpu-check` runs via
  the `ai-edge-litert` package.

Both implement the same int8 quantization spec, but they are separate
codebases, and the spec leaves small freedoms in fixed-point op
implementations — notably MEAN (rounding of the 7x7 global average) and
SOFTMAX (the integer exponential approximation). Each choice is worth a
rounding step or two at the logit level, but softmax renormalizes across
1000 classes, so those small logit differences become the 3-38 LSB spread
measured at the output. The conv kernels are *not* the source: on the
simulator, upstream TFLM reference convolutions match the optimized kernels
to 1 LSB (cat top-1 raw -60 vs -61). This cross-runtime drift is the reason
the score gate is an empirical 48-LSB envelope and the label gates are
aggregate rather than exact — the check compares across runtimes, and
bit-exactness is only a meaningful expectation within one.

### Regenerating / resampling the images

The images and `val10_manifest.json` are produced by
`prepare_val_images.py`, a host-side helper (needs `numpy`, `pillow`, and
network access). It samples one image per class from the public
one-image-per-class ImageNet mirror
[`EliSchwartz/imagenet-sample-images`](https://github.com/EliSchwartz/imagenet-sample-images),
center-crops each to 224x224x3, and writes `images_224x224x3/val_<class>_<label>_224x224.npy`
plus the manifest:

```bash
python3 tests/npusim_examples/mobilenet/prepare_val_images.py --seed 42 --count 10
```

The mirror lists files in synset order, so a file's position equals its
ImageNet class index, which lines up with `labels/imagenet_labels.txt`. The
default `--seed 42 --count 10` reproduces the committed set exactly (classes
25, 104, 114, 142, 228, 250, 281, 654, 754, 759); change `--seed` to draw a
different sample. To swap the set, delete the old `images_224x224x3/val_*_224x224.npy`
files and the manifest first, then rerun; the `glob` in `BUILD` picks up
whatever `val_*` files are present, so no `BUILD` edit is needed.

## How the flow works

1. The C++ program (`run_full_mobilenet_v1_real.cc`) is cross-compiled to a
   RISC-V ELF with the `.tflite` model embedded in `.rodata`. It exposes three
   `extern "C"` globals so the host can find them by symbol name:
   `inference_input` (224*224*3 bytes), `inference_output` (one int8 score per
   class), and `inference_status`.
2. The Python driver (`npusim_run_real_mobilenet.py`) parses the ELF, writes
   the image bytes into `inference_input`, runs the simulator to completion,
   and reads the class scores back from `inference_output`.
3. Output scores are quantized softmax probabilities:
   `probability = (raw + 128) / 256`.

## Flow by which the Keras-based MobileNet V1 model gets converted to an executable binary on the CoralNPU

The model graph is never compiled. The Keras network becomes a quantized
flatbuffer, the flatbuffer becomes a byte array, and that array is linked into
a RISC-V program whose code is the TFLite Micro interpreter plus hand-written
RVV kernels. The interpreter walks the flatbuffer at run time on the CoralNPU
core. There are two compilers in the chain, and neither sees the model as code:
the TFLite converter (graph lowering and quantization, host side) and the
RISC-V C++ cross-compiler (interpreter, kernels, runtime). There is no IREE or
MLIR stage anywhere in this repo.

### Stage 1 — Host: Keras → int8 `.tflite` (`make_models/make_real_model4.py`)

```python
model = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3), alpha=0.25, weights='imagenet')

cat = np.load(CAT_NPY).astype(np.float32)  # raw [0,255] pixels
ref = model(np.expand_dims(cat / 127.5 - 1.0, 0)).numpy()

conv1 = model.get_layer('conv1')
bn1 = model.get_layer('conv1_bn')
(W,) = conv1.get_weights()
gamma, beta, mean, var = bn1.get_weights()
conv1.set_weights([W / 127.5])
bn1.set_weights([gamma, beta, mean + W.sum(axis=(0, 1, 2)), var])
```

1. Instantiate Keras Applications MobileNet V1 (alpha=0.25, 224x224, 1000
   classes) with the pretrained ImageNet weights.
2. Fold the `x/127.5 - 1` preprocessing into `conv1`'s weights and
   `conv1_bn.moving_mean`, so the model takes raw `[0, 255]` pixels and no
   rescale op is in the graph.
3. Run `tf.lite.TFLiteConverter.from_keras_model`. This is the real "model
   compiler" step: it traces the Keras graph to TensorFlow ops, lowers them to
   TFLite builtins (CONV_2D, DEPTHWISE_CONV_2D, MEAN, RESHAPE, SOFTMAX), folds
   BatchNorm into conv biases, fuses ReLU6, and — driven by the 64-image
   representative dataset — does full-integer post-training quantization:
   per-channel symmetric int8 weights, int32 biases, per-tensor int8
   activations, int8 I/O.

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
open(OUT, 'wb').write(tflite_model)
```

Output: `models/mobilenet_v1_025_224_int8_real.tflite` (584 KB), a flatbuffer
with the op graph, tensor shapes, quant params, and weights. Checked into the
repo; everything downstream is Bazel.

### Stage 2 — Bazel: `.tflite` → C byte array (`generate_cc_arrays`)

TFLM's `generate_cc_arrays.py` hex-dumps the file into
`alignas(16) const unsigned char g_mobilenet_v1_025_224_int8_real_model_data[]`,
plus a header with `extern` and a `_size` constant.
`cc_library(mobilenet_v1_025_224_int8_real_lib)` wraps it. The bytes are
unchanged; this just makes them linkable.

### Stage 3 — Bazel: cross-compile and link the interpreter program (`coralnpu_v2_binary`)

`run_full_mobilenet_v1_real.cc` is the entry point. It declares the data
buffers the host will poke (`inference_input`, `inference_output`,
`inference_status`) plus a 4 MB `tensor_arena` in `.extdata`, and in `main()`:

```cpp
const tflite::Model* model =
    tflite::GetModel(g_mobilenet_v1_025_224_int8_real_model_data);
MobilenetOpResolver op_resolver;
RegisterOps(op_resolver);
static CycleProfiler profiler;
tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                     kTensorArenaSize,
                                     /*resource_variables=*/nullptr,
                                     &profiler);
if (interpreter.AllocateTensors() != kTfLiteOk) { ... }
```

`RegisterOps` is where CoralNPU plugs in its own kernels. CONV_2D and
DEPTHWISE_CONV_2D are registered with
`coralnpu_v2::opt::litert_micro::Register_CONV_2D()` /
`Register_DEPTHWISE_CONV_2D()` from `sw/opt/litert-micro/`, while the remaining
ops (MEAN, SOFTMAX, RESHAPE, ...) use stock TFLM reference kernels. The
CoralNPU registration reuses TFLM's `prepare` and swaps in its own `invoke`:

```cpp
TFLMRegistration Register_CONV_2D() {
  auto registration = tflite::Register_CONV_2D();
  // Prepare is the same as the reference implementation.
  registration.invoke = ConvEval;
```

`ConvEval` → `ConvPerChannel` → shape-based dispatch to RVV intrinsics kernels
(`Conv_3_3_3_8` for the stem, `Conv_1x1_Pointwise` for the 13 pointwise convs,
`Conv_4_4_*` variants, and `reference_integer_ops::ConvPerChannel` as
fallback).

The `coralnpu_v2_binary` macro (`rules/coralnpu_v2.bzl`) then:

* applies a platform transition to `//platforms:coralnpu_v2_semihosting`, so
  every dep is built with the RISC-V toolchain:
  `-march=rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma`
  (`toolchain/cc_toolchain_config.bzl`), i.e. RV32 with the Zve32f vector
  extension the kernels target;
* generates a linker script from `toolchain/coralnpu_tcm.ld.tpl` with
  ITCM/DTCM at 1 MB each (highmem). `.text`/`.rodata` (code + the model array)
  → ITCM, `.data` (I/O buffers) → DTCM, `.extdata` (tensor arena) → EXTMEM at
  `0x20000000`;
* compiles the sources and links with `-Wl,-T,<script>`, adding
  `//toolchain/crt:crt_semihosting` for startup/`printf`;
* emits `run_full_mobilenet_v1_real_binary.elf` (and `.bin` via `objcopy`,
  `.vmem` via `srec_cat`).

### Stage 4 — Run: host driver + simulator

`npusim_run_real_mobilenet.py` loads the ELF into `CoralNPUV2Simulator` (the
MPACT ISS, `@coralnpu_mpact`), reads the symbol table, writes the
`uint8 → int8` image into `inference_input`, runs to `ebreak`, then reads
`inference_output` and `inference_status`. On the core,
`MicroInterpreter::AllocateTensors` plans the arena from the flatbuffer, and
`Invoke()` walks the op list, calling each registered kernel. For the RTL path
(`tests/cocotb/imagenet`), the same ELF is loaded into the Verilator/VCS
testbench via HTIF.

### Pipeline summary

```
tf.keras.applications.MobileNet(alpha=0.25, weights='imagenet')
   |  fold x/127.5-1 into conv1 + conv1_bn            (make_real_model4.py, host)
   |  TFLiteConverter: lower to TFLite builtins,
   |  BN fold, ReLU6 fuse, int8 PTQ (64-image calib)
   v
models/mobilenet_v1_025_224_int8_real.tflite          <- committed artifact
   |  generate_cc_arrays (TFLM tool, genrule)
   v
g_mobilenet_v1_025_224_int8_real_model_data[]  (.rodata)
   |  cc_library
   v
coralnpu_v2_binary                                    (Bazel, RISC-V transition)
   +- run_full_mobilenet_v1_real.cc   GetModel -> MicroInterpreter
   +- @tflite_micro  framework + reference kernels (MEAN, SOFTMAX, ...)
   +- sw/opt/litert-micro  CONV_2D / DEPTHWISE_CONV_2D in RVV intrinsics
   +- crt_semihosting, generated TCM linker script
   v
run_full_mobilenet_v1_real_binary.elf  (+ .bin, .vmem)
   |  py_binary runfiles
   v
CoralNPUV2Simulator (MPACT ISS)  or  cocotb RTL testbench
   write inference_input -> run -> read inference_output
```

The split of responsibilities: the TFLite converter decides *what* ops run and
at what precision; TFLite Micro decides *the order and memory plan* at run time
from the flatbuffer; the CoralNPU kernels in `sw/opt/litert-micro/` decide
*how* each conv executes on the vector unit; the RISC-V toolchain and linker
script decide *where in memory* it all lives.

## From .tflite to RISC-V ELF: how the model gets into the simulator

The `.tflite` model is never converted or lowered into code. Its raw flatbuffer
bytes are embedded as a C array inside the C++ runner, and that program is
cross-compiled to a RISC-V ELF. The whole pipeline lives in this folder's
`BUILD` file:

```
models/*.tflite --(generate_cc_arrays)--> model .cc/.h (const unsigned char[])
                                              |
run_full_mobilenet_v1_real.cc  +  model array | --(coralnpu_v2_binary)--> .elf
                                              |
npusim_run_*.py --(load_program(.elf))--> simulator executes it
```

Stage by stage:

1. **`.tflite` -> C array** (`generate_cc_arrays` targets in `BUILD`). This is
   a genrule (defined in `rules/utils.bzl`) that runs TFLite Micro's
   `//tensorflow/lite/micro/tools:generate_cc_arrays` tool. It dumps the
   flatbuffer bytes into a `const unsigned char g_..._data[]` plus a length
   constant, emitted as `mobilenet_v1_025_224_int8_real.cc/.h` and wrapped in
   the `mobilenet_v1_025_224_int8_real_lib` cc_library. The model stays a
   flatbuffer; at runtime TFLM's `MicroInterpreter` parses and walks it in
   place (`tflite::GetModel(g_..._data)`), so there is no ahead-of-time
   compilation of the graph.
2. **C++ + model array -> ELF** (`coralnpu_v2_binary` target in `BUILD`,
   rule in `rules/coralnpu_v2.bzl`). `run_full_mobilenet_v1_real.cc` includes
   the generated header and hands the array to the interpreter. The rule
   cross-compiles it with the CoralNPU RISC-V toolchain, links against the
   optimized kernels (`//sw/opt/litert-micro:conv`, `:depthwise_conv`) and the
   TFLM framework, using a generated linker script sized by
   `itcm_size_kbytes` / `dtcm_size_kbytes` (1024 KB each here, the "highmem"
   layout), and emits `run_full_mobilenet_v1_real_binary.elf` (plus an
   `objcopy`'d `.bin`).
3. **ELF -> simulator.** The Python drivers resolve the ELF from runfiles and
   call `npu_sim.load_program(elf)`; the simulator maps the ELF's `PT_LOAD`
   segments into memory (see the log-line section below) and runs it.

So the only true model "conversion" happens back at quantization time (see
`make_models/README.md`); from there on the model is data, not code.

## Adding a test case for a new real image

### 1. Prepare the image as a .npy file

The input must be a `(224, 224, 3)` array, RGB channel order:

* `uint8` values in `[0, 255]` (recommended), or
* `int8` values in `[-128, 127]` (already shifted by -128).

The driver's `load_input_from_npy()` validates the shape and converts uint8 to
the int8 domain the model expects (`pixel - 128`).

Example conversion from a JPEG/PNG (run on the host; needs `pillow` + `numpy`):

```python
import numpy as np
from PIL import Image

img = Image.open("my_image.jpg").convert("RGB")
# Resize the short side to 224, then center-crop 224x224.
w, h = img.size
s = 224 / min(w, h)
img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
w, h = img.size
img = img.crop(((w - 224) // 2, (h - 224) // 2,
                (w - 224) // 2 + 224, (h - 224) // 2 + 224))
np.save("images_224x224x3/my_image_224x224.npy", np.asarray(img, dtype=np.uint8))
```

### 2. Add the file to the test's runfiles

New data files must be listed in the `data` attribute of the `py_binary` in
`BUILD`, or Bazel will not make them available at runtime:

```python
py_binary(
    name = "npusim_run_real_mobilenet",
    ...
    data = [
        "images_224x224x3/cat_224x224_real.npy",
        "images_224x224x3/my_image_224x224.npy",   # <-- add
        "labels/imagenet_labels.txt",
        ":run_full_mobilenet_v1_real_binary",
    ],
    ...
)
```

### 3. Point the driver at the image

Either edit the `image_file` path in `npusim_run_real_mobilenet.py`, or (for a
separate test case) copy the `run_real_mobilenet()` function / the whole
driver under a new name and add a matching `py_binary` target. Runfile paths
are resolved as:

```python
image_file = r.Rlocation(
    'coralnpu_hw/tests/npusim_examples/mobilenet/images_224x224x3/my_image_224x224.npy')
```

### 4. Run and interpret

```bash
bazel run //tests/npusim_examples/mobilenet:npusim_run_real_mobilenet
```

The driver prints the top-5 classes with their ImageNet label names (from
`labels/imagenet_labels.txt`; the file has 1001 lines and the leading
"background" entry is dropped so line N+1 corresponds to class index N).
Expect most classes to sit at raw score -128 (probability 0); only the top few
carry probability mass. Ties at the same raw score are unordered - the int8
softmax resolution is 1/256 (~0.4%).

## Notes and constraints

* **Image size is fixed at 224x224x3** by the embedded model's input tensor.
  A different resolution requires converting a new `.tflite` and updating the
  `inference_input` buffer size in the `.cc` file.
* **ITCM budget:** the model lives in `.rodata` inside the 1 MB ITCM region,
  which is ~84% full with the 597 KB real model. Bigger models (e.g. alpha
  0.5) will not fit without moving the model to `.extdata` (see the `extdata`
  option of `generate_cc_arrays` in `rules/utils.bzl`) or growing
  `itcm_size_kbytes` in `BUILD`.
* **Conv kernels:** the dispatch in `sw/opt/litert-micro/conv.cc` routes the
  stem (3x3x3->8) to `Conv_3_3_3_8` (broadcast MAC at `e32m2`, whose 8 lanes
  cover all 8 output channels in one tile) and the 1x1 pointwise shapes to
  `Conv_1x1_Pointwise` (broadcast MAC at `e32m8`, 32 output channels per
  instruction). Both seed the accumulator with `input_offset * sum(W)` instead
  of adding the input zero point to every input element; the stem does this on
  a fast path for output pixels whose 3x3 window is fully in bounds, falling
  back to the bounds-checked path at the borders. Both are bit-exact against
  the scalar `_V2` reference kernels, which remain in the file as an unused
  bit-reference from bring-up.
* **Expected result for the cat image:** top-1 "tabby" (raw -61, ~26%), with
  Egyptian cat / tiger cat close behind, in ~26M cycles. The stem is the
  single most expensive node (~4.3M cycles, 16%): it runs at the largest
  spatial extent (112x112) while being the narrowest layer in the channel
  dimension that these kernels vectorize over.

## Understanding the simulator log lines

Each run prints a few informational (`I0000 ...`) log lines from the
simulator (`coralnpu_mpact/sim/coralnpu_simulator.cc`) as it loads the ELF.
They are harmless boot noise, repeated once per image in the val10 run. To
suppress them, run with `GLOG_minloglevel=1`.

```
Adding memory region for segment: 0x00000000:0x000d2044 (0x5)
Adding memory region for segment: 0x00100000:0x00100000 (0x3)
Adding memory region for segment: 0x20000000:0x00400000 (0x3)
HTIF magic addresses: tohost=0x00100000, tohost_ready=0x00100040, fromhost=0x00100080, fromhost_ready=0x001000c0
```

Because semihosting is enabled, the simulator walks the ELF's `PT_LOAD`
segments and registers each as a RAM region. The format is
`start_address:size (permission_bits)`, where the permission bits are
`Read=1, Write=2, Execute=4` OR-ed together:

* `0x00000000:0x000d2044 (0x5)` - ITCM segment (~856 KB), `Read+Execute`.
  Holds code + `.rodata`, including the embedded `.tflite` model. Not
  writable.
* `0x00100000:0x00100000 (0x3)` - DTCM (1 MB), `Read+Write`. Data/bss/heap/
  stack; matches `dtcm_size_kbytes = 1024` in `BUILD`.
* `0x20000000:0x00400000 (0x3)` - external memory / `.extdata` (4 MB),
  `Read+Write`. Backs the 4 MB `tensor_arena` in
  `run_full_mobilenet_v1_real.cc`.
* `HTIF magic addresses: tohost=... fromhost=...` - addresses of the HTIF
  (host-target interface) handshake buffers found in the ELF. Semihosting
  uses these `tohost`/`fromhost` mailboxes so the RISC-V program can call
  back into the host for `printf` output and the exit/halt request.
