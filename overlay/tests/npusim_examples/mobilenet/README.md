# NPU Simulator MobileNet Examples

This folder contains end-to-end examples that run MobileNet V1 (alpha 0.25,
224x224 input) on the CoralNPU instruction-level simulator (`npusim`).

| Target | Model | Input | Output |
|---|---|---|---|
| `:npusim_run_mobilenet` | `mobilenet_v1_0.25_224_int8_dummy.tflite` (untrained weights, 5 classes) | random data | 5 scores |
| `:npusim_run_real_mobilenet` | `models/mobilenet_v1_025_224_int8_real.tflite` (ImageNet-trained, 1000 classes) | `images/cat_224x224_real.npy` | top-5 ImageNet labels |
| `:npusim_verify_val10` | same real model | 10 random ImageNet val images (`images/val_*_224x224.npy`) | pass/fail on top-1 & top-5 accuracy |

Run them from the repo root:

```bash
bazel run //tests/npusim_examples/mobilenet:npusim_run_real_mobilenet
```

## Verifying the kernels on real validation images

`:npusim_verify_val10` is an end-to-end kernel check that runs the real int8
MobileNet V1 0.25 over 10 images sampled at random (fixed seed) from the
ImageNet (ILSVRC-2012) validation set, one per class. Each image is
center-cropped to 224x224x3 and stored as `images/val_<class>_<label>_224x224.npy`;
`images/val10_manifest.json` records the ground-truth class index and label for
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
*tabby* landing just behind *Egyptian cat*), at ~33M cycles per image.

### Regenerating / resampling the images

The images and `val10_manifest.json` are produced by
`prepare_val_images.py`, a host-side helper (needs `numpy`, `pillow`, and
network access). It samples one image per class from the public
one-image-per-class ImageNet mirror
[`EliSchwartz/imagenet-sample-images`](https://github.com/EliSchwartz/imagenet-sample-images),
center-crops each to 224x224x3, and writes `images/val_<class>_<label>_224x224.npy`
plus the manifest:

```bash
python3 tests/npusim_examples/mobilenet/prepare_val_images.py --seed 42 --count 10
```

The mirror lists files in synset order, so a file's position equals its
ImageNet class index, which lines up with `labels/imagenet_labels.txt`. The
default `--seed 42 --count 10` reproduces the committed set exactly (classes
25, 104, 114, 142, 228, 250, 281, 654, 754, 759); change `--seed` to draw a
different sample. To swap the set, delete the old `images/val_*_224x224.npy`
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
np.save("images/my_image_224x224.npy", np.asarray(img, dtype=np.uint8))
```

### 2. Add the file to the test's runfiles

New data files must be listed in the `data` attribute of the `py_binary` in
`BUILD`, or Bazel will not make them available at runtime:

```python
py_binary(
    name = "npusim_run_real_mobilenet",
    ...
    data = [
        "images/cat_224x224_real.npy",
        "images/my_image_224x224.npy",   # <-- add
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
    'coralnpu_hw/tests/npusim_examples/mobilenet/images/my_image_224x224.npy')
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
  stem (3x3x3->8) to `Conv_3_3_3_8` (vectorized, tiled over output channels)
  and the 1x1 pointwise shapes to `Conv_1x1_Pointwise` (broadcast MAC at
  `e32m8`, 32 output channels per instruction). Both are bit-exact against the
  scalar `_V2` reference kernels, which remain in the file as an unused
  bit-reference from bring-up.
* **Expected result for the cat image:** top-1 "tabby" (raw -61, ~26%), with
  Egyptian cat / tiger cat close behind, in ~33M cycles.

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
