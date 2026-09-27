# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs the full ImageNet MobileNet V1 0.25 int8 model on the RTL via cocotb.

RTL analog of tests/npusim_examples/mobilenet/npusim_run_real_mobilenet.py:
loads the same ELF (run_full_mobilenet_v1_real_binary.elf, with the real
.tflite embedded), feeds the cat image into `inference_input`, runs to
completion, and prints the top-5 ImageNet predictions.

HTIF serving, per-node progress reporting, and wedge detection live in
htif_util.py (shared with the head_repro testcase).

Expect a LONG run: inference is ~26M cycles on npusim, and RTL cycle counts
run several times higher (~130M+ to reach the classifier head), which at
typical Verilator speeds (~10-15k cycles/s) is multiple hours of wall time.
"""

import cocotb
import numpy as np

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_test_utils.backdoor import backdoor_load
from coralnpu_test_utils.sim_test_fixture import Fixture

try:
    from tests.cocotb.imagenet.htif_util import (
        ElfStrings, PROFILER_SYM, serve_htif_until_exit)
except ImportError:
    from htif_util import ElfStrings, PROFILER_SYM, serve_htif_until_exit

PKG = "coralnpu_hw/tests/npusim_examples/mobilenet/"
NUM_CLASSES = 1000

# Cat classes in ImageNet: tabby, tiger cat, Persian, Siamese, Egyptian.
CAT_CLASS_IDS = {281, 282, 283, 284, 285}


def load_cat_image(npy_path):
    """uint8 HWC image -> flat int8, exact zero-point encoding (pixel - 128).

    The model's input quantization is (scale=1.0, zero_point=-128); the conv
    kernels add input_offset = +128 back. See npusim_run_real_mobilenet.py.
    """
    image = np.load(npy_path)
    assert image.shape == (224, 224, 3), f"bad image shape {image.shape}"
    return (image.astype(np.int16) - 128).astype(np.int8).reshape(-1)


def load_labels(labels_path):
    with open(labels_path) as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels[1:]  # drop the leading "background" entry


@cocotb.test()
async def core_mini_rvv_imagenet_mobilenet(dut):
    """Full MobileNet V1 0.25 (real ImageNet weights) on RTL, cat image."""
    fixture = await Fixture.Create(dut, highmem=True)
    r = runfiles.Create()

    elf_path = r.Rlocation(PKG + "run_full_mobilenet_v1_real_binary.elf")
    await fixture.load_elf_and_lookup_symbols(
        elf_path,
        ["inference_input", "inference_output", "inference_status",
         "tohost", "tohost_ready", "fromhost", "fromhost_ready"],
        optional_symbols=[PROFILER_SYM])
    elf_strings = ElfStrings(elf_path)

    # Backdoor-write the image into DTCM (147 KB; frontdoor AXI would burn
    # thousands of bus transactions before the test even starts).
    image = load_cat_image(
        r.Rlocation(PKG + "images_224x224x3/cat_224x224_real.npy"))
    backdoor_load(fixture.symbols["inference_input"],
                  image.view(np.uint8))

    await fixture.core_mini_axi.execute_from(fixture.entry_point)
    exit_code, cycles = await serve_htif_until_exit(fixture, elf_strings)
    print(f"\nProgram exited with code {exit_code} after ~{cycles} cycles",
          flush=True)

    status = int((await fixture.read("inference_status", 1)).view(np.int8)[0])
    assert status == 0, f"inference_status = {status} (expected 0)"

    scores = (await fixture.read("inference_output", NUM_CLASSES)).view(np.int8)
    labels = load_labels(r.Rlocation(PKG + "labels/imagenet_labels.txt"))
    top5 = np.argsort(scores)[::-1][:5]
    print("Top 5 predictions:")
    for idx in top5:
        print(f"  class {idx:4d} ({labels[idx]}): {scores[idx]}")

    assert int(top5[0]) in CAT_CLASS_IDS, (
        f"top-1 class {top5[0]} ({labels[top5[0]]}) is not a cat")
