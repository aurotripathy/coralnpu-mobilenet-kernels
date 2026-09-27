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

"""RTL repros for the MobileNet-head STRIDED_SLICE wedge.

Two sizes of the same bug (full model wedges in STRIDED_SLICE after ~4h):
  - core_mini_rvv_head_repro: MEAN -> CONV_2D -> SHAPE -> STRIDED_SLICE ->
    PACK -> RESHAPE -> SOFTMAX (7x7x256 input), wedges at ~10M cycles.
  - core_mini_rvv_reshape_repro: SHAPE -> STRIDED_SLICE -> PACK -> RESHAPE
    -> SOFTMAX only (1x1x1000 input), minutes per iteration.

Outputs are compared against host-TFLite goldens from make_*_model.py.
"""

import os

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

PKG = "coralnpu_hw/tests/cocotb/imagenet/"
NUM_CLASSES = 1000


async def _run_repro(dut, elf_name, input_npy, golden_npy,
                     progress_cycles=2_000_000):
    fixture = await Fixture.Create(dut, highmem=True)
    r = runfiles.Create()

    elf_path = r.Rlocation(PKG + elf_name)
    await fixture.load_elf_and_lookup_symbols(
        elf_path,
        ["inference_input", "inference_output", "inference_status",
         "tohost", "tohost_ready", "fromhost", "fromhost_ready"],
        optional_symbols=[PROFILER_SYM])
    elf_strings = ElfStrings(elf_path)

    data = np.load(r.Rlocation(PKG + input_npy))
    backdoor_load(fixture.symbols["inference_input"],
                  data.astype(np.int8).reshape(-1).view(np.uint8))

    await fixture.core_mini_axi.execute_from(fixture.entry_point)
    # WEDGE_MAX_CYCLES caps the run for waveform capture (the wedge hits at
    # ~72k cycles; a small cap keeps the VCD from the trace model tiny).
    max_cycles = int(os.environ.get("WEDGE_MAX_CYCLES", 100_000_000))
    exit_code, cycles = await serve_htif_until_exit(
        fixture, elf_strings, max_cycles=max_cycles,
        progress_cycles=progress_cycles, wedge_intervals=3)
    print(f"\nProgram exited with code {exit_code} after ~{cycles} cycles",
          flush=True)

    status = int((await fixture.read("inference_status", 1)).view(np.int8)[0])
    assert status == 0, f"inference_status = {status} (expected 0)"

    scores = (await fixture.read("inference_output", NUM_CLASSES)).view(np.int8)
    golden = np.load(r.Rlocation(PKG + golden_npy))
    diff = np.abs(scores.astype(np.int32) - golden.astype(np.int32))
    print(f"vs host TFLite golden: max abs diff {diff.max()} LSB, "
          f"{int((diff > 0).sum())} of {NUM_CLASSES} scores differ")
    # TFLM-vs-TFLite fixed-point drift allowance (see mobilenet README).
    assert diff.max() <= 4, f"output differs from golden by {diff.max()} LSB"


@cocotb.test()
async def core_mini_rvv_vl0_repro(dut):
    """Bare-metal proof: vle8/vse8 with vl=0 deadlock the pipeline."""
    fixture = await Fixture.Create(dut, highmem=True)
    r = runfiles.Create()
    elf_path = r.Rlocation(PKG + "run_vl0_repro_binary.elf")
    await fixture.load_elf_and_lookup_symbols(
        elf_path,
        ["tohost", "tohost_ready", "fromhost", "fromhost_ready"],
        optional_symbols=[PROFILER_SYM])
    await fixture.core_mini_axi.execute_from(fixture.entry_point)
    exit_code, cycles = await serve_htif_until_exit(
        fixture, ElfStrings(elf_path), max_cycles=2_000_000,
        progress_cycles=200_000, wedge_intervals=3)
    print(f"\nProgram exited with code {exit_code} after ~{cycles} cycles",
          flush=True)
    assert exit_code == 0


@cocotb.test()
async def core_mini_rvv_head_repro(dut):
    """MobileNet classifier-head op chain on RTL (STRIDED_SLICE wedge repro)."""
    await _run_repro(dut, "run_head_repro_binary.elf",
                     "head_repro/head_input.npy",
                     "head_repro/head_golden.npy")


@cocotb.test()
async def core_mini_rvv_reshape_repro(dut):
    """SHAPE/STRIDED_SLICE/PACK/RESHAPE chain only -- fastest wedge repro."""
    # Wedges (when it wedges) at ~72k cycles; sample fast to abort early.
    await _run_repro(dut, "run_reshape_repro_binary.elf",
                     "head_repro/reshape_input.npy",
                     "head_repro/reshape_golden.npy",
                     progress_cycles=500_000)
