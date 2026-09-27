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

"""npusim sanity run for the MobileNet-head repro ELF.

The repro chains must complete and match the host golden on npusim (MPACT);
the point of the repros is that the same ELFs wedge on RTL. Run with:

    bazel run //tests/cocotb/imagenet:npusim_run_head [-- reshape]
"""

import sys

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np

PKG = "coralnpu_hw/tests/cocotb/imagenet/"
NUM_CLASSES = 1000


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "head"
    npu_sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True)
    r = runfiles.Create()
    elf_file = r.Rlocation(PKG + f"run_{variant}_repro_binary.elf")

    if variant == "vl0":
        # Bare-metal vl=0 test: no tensors, just run to completion.
        entry_point, _ = npu_sim.get_elf_entry_and_symbol(elf_file, [])
        npu_sim.load_program(elf_file, entry_point)
        npu_sim.run()
        npu_sim.wait()
        print(f"cycles taken by the simulation {npu_sim.get_cycle_count()}")
        print("npusim vl0 repro PASS")
        return

    input_file = r.Rlocation(PKG + f"head_repro/{variant}_input.npy")
    golden_file = r.Rlocation(PKG + f"head_repro/{variant}_golden.npy")

    entry_point, symbol_map = npu_sim.get_elf_entry_and_symbol(
        elf_file, ["inference_status", "inference_input", "inference_output"])
    npu_sim.load_program(elf_file, entry_point)

    image = np.load(input_file).astype(np.int8).reshape(-1)
    npu_sim.write_memory(symbol_map["inference_input"], image)

    print("Running head repro on npusim...", flush=True)
    npu_sim.run()
    npu_sim.wait()
    print(f"cycles taken by the simulation {npu_sim.get_cycle_count()}")

    status = np.int8(npu_sim.read_memory(symbol_map["inference_status"], 1)[0])
    print(f"inference_status {status}")
    assert status == 0, "inference did not complete"

    scores = np.array(
        npu_sim.read_memory(symbol_map["inference_output"], NUM_CLASSES),
        dtype=np.int8)
    golden = np.load(golden_file)
    diff = np.abs(scores.astype(np.int32) - golden.astype(np.int32))
    print(f"vs host TFLite golden: max abs diff {diff.max()} LSB, "
          f"{int((diff > 0).sum())} of {NUM_CLASSES} scores differ")
    assert diff.max() <= 4, f"output differs from golden by {diff.max()} LSB"
    print("npusim head repro PASS")


if __name__ == "__main__":
    main()
