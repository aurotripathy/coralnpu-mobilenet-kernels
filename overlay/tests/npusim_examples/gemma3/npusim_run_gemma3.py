# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs the Gemma 3 270M prefill forward on the NPU simulator.

Writes the packed int8 weight blob and the reference prompt tokens into the
simulator, runs a single prefill pass, and checks that the argmax of the final
logits matches the host reference (Phase 3b gate).

The weight blob + reference live outside the repo (default ~/gemma3_ref),
produced by host_ref/pack_weights.py and host_ref/gen_reference.py.
"""

import argparse
import os

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np

# 512 MB DDR: 270 MB weights + logits + KV cache + headroom.
DDR_LENGTH_BYTES = 0x20000000
SYMBOLS = ["g_weights", "g_logits", "g_tokens", "g_num_tokens", "g_max_new",
           "g_gen", "g_num_gen", "g_argmax", "g_status", "g_cycles"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.expanduser("~/gemma3_ref"))
    ap.add_argument("--max-new", type=int, default=0,
                    help="tokens to generate (0 = match reference length)")
    args = ap.parse_args()

    r = runfiles.Create()
    elf = r.Rlocation(
        "coralnpu_hw/tests/npusim_examples/gemma3/run_gemma3_decode_binary.elf")

    blob = np.fromfile(os.path.join(args.ref, "gemma3_weights.bin"),
                       dtype=np.uint8)
    ref = np.load(os.path.join(args.ref, "reference.npz"))
    tokens = ref["input_ids"].astype(np.int32)
    ref_gen = ref["gen_ids"].astype(np.int64)
    expected = int(ref["argmax0"])
    max_new = args.max_new if args.max_new > 0 else len(ref_gen)
    print(f"prompt tokens: {len(tokens)}, max_new: {max_new}, "
          f"expected first token: {expected}, weight blob: {blob.size/1e6:.1f} MB")

    sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True,
                              ddr_length_bytes=DDR_LENGTH_BYTES)
    entry, sym = sim.get_elf_entry_and_symbol(elf, SYMBOLS)
    sim.load_program(elf, entry)

    print(f"writing weight blob to g_weights @ {sym['g_weights']:#x} ...",
          flush=True)
    sim.write_memory(sym["g_weights"], blob)
    sim.write_memory(sym["g_tokens"], tokens)
    sim.write_memory(sym["g_num_tokens"],
                     np.array([len(tokens)], dtype=np.uint32))
    sim.write_memory(sym["g_max_new"], np.array([max_new], dtype=np.uint32))

    print("running (prefill + greedy decode) ...", flush=True)
    sim.run()
    sim.wait()

    status = int(np.array(sim.read_memory(sym["g_status"], 4)).view(np.uint32)[0])
    argmax = int(np.array(sim.read_memory(sym["g_argmax"], 4)).view(np.int32)[0])
    ng = int(np.array(sim.read_memory(sym["g_num_gen"], 4)).view(np.uint32)[0])
    cycles = int(np.array(sim.read_memory(sym["g_cycles"], 4)).view(np.uint32)[0])
    gen = np.array(sim.read_memory(sym["g_gen"], 4 * ng)).view(np.int32).astype(np.int64)

    n_cmp = min(len(gen), len(ref_gen))
    match = int((gen[:n_cmp] == ref_gen[:n_cmp]).sum())
    print(f"status {status}  first token {argmax} (expected {expected})")
    print(f"generated ({ng}): {gen.tolist()}")
    print(f"reference ({len(ref_gen)}): {ref_gen.tolist()}")
    print(f"first-token {'PASS' if argmax == expected else 'FAIL'}; "
          f"vs fp32 reference {match}/{n_cmp} tokens match")
    print(f"decode mcycle delta {cycles}  sim total cycles {sim.get_cycle_count()}")


if __name__ == "__main__":
    main()
