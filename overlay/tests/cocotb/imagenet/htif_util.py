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

"""HTIF server + progress/wedge diagnostics for semihosting ELFs on RTL.

ELFs built with `semihosting = True` use HTIF (tohost/fromhost words in
DTCM, see toolchain/crt/coralnpu_htif_gloss.cc): every printf spins until a
host services the syscall, and _exit writes `(status << 1) | 1`. npusim
serves HTIF natively; the cocotb testbench does not, so serve_htif_until_exit
polls tohost over AXI, prints SYS_write payloads, and detects the exit code.

It also reads live diagnostics while the core runs:
  - the program's `static CycleProfiler profiler` (symbol _ZZ4mainE8profiler,
    updated by TFLM around every node) -> which node is executing;
  - the core's internal CSRs, exported over AXI at csr_base + 0x100
    (CoreAxiCSR.scala): mepc/mtval/mcause (trap history) and minstret
    (retired instructions -- distinguishes a spinning PC from a parked one);
  - a checksum of the EXTMEM tensor arena (testbench-side numpy, free).

If node/arena/minstret-rate stop changing for several samples, it raises
with a full dump instead of burning hours of wall time.
"""

import collections

import numpy as np
from cocotb.triggers import ClockCycles
from cocotb.utils import get_sim_time
from elftools.elf.elffile import ELFFile

SYS_WRITE = 64

# Layout of the `static CycleProfiler profiler` in main() (rv32):
# vptr@0, tags_[64]@4, starts_[64]@264, cycles_[64]@776, count_@1032.
PROFILER_SYM = "_ZZ4mainE8profiler"
PROFILER_TAGS_OFF = 4
PROFILER_COUNT_OFF = 1032

# Internal-CSR window in the core's AXI CSR block (CoreAxiCSR.scala:
# kCsrBaseAddr = 0x100, one 32-bit reg per index of io.csr.out.value).
CSR_MEPC_OFF = 0x104
CSR_MTVAL_OFF = 0x108
CSR_MCAUSE_OFF = 0x10C
CSR_MINSTRET_OFF = 0x118


def install_axi_spy(core, depth=64):
    """Records the core's outbound (master-port) EXTMEM traffic.

    Wraps the testbench memory model's read_memory/write_memory so the last
    `depth` requests are kept with timestamps. If the pipeline dead-stalls on
    a memory access (minstret frozen, no fault), the tail of this log is the
    smoking gun: a request to a bogus address shows up here as ok=False
    (testbench answers SLVERR), and a request that never even reached the
    external port shows up as a clean tail with no failing entry.
    """
    log = collections.deque(maxlen=depth)
    orig_read, orig_write = core.read_memory, core.write_memory

    def spy_read(ardata):
        data = orig_read(ardata)
        log.append(f"{get_sim_time('ns'):.0f}ns R addr=0x{int(ardata['addr']):08x} "
                   f"len={int(ardata['len'])} size={int(ardata['size'])} "
                   f"ok={data is not None}")
        return data

    def spy_write(wdata):
        ok = orig_write(wdata)
        log.append(f"{get_sim_time('ns'):.0f}ns W addr=0x{int(wdata['addr']):08x} "
                   f"len={int(wdata['len'])} size={int(wdata['size'])} ok={ok}")
        return ok

    core.read_memory = spy_read
    core.write_memory = spy_write
    return log


class ElfStrings:
    """Resolves in-ELF pointers (e.g. op-name literals in .rodata) offline."""

    def __init__(self, elf_path):
        self.chunks = []
        with open(elf_path, "rb") as f:
            for sec in ELFFile(f).iter_sections():
                if sec["sh_type"] == "SHT_PROGBITS" and sec["sh_addr"]:
                    self.chunks.append((sec["sh_addr"], sec.data()))

    def string_at(self, ptr):
        for base, data in self.chunks:
            if base <= ptr < base + len(data):
                end = data.find(b"\0", ptr - base)
                return data[ptr - base:end].decode(errors="replace")
        return f"<ptr 0x{ptr:08x}>"


async def serve_htif_until_exit(fixture, elf_strings,
                                max_cycles=2_000_000_000,
                                poll_cycles=10_000,
                                progress_cycles=10_000_000,
                                wedge_intervals=5):
    """Serves HTIF until the program exits; returns (exit_code, cycles)."""
    core = fixture.core_mini_axi
    clk = core.dut.io_aclk
    tohost_addr = fixture.symbols["tohost"]
    prof_addr = fixture.symbols.get(PROFILER_SYM)
    axi_log = install_axi_spy(core)

    async def read_csr(offset):
        return int((await core.read(core.csr_base_addr + offset, 4))
                   .view(np.uint32)[0])

    async def read_node():
        """Returns (nodes_started, current op tag) from the live profiler."""
        if prof_addr is None:
            return -1, "?"
        count = int((await core.read(
            prof_addr + PROFILER_COUNT_OFF, 4)).view(np.int32)[0])
        if count <= 0:
            return 0, "-"
        tag_ptr = int((await core.read(
            prof_addr + PROFILER_TAGS_OFF + 4 * (count - 1),
            4)).view(np.uint32)[0])
        return count, elf_strings.string_at(tag_ptr)

    async def diag_dump():
        words = []
        for sym in ("tohost", "tohost_ready", "fromhost", "fromhost_ready"):
            v = int((await core.read(fixture.symbols[sym], 8))
                    .view(np.uint64)[0])
            words.append(f"{sym}=0x{v:x}")
        mepc = await read_csr(CSR_MEPC_OFF)
        mtval = await read_csr(CSR_MTVAL_OFF)
        mcause = await read_csr(CSR_MCAUSE_OFF)
        minstret = await read_csr(CSR_MINSTRET_OFF)
        wfi = getattr(core.dut, "io_wfi", None)
        # Live master-port channel state: is a request or response pending?
        arv = int(core.dut.io_axi_master_read_addr_valid.value)
        ara = int(core.dut.io_axi_master_read_addr_bits_addr.value)
        awv = int(core.dut.io_axi_master_write_addr_valid.value)
        awa = int(core.dut.io_axi_master_write_addr_bits_addr.value)
        rready = int(core.dut.io_axi_master_read_data_ready.value)
        axi_tail = "\n  ".join(list(axi_log))
        return (f"{' '.join(words)} "
                f"mepc=0x{mepc:08x} mtval=0x{mtval:08x} mcause=0x{mcause:x} "
                f"minstret={minstret} "
                f"halted={int(core.dut.io_halted.value)} "
                f"fault={int(core.dut.io_fault.value)} "
                f"wfi={int(wfi.value) if wfi is not None else '?'}\n"
                f"master AXI live: arvalid={arv} araddr=0x{ara:08x} "
                f"awvalid={awv} awaddr=0x{awa:08x} rready={rready}\n"
                f"last EXTMEM requests (oldest first):\n  {axi_tail}")

    elapsed = 0
    wait = poll_cycles
    next_progress = progress_cycles
    arena_sum = int(np.sum(core.memory, dtype=np.uint64))
    last_state = None
    last_minstret = 0
    stuck = 0
    while elapsed < max_cycles:
        await ClockCycles(clk, wait)
        elapsed += wait
        if fixture.fault():
            raise AssertionError(
                f"core fault at ~{elapsed} cycles; {await diag_dump()}")
        if core.dut.io_halted.value == 1:
            raise AssertionError(
                f"core halted without HTIF exit at ~{elapsed} cycles; "
                f"{await diag_dump()}")
        if elapsed >= next_progress:
            next_progress += progress_cycles
            new_sum = int(np.sum(core.memory, dtype=np.uint64))
            count, tag = await read_node()
            minstret = await read_csr(CSR_MINSTRET_OFF)
            state = "arena active" if new_sum != arena_sum else "arena idle"
            print(f"[progress] ~{elapsed / 1e6:.0f}M cycles, "
                  f"node {count - 1} ({tag}), {state}, "
                  f"minstret +{minstret - last_minstret}", flush=True)
            last_minstret = minstret
            if (count, new_sum) == last_state:
                stuck += 1
                if stuck >= wedge_intervals:
                    raise AssertionError(
                        f"wedged in node {count - 1} ({tag}) for "
                        f"~{stuck * progress_cycles / 1e6:.0f}M cycles; "
                        f"{await diag_dump()}")
            else:
                stuck = 0
            last_state = (count, new_sum)
            arena_sum = new_sum
        tohost = int((await core.read(tohost_addr, 8)).view(np.uint64)[0])
        if tohost == 0:
            wait = poll_cycles
            continue
        if tohost & 1:
            return (tohost >> 1), elapsed  # program exited
        # Service the syscall. buf layout: n, a0..a5 (uint64 each).
        buf = (await core.read(tohost, 32)).view(np.uint64)
        n, a0, a1, a2 = (int(buf[0]), int(buf[1]), int(buf[2]), int(buf[3]))
        ret = 0
        if n == SYS_WRITE:
            payload = bytes(await core.read(a1, a2))
            print(payload.decode(errors="replace"), end="", flush=True)
            ret = a2
        # Return value goes into buf[0]; clear tohost/tohost_ready *before*
        # releasing the core so the next poll doesn't see a stale syscall.
        await core.write(tohost, np.array([ret], dtype=np.uint64).view(np.uint8))
        await core.write(tohost_addr, np.zeros(8, dtype=np.uint8))
        await core.write(fixture.symbols["tohost_ready"],
                         np.zeros(8, dtype=np.uint8))
        await core.write(fixture.symbols["fromhost_ready"],
                         np.array([1], dtype=np.uint64).view(np.uint8))
        # Output usually comes in bursts (profiler summary): re-poll quickly.
        wait = 200
    raise TimeoutError(f"no HTIF exit within {max_cycles} cycles")
