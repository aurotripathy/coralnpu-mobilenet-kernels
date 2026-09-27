// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Minimal bare-metal repro for the vl=0 vector load/store pipeline deadlock.
//
// Per the RISC-V V spec, vector memory ops with vl=0 must retire as no-ops.
// On the RvvCoreMiniHighmemAxi RTL they instead hang retirement forever
// (RVV core never sends the rvv2lsu handshake for a 0-uop op, so the scalar
// LSU slot waits indefinitely). MPACT/npusim executes them correctly, which
// is how the bug reached full-model runs (GCC's RVV memcpy expansion emits a
// do-while loop that runs vle8/vse8 once even for length 0 -- see
// tflite::micro::GetTensorShape on rank-0 tensors, first hit by the
// STRIDED_SLICE node of MobileNet).

#include <stdint.h>
#include <stdio.h>

namespace {
volatile uint8_t src[16] = {1, 2, 3, 4};
volatile uint8_t dst[16];
}  // namespace

int main(int argc, char** argv) {
  // Control: same sequence with vl=1 must work.
  printf("vl=1 vector copy: start\n");
  asm volatile(
      "li t0, 1\n"
      "vsetvli t1, t0, e8, m1, ta, ma\n"
      "vle8.v v1, (%0)\n"
      "vse8.v v1, (%1)\n"
      :
      : "r"(src), "r"(dst)
      : "t0", "t1", "memory");
  printf("vl=1 vector copy: done (dst[0]=%d)\n", dst[0]);

  // Suspect: identical sequence with vl=0. Spec: both ops are no-ops.
  printf("vl=0 vector copy: start\n");
  asm volatile(
      "li t0, 0\n"
      "vsetvli t1, t0, e8, m1, ta, ma\n"
      "vle8.v v1, (%0)\n"
      "vse8.v v1, (%1)\n"
      :
      : "r"(src), "r"(dst)
      : "t0", "t1", "memory");
  printf("vl=0 vector copy: done\n");
  return 0;
}
