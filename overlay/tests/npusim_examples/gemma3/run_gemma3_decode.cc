// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Bare-metal Gemma 3 270M prefill forward on the CoralNPU simulator.
//
// Weights are int8 (per-output-channel symmetric) stored in DDR; the Python
// driver writes the packed blob into g_weights and the prompt token IDs into
// g_tokens before running. The program runs a single prefill pass over the
// prompt and writes the argmax of the final-position logits into g_argmax.
// This is a direct translation of host_ref/decode_numpy.py (validated to
// reproduce the reference argmax 818).

#include <math.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "sw/utils/utils.h"
#include "tests/npusim_examples/gemma3/gemma3_layout.h"

#define MAX_SEQ 64
#define MAX_CTX 64
#define EOS_TOKEN 1

#define H GEMMA3_HIDDEN            // 640
#define HD GEMMA3_HEAD_DIM         // 256
#define NH GEMMA3_N_HEADS          // 4
#define NKV GEMMA3_N_KV_HEADS      // 1
#define QDIM (NH * HD)             // 1024
#define KVDIM (NKV * HD)           // 256
#define I GEMMA3_INTERMEDIATE      // 2048

// ---- Control block (symbols looked up + poked by the driver). ----
// Placed in .extdata (a loaded, non-cleared EXTMEM section) so the driver's
// pre-run pokes survive the CRT .bss clear at boot.
#define CTRL __attribute__((section(".extdata"))) __attribute__((aligned(16)))
extern "C" {
volatile uint32_t g_num_tokens CTRL = 0;   // prompt length (driver -> sim)
volatile uint32_t g_max_new CTRL = 1;      // tokens to generate (driver -> sim)
volatile int32_t g_tokens[MAX_SEQ] CTRL = {0};  // prompt ids (driver -> sim)
volatile int32_t g_gen[MAX_CTX] CTRL = {0};     // generated ids (sim -> driver)
volatile uint32_t g_num_gen CTRL = 0;      // count generated (sim -> driver)
volatile int32_t g_argmax CTRL = -1;       // first generated id (sim -> driver)
volatile uint32_t g_status CTRL = 0;
volatile uint32_t g_cycles CTRL = 0;

// Weight arena in DDR (NOLOAD): driver writes the packed blob here.
int8_t g_weights[GEMMA3_TOTAL_BYTES] __attribute__((section(".ddr_bss")))
__attribute__((aligned(16)));
// Final logits in DDR (1 MB, too big for DTCM).
float g_logits[GEMMA3_VOCAB] __attribute__((section(".ddr_bss")))
__attribute__((aligned(16)));
// KV cache in DDR: [layers][ctx][head_dim] for the single KV head (MQA).
float g_kcache[GEMMA3_N_LAYERS * MAX_CTX * KVDIM]
    __attribute__((section(".ddr_bss"))) __attribute__((aligned(16)));
float g_vcache[GEMMA3_N_LAYERS * MAX_CTX * KVDIM]
    __attribute__((section(".ddr_bss"))) __attribute__((aligned(16)));
}

// ---- Single-position activation working set (EXTMEM). ----
namespace {
float a_hidden[H] __attribute__((section(".extbss")));
float a_x[H] __attribute__((section(".extbss")));
float a_q[QDIM] __attribute__((section(".extbss")));
float a_k[KVDIM] __attribute__((section(".extbss")));
float a_v[KVDIM] __attribute__((section(".extbss")));
float a_attn[QDIM] __attribute__((section(".extbss")));
float a_gate[I] __attribute__((section(".extbss")));
float a_up[I] __attribute__((section(".extbss")));
float a_tmp[H] __attribute__((section(".extbss")));
float a_scores[MAX_CTX] __attribute__((section(".extbss")));

const float* wf(long off) {
  return reinterpret_cast<const float*>(g_weights + off);
}
const int8_t* wi(long off) {
  return reinterpret_cast<const int8_t*>(g_weights + off);
}
long layer_base(int i) {
  return GEMMA3_LAYER0_OFF + static_cast<long>(i) * GEMMA3_LAYER_STRIDE;
}

// Scratch for the dynamically int8-quantized activation vector (max in_dim=I).
int8_t g_xq[I] __attribute__((section(".extbss")));

// y[o] = scale[o] * xscale * sum_k w_q[o,k] * xq[k].
//
// CoralNPU's RVV backend has no vector int->float conversion, so the matmul
// runs entirely in integers (W8A8): the fp32 activation is dynamically
// quantized to int8 per call, accumulated int8*int8->int32 with widening MACs,
// then a single scalar int->float per output. w is row-major [out_dim, in_dim].
void MatVecI8W(int out_dim, int in_dim, const int8_t* w, const float* scale,
               const float* x, float* y) {
  float amax = 0.0f;
  for (int k = 0; k < in_dim; ++k) {
    float a = x[k] < 0 ? -x[k] : x[k];
    if (a > amax) amax = a;
  }
  float xscale = amax > 0.0f ? amax / 127.0f : 1.0f;
  float inv = 1.0f / xscale;
  for (int k = 0; k < in_dim; ++k) {
    float f = x[k] * inv;
    int q = static_cast<int>(f + (f >= 0 ? 0.5f : -0.5f));
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    g_xq[k] = static_cast<int8_t>(q);
  }

  size_t vlfull = __riscv_vsetvlmax_e32m8();
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
  for (int o = 0; o < out_dim; ++o) {
    const int8_t* wp = w + static_cast<long>(o) * in_dim;
    const int8_t* xp = g_xq;
    vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vlfull);
    size_t k = in_dim;
    while (k) {
      size_t vl = __riscv_vsetvl_e8m2(k);
      vint8m2_t wv = __riscv_vle8_v_i8m2(wp, vl);
      vint16m4_t w16 = __riscv_vwadd_vx_i16m4(wv, 0, vl);
      vint8m2_t xv = __riscv_vle8_v_i8m2(xp, vl);
      vint16m4_t x16 = __riscv_vwadd_vx_i16m4(xv, 0, vl);
      acc = __riscv_vwmacc_vv_i32m8(acc, w16, x16, vl);
      wp += vl;
      xp += vl;
      k -= vl;
    }
    int32_t dot = __riscv_vmv_x_s_i32m1_i32(
        __riscv_vredsum_vs_i32m8_i32m1(acc, vzero, vlfull));
    y[o] = static_cast<float>(dot) * xscale * scale[o];
  }
}

// RMSNorm over one vector of length n: y = x*rsqrt(mean(x^2)+eps)*(1+gamma).
void RmsNorm(int n, const float* x, const float* gamma, float* y) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) ss += x[i] * x[i];
  float inv = 1.0f / sqrtf(ss / n + GEMMA3_RMS_EPS);
  for (int i = 0; i < n; ++i) y[i] = x[i] * inv * (1.0f + gamma[i]);
}

// RoPE (rotate-half) on a [HD] head vector at position p.
void Rope(float* vec, int p, float theta) {
  int half = HD / 2;
  for (int i = 0; i < half; ++i) {
    float freq = 1.0f / powf(theta, (2.0f * i) / HD);
    float ang = p * freq;
    float c = cosf(ang), s = sinf(ang);
    float x1 = vec[i], x2 = vec[i + half];
    vec[i] = x1 * c - x2 * s;
    vec[i + half] = x2 * c + x1 * s;
  }
}

float GeluTanh(float g) {
  return 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
}

const float kAttnScale = 1.0f / sqrtf((float)GEMMA3_HEAD_DIM);

// One transformer layer for a single position `pos`, with a KV cache. The
// residual `hidden[H]` is updated in place; this position's K/V are written to
// g_kcache/g_vcache[L][pos] and attention runs over cached positions.
void DecodeLayer(int L, int pos, float* hidden) {
  long b = layer_base(L);
  bool sliding = ((L % 6) != 5);
  float theta = sliding ? 10000.0f : 1000000.0f;
  float* kc = &g_kcache[(static_cast<long>(L) * MAX_CTX + pos) * KVDIM];
  float* vc = &g_vcache[(static_cast<long>(L) * MAX_CTX + pos) * KVDIM];

  RmsNorm(H, hidden, wf(b + GEMMA3_L_INPUT_LAYERNORM), a_x);
  MatVecI8W(QDIM, H, wi(b + GEMMA3_L_Q_PROJ_Q), wf(b + GEMMA3_L_Q_PROJ_SCALE),
            a_x, a_q);
  MatVecI8W(KVDIM, H, wi(b + GEMMA3_L_K_PROJ_Q), wf(b + GEMMA3_L_K_PROJ_SCALE),
            a_x, a_k);
  MatVecI8W(KVDIM, H, wi(b + GEMMA3_L_V_PROJ_Q), wf(b + GEMMA3_L_V_PROJ_SCALE),
            a_x, a_v);
  for (int h = 0; h < NH; ++h) {
    float* qh = &a_q[h * HD];
    RmsNorm(HD, qh, wf(b + GEMMA3_L_Q_NORM), qh);
    Rope(qh, pos, theta);
  }
  RmsNorm(HD, a_k, wf(b + GEMMA3_L_K_NORM), a_k);
  Rope(a_k, pos, theta);
  for (int d = 0; d < KVDIM; ++d) {
    kc[d] = a_k[d];
    vc[d] = a_v[d];
  }

  int lo = 0;
  if (sliding && pos >= GEMMA3_SLIDING_WINDOW)
    lo = pos - GEMMA3_SLIDING_WINDOW + 1;
  const float* kbase = &g_kcache[static_cast<long>(L) * MAX_CTX * KVDIM];
  const float* vbase = &g_vcache[static_cast<long>(L) * MAX_CTX * KVDIM];
  for (int h = 0; h < NH; ++h) {
    const float* qh = &a_q[h * HD];
    float mx = -INFINITY;
    for (int j = lo; j <= pos; ++j) {
      const float* kj = kbase + static_cast<long>(j) * KVDIM;
      float s = 0.0f;
      for (int d = 0; d < HD; ++d) s += qh[d] * kj[d];
      s *= kAttnScale;
      a_scores[j] = s;
      if (s > mx) mx = s;
    }
    float denom = 0.0f;
    for (int j = lo; j <= pos; ++j) {
      a_scores[j] = expf(a_scores[j] - mx);
      denom += a_scores[j];
    }
    float* oh = &a_attn[h * HD];
    for (int d = 0; d < HD; ++d) oh[d] = 0.0f;
    for (int j = lo; j <= pos; ++j) {
      float p = a_scores[j] / denom;
      const float* vj = vbase + static_cast<long>(j) * KVDIM;
      for (int d = 0; d < HD; ++d) oh[d] += p * vj[d];
    }
  }
  MatVecI8W(H, QDIM, wi(b + GEMMA3_L_O_PROJ_Q), wf(b + GEMMA3_L_O_PROJ_SCALE),
            a_attn, a_tmp);
  RmsNorm(H, a_tmp, wf(b + GEMMA3_L_POST_ATTENTION_LAYERNORM), a_tmp);
  for (int d = 0; d < H; ++d) hidden[d] += a_tmp[d];

  RmsNorm(H, hidden, wf(b + GEMMA3_L_PRE_FEEDFORWARD_LAYERNORM), a_x);
  MatVecI8W(I, H, wi(b + GEMMA3_L_GATE_PROJ_Q), wf(b + GEMMA3_L_GATE_PROJ_SCALE),
            a_x, a_gate);
  MatVecI8W(I, H, wi(b + GEMMA3_L_UP_PROJ_Q), wf(b + GEMMA3_L_UP_PROJ_SCALE),
            a_x, a_up);
  for (int d = 0; d < I; ++d) a_gate[d] = GeluTanh(a_gate[d]) * a_up[d];
  MatVecI8W(H, I, wi(b + GEMMA3_L_DOWN_PROJ_Q), wf(b + GEMMA3_L_DOWN_PROJ_SCALE),
            a_gate, a_tmp);
  RmsNorm(H, a_tmp, wf(b + GEMMA3_L_POST_FEEDFORWARD_LAYERNORM), a_tmp);
  for (int d = 0; d < H; ++d) hidden[d] += a_tmp[d];
}

// Embed one token into `hidden` (dequant int8 row * sqrt(H)).
void Embed(int tok, float* hidden) {
  const int8_t* row = wi(GEMMA3_EMBED_Q_OFF) + static_cast<long>(tok) * H;
  float sc = wf(GEMMA3_EMBED_SCALE_OFF)[tok] * sqrtf((float)H);
  for (int d = 0; d < H; ++d) hidden[d] = row[d] * sc;
}

// Final norm + tied-embedding logits + argmax for the current residual.
int LogitsArgmax(const float* hidden) {
  RmsNorm(H, hidden, wf(GEMMA3_FINAL_NORM_OFF), a_x);
  MatVecI8W(GEMMA3_VOCAB, H, wi(GEMMA3_EMBED_Q_OFF), wf(GEMMA3_EMBED_SCALE_OFF),
            a_x, g_logits);
  int best = 0;
  float bestv = g_logits[0];
  for (int v = 1; v < GEMMA3_VOCAB; ++v) {
    if (g_logits[v] > bestv) {
      bestv = g_logits[v];
      best = v;
    }
  }
  return best;
}
}  // namespace

extern "C" int main() {
  g_status = 2;
  uint32_t start = mcycle_read();
  const int S = static_cast<int>(g_num_tokens);
  int max_new = static_cast<int>(g_max_new);
  if (S + max_new > MAX_CTX) max_new = MAX_CTX - S;

  // Prefill: run every prompt position through all layers to fill the cache.
  int tok = 0;
  for (int pos = 0; pos < S; ++pos) {
    Embed(g_tokens[pos], a_hidden);
    for (int L = 0; L < GEMMA3_N_LAYERS; ++L) DecodeLayer(L, pos, a_hidden);
    if (pos == S - 1) tok = LogitsArgmax(a_hidden);
  }
  g_status = 3;  // prefill done

  // Greedy decode.
  g_argmax = tok;
  g_gen[0] = tok;
  int ng = 1;
  for (int step = 1; step < max_new; ++step) {
    if (tok == EOS_TOKEN) break;
    int pos = S - 1 + step;
    Embed(tok, a_hidden);
    for (int L = 0; L < GEMMA3_N_LAYERS; ++L) DecodeLayer(L, pos, a_hidden);
    tok = LogitsArgmax(a_hidden);
    g_gen[ng++] = tok;
  }
  g_num_gen = ng;
  g_status = 1;  // fully complete
  g_cycles = static_cast<uint32_t>(mcycle_read() - start);
  return 0;
}
