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

"""
MobileNet V1 (Howard et al., 2017) with width multiplier alpha, fully int8:
weights, biases, activations and the arithmetic in between are the integer
operations that LiteRT-Micro executes for the coralnpu int8 .tflite, so every
activation can be compared with the .tflite (or the NPU) bit for bit.

The architecture constants (make_divisible, SamePad3x3, BLOCK_SPECS) come from
mobilenet_v1_fp32.py; only the arithmetic differs. The module layout mirrors the
float model so tooling can walk both the same way:
    stem, blocks[i].dw, blocks[i].pw, head.avgpool, head.fc, softmax
There are no BatchNorm modules: the .tflite has BN folded into the conv bias.
Nothing here is trainable, so weights and quantization parameters are buffers,
not Parameters.

Integer pipeline (TFLM reference kernels; coralnpu's optimized kernels error
out if TFLITE_SINGLE_ROUNDING is set, so double rounding is used throughout)
------------------------------------------------------------------------------
  Requantization primitive, from gemmlowp:
      MBQM(x, mult, shift) = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(x << max(shift, 0), mult), max(-shift, 0))
      (mult, shift) = QuantizeMultiplier(real)  so that  real ~= mult * 2^(shift-31)

  CONV_2D / DEPTHWISE_CONV_2D, per-output-channel symmetric int8 weights:
      acc[c] = sum_k (x_q - x_zp) * w_q[c, k] + bias_q[c]                      int32
      y_q[c] = clamp(MBQM(acc[c], mult[c], shift[c]) + y_zp, act_min, act_max)  int8
      real multiplier = x_scale * w_scale[c] / y_scale
      The fused ReLU6 is the clamp: act_min = y_zp + round(0 / y_scale),
      act_max = min(127, y_zp + round(6 / y_scale)). For y_scale = 6/255,
      y_zp = -128 this is the full int8 range [-128, 127].

  MEAN over H, W (global average pool):
      (mult, shift) = QuantizeMultiplier(x_scale / y_scale), then 1/(H*W) is
      folded in:  mult = (mult << n_shift) // (H*W),  shift -= n_shift
      y_q = clamp(MBQM(sum_hw(x_q) - x_zp * H*W, mult, shift) + y_zp)

  SOFTMAX (output scale 1/256, zero point -128, as TFLM requires):
      gemmlowp fixed-point exp on Q5.26 differences, Q12.19 accumulation,
      Newton-Raphson reciprocal; see Int8Softmax.

Every integer tensor is held as torch.int64 so intermediate products cannot
overflow; the values themselves stay inside int32 exactly as in the C++.

Verification against the coralnpu .tflite (cat image, builtin LiteRT kernels)
------------------------------------------------------------------------------
  * all 27 conv activations (stem, 13 x dw/pw) are bit-exact;
  * fc and softmax are bit-exact when fed the interpreter's own inputs;
  * avgpool differs by +-1 LSB on some channels: desktop LiteRT uses
    optimized_integer_ops::Mean, TFLM (what coralnpu runs) uses the
    QuantizedMeanOrSum formula implemented here. The coralnpu README notes
    the same TFLM-vs-TFLite MEAN/SOFTMAX discrepancy.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mobilenet_v1_fp32 import MobileNetV1, SamePad3x3, make_divisible

INT8_MIN, INT8_MAX = -128, 127
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


# ---------------------------------------------------------------------------
# gemmlowp / TFLite fixed-point primitives (int64 tensors holding int32 values)
# ---------------------------------------------------------------------------

def _i64(x):
    return torch.as_tensor(x, dtype=torch.int64)


def tflite_round(v):
    """std::round / TfLiteRound: round half away from zero."""
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def quantize_multiplier(real_multiplier):
    """
    TFLite QuantizeMultiplier: real ~= mult * 2^(shift - 31), mult an int32 in
    [2^30, 2^31) (Q31 fixed point), shift an int. real_multiplier is a double.
    """
    if real_multiplier == 0.0:
        return 0, 0
    q, shift = math.frexp(real_multiplier)               # real = q * 2^shift, 0.5 <= |q| < 1
    q_fixed = tflite_round(q * (1 << 31))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:                                       # flush tiny multipliers to zero
        shift, q_fixed = 0, 0
    return q_fixed, shift


def saturating_rounding_doubling_high_mul(a, b):
    """gemmlowp SRDHM: round((a * b) / 2^31), saturating INT32_MIN * INT32_MIN."""
    a, b = _i64(a), _i64(b)
    ab = a * b
    nudge = torch.where(ab >= 0, _i64(1 << 30), _i64(1 - (1 << 30)))
    hi = torch.div(ab + nudge, 1 << 31, rounding_mode="trunc")
    return torch.where((a == INT32_MIN) & (b == INT32_MIN), _i64(INT32_MAX), hi)


def rounding_divide_by_pot(x, exponent):
    """gemmlowp RoundingDivideByPOT: x / 2^exponent, rounded to nearest, ties away from zero."""
    x, exponent = _i64(x), _i64(exponent)
    mask = torch.bitwise_left_shift(_i64(1), exponent) - 1
    remainder = torch.bitwise_and(x, mask)
    threshold = torch.bitwise_right_shift(mask, 1) + (x < 0).to(torch.int64)
    return torch.bitwise_right_shift(x, exponent) + (remainder > threshold).to(torch.int64)


def saturating_rounding_multiply_by_pot(x, exponent):
    """gemmlowp SaturatingRoundingMultiplyByPOT: x * 2^exponent; saturating for
    exponent > 0, rounding for exponent < 0."""
    x = _i64(x)
    if exponent == 0:
        return x
    if exponent < 0:
        return rounding_divide_by_pot(x, -exponent)
    threshold = (1 << (31 - exponent)) - 1
    y = torch.bitwise_left_shift(x, exponent)
    y = torch.where(x > threshold, _i64(INT32_MAX), y)
    y = torch.where(x < -threshold, _i64(INT32_MIN), y)
    return y


def rounding_half_sum(a, b):
    """gemmlowp RoundingHalfSum: (a + b) / 2 rounded away from zero."""
    s = _i64(a) + _i64(b)
    sign = torch.where(s >= 0, _i64(1), _i64(-1))
    return torch.div(s + sign, 2, rounding_mode="trunc")


def multiply_by_quantized_multiplier(x, multiplier, shift):
    """TFLite MultiplyByQuantizedMultiplier (double rounding): x * mult * 2^(shift-31)."""
    shift = _i64(shift)
    left = shift.clamp(min=0)
    right = (-shift).clamp(min=0)
    x = torch.bitwise_left_shift(_i64(x), left)
    return rounding_divide_by_pot(saturating_rounding_doubling_high_mul(x, multiplier), right)


def dequantize(q, scale, zero_point):
    """real = scale * (q - zero_point), as float32."""
    return (q.to(torch.float32) - float(zero_point)) * float(scale)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class Int8Conv2d(nn.Module):
    """
    int8 convolution as executed by TFLite CONV_2D / DEPTHWISE_CONV_2D with
    per-output-channel symmetric weights and an optionally fused ReLU6.
    Replaces the float model's Conv2d + BatchNorm2d + ReLU6 triple: BN is
    already folded into the int32 bias in the .tflite.

    3x3 kernels use TF 'SAME' padding via SamePad3x3 (padding after the zero
    point is subtracted, so padded taps contribute 0, as in TFLite).
    Call set_quant() (see utils.load_tflite_weights_int8) before forward().
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1, relu6=True):
        super().__init__()
        self.in_ch, self.out_ch, self.kernel_size = in_ch, out_ch, kernel_size
        self.stride, self.groups, self.relu6 = stride, groups, relu6
        self.pad = SamePad3x3(stride) if kernel_size == 3 else nn.Identity()
        self.register_buffer("weight", torch.zeros(out_ch, in_ch // groups, kernel_size, kernel_size, dtype=torch.int8))
        self.register_buffer("bias", torch.zeros(out_ch, dtype=torch.int32))
        self.register_buffer("weight_scale", torch.ones(out_ch))
        self.register_buffer("in_scale", torch.tensor(1.0))
        self.register_buffer("in_zp", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("out_scale", torch.tensor(1.0))
        self.register_buffer("out_zp", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("multiplier", torch.zeros(out_ch, dtype=torch.int64))   # Q31, per channel
        self.register_buffer("shift", torch.zeros(out_ch, dtype=torch.int64))        # per channel
        self.register_buffer("act_min", torch.tensor(INT8_MIN, dtype=torch.int64))
        self.register_buffer("act_max", torch.tensor(INT8_MAX, dtype=torch.int64))

    def set_quant(self, weight, weight_scale, bias, in_scale, in_zp, out_scale, out_zp):
        """
        weight       int8  [Cout, Cin/groups, kh, kw]
        weight_scale float [Cout]        (zero points are 0: symmetric)
        bias         int32 [Cout]        (scale = in_scale * weight_scale, zero point 0)
        in/out scale, zero point: quantization of the input / output activation.
        Derives the per-channel requantization multiplier/shift and the
        activation clamp exactly as TFLM's conv Prepare() does.
        """
        weight, bias, weight_scale = torch.as_tensor(weight), torch.as_tensor(bias), torch.as_tensor(weight_scale)
        assert weight.shape == self.weight.shape, f"{tuple(self.weight.shape)} vs {tuple(weight.shape)}"
        assert bias.shape == self.bias.shape and weight_scale.shape == self.weight_scale.shape
        with torch.no_grad():
            self.weight.copy_(weight.to(torch.int8))
            self.bias.copy_(bias.to(torch.int32))
            self.weight_scale.copy_(weight_scale.to(torch.float32))
            self.in_scale.fill_(float(in_scale))
            self.in_zp.fill_(int(in_zp))
            self.out_scale.fill_(float(out_scale))
            self.out_zp.fill_(int(out_zp))
            # scales are float32 in the .tflite; TFLM promotes them to double and multiplies there
            in_scale, out_scale = float(self.in_scale), float(self.out_scale)
            for c in range(self.out_ch):
                m, s = quantize_multiplier(in_scale * float(self.weight_scale[c]) / out_scale)
                self.multiplier[c], self.shift[c] = m, s
            # CalculateActivationRangeQuantized
            if self.relu6:
                self.act_min.fill_(max(INT8_MIN, int(out_zp) + tflite_round(0.0 / out_scale)))
                self.act_max.fill_(min(INT8_MAX, int(out_zp) + tflite_round(6.0 / out_scale)))
            else:
                self.act_min.fill_(INT8_MIN)
                self.act_max.fill_(INT8_MAX)

    def forward(self, x):
        assert x.dtype == torch.int8, f"expected int8 input, got {x.dtype}"
        x = self.pad(x.to(torch.int64) - self.in_zp)
        # int8 x int8 products summed over at most 3*3*3 or 256 taps stay far below
        # 2^53, so the float64 conv is exact integer arithmetic.
        acc = F.conv2d(x.to(torch.float64), self.weight.to(torch.float64),
                       stride=self.stride, groups=self.groups).to(torch.int64)
        acc = acc + self.bias.to(torch.int64).view(1, -1, 1, 1)
        y = multiply_by_quantized_multiplier(acc, self.multiplier.view(1, -1, 1, 1),
                                             self.shift.view(1, -1, 1, 1)) + self.out_zp
        return y.clamp(int(self.act_min), int(self.act_max)).to(torch.int8)

    def dequantize_output(self, y_q):
        return dequantize(y_q, self.out_scale, self.out_zp)

    def extra_repr(self):
        kind = "depthwise " if self.groups > 1 and self.groups == self.in_ch else ""
        return (f"{kind}{self.in_ch} -> {self.out_ch}, kernel={self.kernel_size}, stride={self.stride}, "
                f"relu6={self.relu6}, in(s={float(self.in_scale):.6g}, zp={int(self.in_zp)}), "
                f"out(s={float(self.out_scale):.6g}, zp={int(self.out_zp)}), int8 weight, int32 bias")


class Int8DepthwiseSeparable(nn.Module):
    """3x3 depthwise int8 conv (carries the stride) + 1x1 pointwise int8 conv, both with fused ReLU6."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = Int8Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch)
        self.pw = Int8Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pw(self.dw(x))


class Int8GlobalAvgPool(nn.Module):
    """
    TFLite MEAN over H, W on int8 with requantization (TFLM QuantizedMeanOrSum):
        (mult, shift) = QuantizeMultiplier(in_scale / out_scale)
        n_shift = min(floor(log2(H*W)), 32, 31 + shift)
        mult = (mult << n_shift) // (H*W);  shift -= n_shift
        y = clamp(MBQM(sum_hw(x_q) - in_zp * H*W, mult, shift) + out_zp, -128, 127)
    Output keeps the spatial dims: [N, C, 1, 1].
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("in_scale", torch.tensor(1.0))
        self.register_buffer("in_zp", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("out_scale", torch.tensor(1.0))
        self.register_buffer("out_zp", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("multiplier", torch.tensor(0, dtype=torch.int64))   # before the 1/(H*W) fold
        self.register_buffer("shift", torch.tensor(0, dtype=torch.int64))

    def set_quant(self, in_scale, in_zp, out_scale, out_zp):
        with torch.no_grad():
            self.in_scale.fill_(float(in_scale))
            self.in_zp.fill_(int(in_zp))
            self.out_scale.fill_(float(out_scale))
            self.out_zp.fill_(int(out_zp))
            m, s = quantize_multiplier(float(self.in_scale) / float(self.out_scale))
            self.multiplier.fill_(m)
            self.shift.fill_(s)

    def forward(self, x):
        assert x.dtype == torch.int8, f"expected int8 input, got {x.dtype}"
        n = x.shape[2] * x.shape[3]
        mult, shift = int(self.multiplier), int(self.shift)
        n_shift = min(n.bit_length() - 1, 32, 31 + shift)      # 63 - clz64(n), clamped
        mult = (mult << n_shift) // n                           # mult > 0: same as C++ truncation
        shift = shift - n_shift
        acc = x.to(torch.int64).sum(dim=(2, 3), keepdim=True) - self.in_zp * n
        y = multiply_by_quantized_multiplier(acc, mult, shift) + self.out_zp
        return y.clamp(INT8_MIN, INT8_MAX).to(torch.int8)

    def dequantize_output(self, y_q):
        return dequantize(y_q, self.out_scale, self.out_zp)

    def extra_repr(self):
        return (f"in(s={float(self.in_scale):.6g}, zp={int(self.in_zp)}), "
                f"out(s={float(self.out_scale):.6g}, zp={int(self.out_zp)})")


class Int8ClassifierHead(nn.Module):
    """
    MobileNet V1 head as it appears in the .tflite:
        MEAN over H,W (int8, requantized) -> 1x1 CONV_2D (the classifier, no
        activation) -> flatten to [N, num_classes].
    fc is an Int8Conv2d with a 1x1 kernel on a [N, C, 1, 1] input; its weight is
    [num_classes, C, 1, 1] where the float model's Linear has [num_classes, C].
    Output is int8 logits with fc.out_scale / fc.out_zp; apply Int8Softmax for
    probabilities.
    """
    def __init__(self, in_ch, num_classes=1000):
        super().__init__()
        self.avgpool = Int8GlobalAvgPool()
        self.fc = Int8Conv2d(in_ch, num_classes, kernel_size=1, relu6=False)

    def forward(self, x):
        x = self.avgpool(x)          # [N, C, 1, 1] int8
        x = self.fc(x)               # [N, num_classes, 1, 1] int8
        return torch.flatten(x, 1)   # [N, num_classes] int8


class Int8Softmax(nn.Module):
    """
    TFLite/TFLM int8 SOFTMAX (reference_ops::Softmax<int8_t, int8_t>), which
    is pure fixed-point arithmetic from gemmlowp:

      Prepare:  input_multiplier, input_left_shift = QuantizeMultiplier(
                    min(beta * in_scale * 2^(31-5), 2^31 - 1))
                diff_min = -floor((2^5 - 1) * 2^(31-5) / 2^input_left_shift)
      Per row:  d      = x_q - max(x_q)                                (<= 0)
                d_q5   = SRDHM(d << input_left_shift, input_multiplier)   Q5.26
                e_q0   = exp_on_negative_values(d_q5)                     Q0.31
                sum    = sum over d >= diff_min of RoundingDivideByPOT(e_q0, 12)   Q12.19
                recip, bits_over_unit = GetReciprocal(sum, 12)           Q0.31
                y      = RoundingDivideByPOT(SRDHM(recip, e_q0), bits_over_unit + 31 - 8) - 128
                entries with d < diff_min are -128 (probability 0).
    Output quantization is fixed at scale 1/256, zero point -128.
    """
    SCALED_DIFF_INTEGER_BITS = 5
    ACCUMULATION_INTEGER_BITS = 12

    def __init__(self):
        super().__init__()
        self.register_buffer("in_scale", torch.tensor(1.0))
        self.register_buffer("in_zp", torch.tensor(0, dtype=torch.int64))      # not used by the math
        self.register_buffer("out_scale", torch.tensor(1.0 / 256))
        self.register_buffer("out_zp", torch.tensor(INT8_MIN, dtype=torch.int64))
        self.register_buffer("input_multiplier", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("input_left_shift", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("diff_min", torch.tensor(0, dtype=torch.int64))

    def set_quant(self, in_scale, in_zp, out_scale, out_zp, beta=1.0):
        assert int(out_zp) == INT8_MIN and float(out_scale) == 1.0 / 256, \
            f"TFLM int8 softmax requires out scale 1/256, zp -128; got {out_scale}, {out_zp}"
        with torch.no_grad():
            self.in_scale.fill_(float(in_scale))
            self.in_zp.fill_(int(in_zp))
            # PreprocessSoftmaxScaling
            bits = self.SCALED_DIFF_INTEGER_BITS
            real = min(beta * float(self.in_scale) * (1 << (31 - bits)), (1 << 31) - 1.0)
            m, left_shift = quantize_multiplier(real)
            assert left_shift >= 0
            self.input_multiplier.fill_(m)
            self.input_left_shift.fill_(left_shift)
            # -CalculateInputRadius
            radius = math.floor(1.0 * ((1 << bits) - 1) * (1 << (31 - bits)) / (1 << left_shift))
            self.diff_min.fill_(-radius)

    # -- gemmlowp exp / reciprocal, on int64 tensors of int32 raw values --------

    @staticmethod
    def _exp_on_interval_between_negative_one_quarter_and_0_excl(a):
        """exp(a) for a in [-1/4, 0), Q0.31 in and out: Taylor expansion around -1/8."""
        srdhm, rdbp = saturating_rounding_doubling_high_mul, rounding_divide_by_pot
        constant_term = 1895147668           # exp(-1/8)
        constant_1_over_3 = 715827883        # 1/3
        x = a + (1 << 28)                    # x = a + 1/8
        x2 = srdhm(x, x)
        x3 = srdhm(x2, x)
        x4 = srdhm(x2, x2)
        x4_over_4 = rdbp(x4, 2)
        x4_over_24_plus_x3_over_6_plus_x2_over_2 = rdbp(srdhm(x4_over_4 + x3, constant_1_over_3) + x2, 1)
        return constant_term + srdhm(constant_term, x + x4_over_24_plus_x3_over_6_plus_x2_over_2)

    @classmethod
    def _exp_on_negative_values(cls, a):
        """exp(a) for a <= 0 given in Q5.26 (SCALED_DIFF_INTEGER_BITS); result Q0.31."""
        integer_bits = cls.SCALED_DIFF_INTEGER_BITS
        fractional_bits = 31 - integer_bits
        srdhm = saturating_rounding_doubling_high_mul
        one_quarter = 1 << (fractional_bits - 2)
        mask = one_quarter - 1
        a_mod_quarter_minus_one_quarter = torch.bitwise_and(a, mask) - one_quarter
        result = cls._exp_on_interval_between_negative_one_quarter_and_0_excl(
            saturating_rounding_multiply_by_pot(a_mod_quarter_minus_one_quarter, integer_bits))
        remainder = a_mod_quarter_minus_one_quarter - a
        # Q0.31 constants for exp(-2^k), k = -2..4 (barrel shifter over the integer part)
        for exponent, multiplier in ((-2, 1672461947), (-1, 1302514674), (0, 790015084), (1, 290630308),
                                     (2, 39332535), (3, 720401), (4, 242)):
            if integer_bits > exponent:
                bit = 1 << (fractional_bits + exponent)
                result = torch.where(torch.bitwise_and(remainder, bit) != 0, srdhm(result, multiplier), result)
        return torch.where(a == 0, _i64(INT32_MAX), result)    # exp(0) = 1.0 = INT32_MAX in Q0.31

    @staticmethod
    def _one_over_one_plus_x_for_x_in_0_1(a):
        """1 / (1 + a) for a in (0, 1), Q0.31: 3 Newton-Raphson steps in Q2.29."""
        srdhm, srmbp = saturating_rounding_doubling_high_mul, saturating_rounding_multiply_by_pot
        half_denominator = rounding_half_sum(a, INT32_MAX)              # (a + 1) / 2, Q0.31
        x = 1515870810 + srdhm(half_denominator, -1010580540)          # 48/17 - 32/17 * hd, Q2.29
        for _ in range(3):
            half_denominator_times_x = srdhm(half_denominator, x)
            one_minus_half_denominator_times_x = (1 << 29) - half_denominator_times_x
            x = x + srmbp(srdhm(x, one_minus_half_denominator_times_x), 2)
        return srmbp(x, 1)      # Rescale<0>(ExactMulByPot<-1>(x)): Q2.29 -> value/2 as Q1.30 -> Q0.31

    @classmethod
    def _get_reciprocal(cls, x):
        """TFLite GetReciprocal for x in Q12.19 (x > 0): returns (1/x as Q0.31, num_bits_over_unit)."""
        x = int(x)
        headroom_plus_one = 32 - x.bit_length()                          # clz(uint32(x))
        num_bits_over_unit = cls.ACCUMULATION_INTEGER_BITS - headroom_plus_one
        shifted_sum_minus_one = (x << headroom_plus_one) - (1 << 31)     # in [0, 2^31)
        return cls._one_over_one_plus_x_for_x_in_0_1(_i64(shifted_sum_minus_one)), num_bits_over_unit

    def forward(self, x):
        assert x.dtype == torch.int8 and x.dim() == 2, "expected int8 logits [N, num_classes]"
        srdhm, rdbp = saturating_rounding_doubling_high_mul, rounding_divide_by_pot
        x = x.to(torch.int64)
        out = torch.full_like(x, INT8_MIN)
        for i in range(x.shape[0]):
            diff = x[i] - x[i].max()
            valid = diff >= self.diff_min
            diff_rescaled = srdhm(torch.bitwise_left_shift(diff, self.input_left_shift), self.input_multiplier)
            exp_q0 = self._exp_on_negative_values(diff_rescaled)
            sum_of_exps = rdbp(exp_q0, self.ACCUMULATION_INTEGER_BITS)[valid].sum()
            shifted_scale, num_bits_over_unit = self._get_reciprocal(sum_of_exps)
            exponent = num_bits_over_unit + 31 - 8
            assert 0 <= exponent <= 31
            unsat = rdbp(srdhm(shifted_scale, exp_q0), exponent)
            y = (unsat + INT8_MIN).clamp(INT8_MIN, INT8_MAX)
            out[i] = torch.where(valid, y, out[i])
        return out.to(torch.int8)

    def dequantize_output(self, y_q):
        return dequantize(y_q, self.out_scale, self.out_zp)

    def extra_repr(self):
        return (f"in(s={float(self.in_scale):.6g}, zp={int(self.in_zp)}), out(s=1/256, zp=-128), "
                f"input_multiplier={int(self.input_multiplier)}, input_left_shift={int(self.input_left_shift)}, "
                f"diff_min={int(self.diff_min)}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MobileNetV1Int8(nn.Module):
    """
    MobileNet V1 with width multiplier alpha in fully-int8 arithmetic.

    Same topology as mobilenet_v1_fp32.MobileNetV1 (see its docstring for the
    architecture and shapes): a 3x3 stride-2 stem, 13 depthwise-separable
    blocks from MobileNetV1.BLOCK_SPECS scaled by alpha, global average pool
    and a classifier. Differences:
      * every Conv2d + BatchNorm2d + ReLU6 is one Int8Conv2d (int8 weights,
        int32 bias with BN folded in, fused ReLU6 as the output clamp);
      * the classifier is a 1x1 Int8Conv2d, as in the .tflite;
      * forward() takes an int8 NCHW tensor (see quantize_input) and returns
        int8 logits [N, num_classes]; self.softmax turns them into int8
        probabilities (scale 1/256, zero point -128).
    All weights and quantization parameters come from utils.load_tflite_weights_int8.

    Args:
        num_classes: size of the classifier output.
        alpha:       width multiplier in (0, 1].
    """
    BLOCK_SPECS = MobileNetV1.BLOCK_SPECS

    def __init__(self, num_classes=1000, alpha=0.25):
        super().__init__()
        in_ch = make_divisible(32 * alpha)                                   # 8 for alpha=0.25
        self.stem = Int8Conv2d(3, in_ch, kernel_size=3, stride=2)

        blocks = []
        for base_out_ch, stride in self.BLOCK_SPECS:
            out_ch = make_divisible(base_out_ch * alpha)
            blocks.append(Int8DepthwiseSeparable(in_ch=in_ch, out_ch=out_ch, stride=stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.head = Int8ClassifierHead(in_ch=in_ch, num_classes=num_classes)   # in_ch == 256
        self.softmax = Int8Softmax()

    def quantize_input(self, image):
        """
        float NCHW image in the units the .tflite expects (raw [0, 255] pixels
        for the coralnpu model) -> int8 NCHW: q = round(x / in_scale) + in_zp.
        """
        scale, zp = float(self.stem.in_scale), int(self.stem.in_zp)
        return (torch.round(image / scale) + zp).clamp(INT8_MIN, INT8_MAX).to(torch.int8)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)          # int8 logits [N, num_classes]

    def dequantize_logits(self, logits_q):
        return self.head.fc.dequantize_output(logits_q)

    def probabilities(self, logits_q):
        """int8 logits -> int8 softmax -> float32 probabilities (multiples of 1/256)."""
        return self.softmax.dequantize_output(self.softmax(logits_q))
