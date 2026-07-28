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

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>

#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/opt/rvv_opt.h"
#include "sw/utils/utils.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tests/npusim_examples/mobilenet/mobilenet_v1_025_224_int8_real.h"

namespace {
using MobilenetOpResolver = tflite::MicroMutableOpResolver<10>;

// Per-node cycle profiler backed by the mcycle CSR (modeled by npusim).
// The interpreter calls BeginEvent/EndEvent around every op invocation with
// the op name as tag (available because TF_LITE_STRIP_ERROR_STRINGS is not
// set in the default build config).
class CycleProfiler : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char* tag) override {
    if (count_ >= kMaxEvents) return kMaxEvents - 1;
    tags_[count_] = tag;
    starts_[count_] = mcycle_read();
    return count_++;
  }
  void EndEvent(uint32_t handle) override {
    cycles_[handle] = static_cast<uint32_t>(mcycle_read() - starts_[handle]);
  }
  void PrintSummary() const {
    printf("== per-node cycles ==\n");
    uint32_t total = 0;
    for (int i = 0; i < count_; ++i) {
      printf("node %2d %-20s %lu\n", i, tags_[i] ? tags_[i] : "?",
             static_cast<unsigned long>(cycles_[i]));
      total += cycles_[i];
    }
    printf("== per-op totals ==\n");
    bool done[kMaxEvents] = {};
    for (int i = 0; i < count_; ++i) {
      if (done[i] || !tags_[i]) continue;
      uint32_t sum = 0;
      int n = 0;
      for (int j = i; j < count_; ++j) {
        if (!done[j] && tags_[j] && strcmp(tags_[i], tags_[j]) == 0) {
          sum += cycles_[j];
          n += 1;
          done[j] = true;
        }
      }
      printf("op %-20s x%2d %10lu cycles (%lu%%)\n", tags_[i], n,
             static_cast<unsigned long>(sum),
             static_cast<unsigned long>(total ? (sum / (total / 100)) : 0));
    }
    printf("total profiled cycles %lu\n", static_cast<unsigned long>(total));
  }

 private:
  static constexpr int kMaxEvents = 64;
  const char* tags_[kMaxEvents] = {};
  uint64_t starts_[kMaxEvents] = {};
  uint32_t cycles_[kMaxEvents] = {};
  int count_ = 0;
};
using coralnpu_v2::opt::litert_micro::Register_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D;
TfLiteStatus RegisterOps(MobilenetOpResolver& op_resolver) {
  // NOTE: The original optimized Conv_3_3_3_8 / Conv_1x1_Pointwise kernels
  // misclassify with this real ImageNet model (top-1 becomes garbage) even
  // though they match the reference bit-for-bit on the dummy model. The
  // dispatch in sw/opt/litert-micro/conv.cc now routes those two shapes to
  // the V2 rewrites (Conv_3_3_3_8_V2 / Conv_1x1_Pointwise_V2), which are
  // verified correct on this model.
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D(Register_CONV_2D()));
  TF_LITE_ENSURE_STATUS(
      op_resolver.AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddShape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPack());
  return kTfLiteOk;
}

// Full ImageNet (ILSVRC-2012) class count, as used by MobileNet V1.
constexpr size_t kNumClasses = 1000;
}  // namespace

extern "C" {
// aligned(16)
constexpr size_t kTensorArenaSize = 4 * 1024 * 1024;
int8_t inference_status = -1;
uint8_t inference_input[224 * 224 * 3]
    __attribute__((section(".data"), aligned(16)));
int8_t inference_output[kNumClasses]
    __attribute__((section(".data"), aligned(16)));
uint8_t tensor_arena[kTensorArenaSize]
    __attribute__((section(".extdata"), aligned(16)));
}

int main(int argc, char** argv) {
  const tflite::Model* model =
      tflite::GetModel(g_mobilenet_v1_025_224_int8_real_model_data);
  MobilenetOpResolver op_resolver;
  RegisterOps(op_resolver);
  printf("Halted after op resolver\n");
  static CycleProfiler profiler;
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize,
                                       /*resource_variables=*/nullptr,
                                       &profiler);
  printf("Halted after Interpreter setup\n");
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("Error during AllocateTensors\n");
    return -1;
  }
  TfLiteTensor* input = interpreter.input(0);
  if (input == nullptr) {
    printf("Error getting input tensor\n");
    return -1;
  }
  coralnpu_v2::opt::Memcpy(input->data.data, inference_input, input->bytes);

  if (interpreter.Invoke() != kTfLiteOk) {
    printf("Error during Invoke\n");
    return -1;
  }

  TfLiteTensor* output = interpreter.output(0);
  if (output == nullptr) {
    printf("Error getting output tensor\n");
    return -1;
  }
  if (output->bytes != kNumClasses) {
    printf("Unexpected output size\n");
    return -1;
  }
  coralnpu_v2::opt::Memcpy(inference_output, output->data.data, kNumClasses);
  profiler.PrintSummary();
  printf("Invoke successful\n");
  inference_status = 0;
  return 0;
}
