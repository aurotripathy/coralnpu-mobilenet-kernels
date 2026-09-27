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

// Smallest repro of the RTL wedge at STRIDED_SLICE: runs reshape_int8.tflite
// (SHAPE -> STRIDED_SLICE -> PACK -> RESHAPE -> SOFTMAX, input int8[1,1,1000],
// no MEAN/CONV). See run_head_repro.cc for the next-larger repro and
// tests/npusim_examples/mobilenet for the original full-model failure.

#include <stdint.h>
#include <stdio.h>

#include <cstring>

#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/utils/utils.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tests/cocotb/imagenet/reshape_int8.h"

namespace {
using HeadOpResolver = tflite::MicroMutableOpResolver<10>;

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
    for (int i = 0; i < count_; ++i) {
      printf("node %2d %-20s %lu\n", i, tags_[i] ? tags_[i] : "?",
             static_cast<unsigned long>(cycles_[i]));
    }
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

// Same registrations as the full-model runner so kernel dispatch matches.
TfLiteStatus RegisterOps(HeadOpResolver& op_resolver) {
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

constexpr size_t kNumClasses = 1000;
constexpr size_t kInputSize = 1 * 1 * 1000;
}  // namespace

extern "C" {
// Same 4MB arena as the wedging runs: the persistent tail (eval tensors,
// node data) then sits at the very top of EXTMEM, preserving the suspect
// access pattern at 0x203fffxx.
constexpr size_t kTensorArenaSize = 4 * 1024 * 1024;
int8_t inference_status = -1;
uint8_t inference_input[kInputSize]
    __attribute__((section(".data"), aligned(16)));
int8_t inference_output[kNumClasses]
    __attribute__((section(".data"), aligned(16)));
uint8_t tensor_arena[kTensorArenaSize]
    __attribute__((section(".extdata"), aligned(16)));
}

int main(int argc, char** argv) {
  const tflite::Model* model = tflite::GetModel(g_reshape_int8_model_data);
  HeadOpResolver op_resolver;
  RegisterOps(op_resolver);
  printf("Reshape repro: op resolver ready\n");
  static CycleProfiler profiler;
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize,
                                       /*resource_variables=*/nullptr,
                                       &profiler);
  printf("Reshape repro: interpreter ready\n");
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("Error during AllocateTensors\n");
    return -1;
  }
  TfLiteTensor* input = interpreter.input(0);
  if (input == nullptr || input->bytes != kInputSize) {
    printf("Error getting input tensor\n");
    return -1;
  }
  std::memcpy(input->data.data, inference_input, input->bytes);

  if (interpreter.Invoke() != kTfLiteOk) {
    printf("Error during Invoke\n");
    return -1;
  }

  TfLiteTensor* output = interpreter.output(0);
  if (output == nullptr || output->bytes != kNumClasses) {
    printf("Error getting output tensor\n");
    return -1;
  }
  std::memcpy(inference_output, output->data.data, kNumClasses);
  profiler.PrintSummary();
  printf("Invoke successful\n");
  inference_status = 0;
  return 0;
}
