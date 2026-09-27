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

load("@coralnpu_hw//third_party/python:requirements.bzl", "requirement")
load("//rules:coco_tb.bzl", "cocotb_test_suite")
load("//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")
load("//rules:utils.bzl", "generate_cc_arrays")

# Shared HTIF server + progress/wedge diagnostics for the test modules.
py_library(
    name = "htif_util",
    srcs = ["htif_util.py"],
    deps = [
        "//coralnpu_test_utils:core_mini_axi_sim_interface",
        requirement("numpy"),
    ],
)

# Runs the full ImageNet MobileNet (real weights, cat image) on the RTL.
# Very long-running (~130M+ cycles), hence manual + enormous.
#
#   bazel run //tests/cocotb/imagenet:imagenet
#
# BEGIN_TESTCASES_FOR_imagenet
IMAGENET_TESTCASES = [
    "core_mini_rvv_imagenet_mobilenet",
]
# END_TESTCASES_FOR_imagenet

cocotb_test_suite(
    name = "imagenet",
    simulators = ["verilator"],
    testcases = IMAGENET_TESTCASES,
    testcases_vname = "IMAGENET_TESTCASES",
    tests_kwargs = {
        "waves": False,
        "tags": ["manual"],
        "hdl_toplevel": "RvvCoreMiniHighmemAxi",
        "seed": "42",
        "size": "enormous",
        "test_module": ["imagenet.py"],
        "deps": [
            ":htif_util",
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        "data": [
            "//tests/npusim_examples/mobilenet:images_224x224x3/cat_224x224_real.npy",
            "//tests/npusim_examples/mobilenet:labels/imagenet_labels.txt",
            "//tests/npusim_examples/mobilenet:run_full_mobilenet_v1_real_binary_elf",
        ],
    },
    verilator_model = "//tests/cocotb:rvv_core_mini_highmem_axi_model",
)

# ---------------------------------------------------------------------------
# Minimal repro of the STRIDED_SLICE wedge: just MobileNet's classifier-head
# op chain (MEAN -> CONV_2D -> SHAPE -> STRIDED_SLICE -> PACK -> RESHAPE ->
# SOFTMAX). Minutes of RTL sim instead of hours. See head_repro/.
#
#   bazel run //tests/cocotb/imagenet:npusim_run_head   (npusim sanity)
#   bazel run //tests/cocotb/imagenet:head_repro        (RTL repro)
# ---------------------------------------------------------------------------

generate_cc_arrays(
    name = "head_int8_cc",
    src = "head_repro/head_int8.tflite",
    out = "head_int8.cc",
)

generate_cc_arrays(
    name = "head_int8_h",
    src = "head_repro/head_int8.tflite",
    out = "head_int8.h",
)

cc_library(
    name = "head_int8_lib",
    srcs = ["head_int8_cc"],
    hdrs = ["head_int8_h"],
)

generate_cc_arrays(
    name = "reshape_int8_cc",
    src = "head_repro/reshape_int8.tflite",
    out = "reshape_int8.cc",
)

generate_cc_arrays(
    name = "reshape_int8_h",
    src = "head_repro/reshape_int8.tflite",
    out = "reshape_int8.h",
)

cc_library(
    name = "reshape_int8_lib",
    srcs = ["reshape_int8_cc"],
    hdrs = ["reshape_int8_h"],
)

coralnpu_v2_binary(
    name = "run_head_repro_binary",
    srcs = ["run_head_repro.cc"],
    dtcm_size_kbytes = 1024,  # Highmem
    itcm_size_kbytes = 1024,  # Highmem
    semihosting = True,
    deps = [
        ":head_int8_lib",
        "//sw/opt:rvv_opt",
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "//sw/utils:utils",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro:micro_log",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/micro:system_setup",
    ],
)

coralnpu_v2_binary(
    name = "run_reshape_repro_binary",
    srcs = ["run_reshape_repro.cc"],
    dtcm_size_kbytes = 1024,  # Highmem
    itcm_size_kbytes = 1024,  # Highmem
    semihosting = True,
    deps = [
        ":reshape_int8_lib",
        "//sw/opt:rvv_opt",
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "//sw/utils:utils",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro:micro_log",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/micro:system_setup",
    ],
)

# Minimal bare-metal repro: vle8/vse8 with vl=0 deadlock the pipeline.
coralnpu_v2_binary(
    name = "run_vl0_repro_binary",
    srcs = ["run_vl0_repro.cc"],
    dtcm_size_kbytes = 1024,  # Highmem
    itcm_size_kbytes = 1024,  # Highmem
    semihosting = True,
)

py_binary(
    name = "npusim_run_head",
    srcs = ["npusim_run_head.py"],
    data = [
        "head_repro/head_golden.npy",
        "head_repro/head_input.npy",
        "head_repro/reshape_golden.npy",
        "head_repro/reshape_input.npy",
        ":run_head_repro_binary",
        ":run_reshape_repro_binary",
        ":run_vl0_repro_binary",
    ],
    deps = [
        "//sw/coralnpu_sim:coralnpu_v2_sim_utils_lib",
        "@bazel_tools//tools/python/runfiles",
        requirement("numpy"),
    ],
)

# BEGIN_TESTCASES_FOR_head_repro
HEAD_REPRO_TESTCASES = [
    "core_mini_rvv_head_repro",
    "core_mini_rvv_reshape_repro",
    "core_mini_rvv_vl0_repro",
]
# END_TESTCASES_FOR_head_repro

cocotb_test_suite(
    name = "head_repro",
    simulators = ["verilator"],
    testcases = HEAD_REPRO_TESTCASES,
    testcases_vname = "HEAD_REPRO_TESTCASES",
    tests_kwargs = {
        "waves": False,
        "tags": ["manual"],
        "hdl_toplevel": "RvvCoreMiniHighmemAxi",
        "seed": "42",
        "size": "enormous",
        "test_module": ["head_repro.py"],
        "deps": [
            ":htif_util",
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        "data": [
            "head_repro/head_golden.npy",
            "head_repro/head_input.npy",
            "head_repro/reshape_golden.npy",
            "head_repro/reshape_input.npy",
            ":run_head_repro_binary.elf",
            ":run_reshape_repro_binary.elf",
            ":run_vl0_repro_binary.elf",
        ],
    },
    verilator_model = "//tests/cocotb:rvv_core_mini_highmem_axi_model",
)

# Waveform-capture variant of the reshape repro: trace-enabled model, run
# capped just past the wedge onset (~72k cycles) so the VCD stays small.
# The dump lands in the test's sim_build directory as dump.vcd.
#
#   bazel run //tests/cocotb/imagenet:head_repro_waves_core_mini_rvv_reshape_repro
#
# BEGIN_TESTCASES_FOR_head_repro_waves
HEAD_REPRO_WAVES_TESTCASES = [
    "core_mini_rvv_reshape_repro",
]
# END_TESTCASES_FOR_head_repro_waves

cocotb_test_suite(
    name = "head_repro_waves",
    simulators = ["verilator"],
    testcases = HEAD_REPRO_WAVES_TESTCASES,
    testcases_vname = "HEAD_REPRO_WAVES_TESTCASES",
    tests_kwargs = {
        "waves": False,
        "tags": ["manual"],
        "hdl_toplevel": "RvvCoreMiniHighmemAxi",
        "seed": "42",
        "size": "enormous",
        "test_module": ["head_repro.py"],
        "test_args": ["--trace"],
        "extra_env": ["WEDGE_MAX_CYCLES=150000"],
        "deps": [
            ":htif_util",
            "//coralnpu_test_utils:core_mini_axi_sim_interface",
            "//coralnpu_test_utils:sim_test_fixture",
            "@rules_python//python/runfiles",
        ],
        "data": [
            "head_repro/head_golden.npy",
            "head_repro/head_input.npy",
            "head_repro/reshape_golden.npy",
            "head_repro/reshape_input.npy",
            ":run_head_repro_binary.elf",
            ":run_reshape_repro_binary.elf",
        ],
    },
    verilator_model = "//tests/cocotb:rvv_core_mini_highmem_axi_model_trace",
)
