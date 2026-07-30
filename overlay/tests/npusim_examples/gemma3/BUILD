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
load("//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")

coralnpu_v2_binary(
    name = "run_gemma3_decode_binary",
    srcs = ["run_gemma3_decode.cc"],
    hdrs = ["gemma3_layout.h"],
    dtcm_size_kbytes = 1024,  # Highmem
    itcm_size_kbytes = 1024,  # Highmem
    stack_size_bytes = 65536,
    semihosting = True,
    visibility = ["//visibility:public"],
    deps = [
        "//sw/utils:utils",
    ],
)

py_binary(
    name = "npusim_run_gemma3",
    srcs = ["npusim_run_gemma3.py"],
    data = [
        ":run_gemma3_decode_binary",
    ],
    deps = [
        "//sw/coralnpu_sim:coralnpu_v2_sim_utils_lib",
        "@bazel_tools//tools/python/runfiles",
        requirement("numpy"),
    ],
)
