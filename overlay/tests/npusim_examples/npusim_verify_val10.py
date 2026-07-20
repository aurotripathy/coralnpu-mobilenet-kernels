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

"""Verifies the NPU conv/depthwise kernels on 10 random ImageNet val images.

Ten images were sampled at random from the ImageNet (ILSVRC-2012) validation
set (one per class, sampled with a fixed seed) and center-cropped to
224x224x3. This driver runs the real int8 MobileNet V1 0.25 on each image and
checks that the model's top-1 prediction matches the image's ground-truth
class. A per-image manifest (val10_manifest.json) carries the ground-truth
class index and human label.

Because MobileNet V1 0.25 is a small (~50% top-1) model, we do not require a
perfect top-1 on every image. The test passes if the top-1 matches on a
majority of images AND the ground-truth class lands in the top-5 for most of
them; any run that is bit-identical garbage (e.g. a broken kernel) collapses
these metrics to ~0, which is the failure signal we care about.
"""

import json

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np

EXPECTED_SHAPE = (224, 224, 3)
NUM_CLASSES = 1000  # ImageNet (ILSVRC-2012)
PKG = 'coralnpu_hw/tests/npusim_examples'

# Pass thresholds (out of 10 images).
MIN_TOP1 = 4
MIN_TOP5 = 6


def load_input_from_npy(npy_path):
    """Loads a 224x224x3 image .npy and returns flat int8 model input."""
    image = np.load(npy_path)
    if image.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected input of shape {EXPECTED_SHAPE}, got {image.shape} "
            f"from {npy_path}")
    if image.dtype == np.uint8:
        image = image.astype(np.int16) - 128
    return image.astype(np.int8).reshape(-1)


def load_imagenet_labels(labels_path):
    """Returns the 1000 ImageNet class labels (drops leading 'background')."""
    with open(labels_path) as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels[1:]


def run_one(elf_file, npy_path, labels, gt):
    """Runs the model on a single image and prints a per-image report.

    Returns the (in_top1, in_top5, cycles) tuple for this image.
    """
    print(f"Running real mobilenet on {labels[gt]}...")
    npu_sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True)
    entry_point, symbol_map = npu_sim.get_elf_entry_and_symbol(
        elf_file, ['inference_status', 'inference_input', 'inference_output'])
    npu_sim.load_program(elf_file, entry_point)

    input_data = load_input_from_npy(npy_path)
    print(f"Writing image to {symbol_map['inference_input']}")
    npu_sim.write_memory(symbol_map['inference_input'], input_data)

    print("Running simulation...", flush=True)
    npu_sim.run()
    npu_sim.wait()
    cycles = npu_sim.get_cycle_count()
    print(f"cycles taken by the simulation {cycles}")

    raw = npu_sim.read_memory(symbol_map['inference_output'], NUM_CLASSES)
    scores = np.array(raw, dtype=np.int8)
    top5 = np.argsort(scores)[::-1][:5]

    print("Top 5 predictions:")
    for idx in top5:
        print(f"  class {idx:4d} ({labels[idx]}): {scores[idx]}")
    max_idx = int(top5[0])
    print(f"Output info: Top index {max_idx} ({labels[max_idx]}) "
          f"with value {scores[max_idx]}")

    inference_status = npu_sim.read_memory(symbol_map['inference_status'], 1)[0]
    print(f"inference_status {inference_status}")

    in_top1 = max_idx == gt
    in_top5 = gt in top5
    verdict = "PASS" if in_top1 else ("top-5" if in_top5 else "MISS")
    print(f"Expected class {gt} ({labels[gt]}) -> {verdict}")
    return in_top1, in_top5, cycles


def main():
    r = runfiles.Create()
    elf_file = r.Rlocation(f'{PKG}/run_full_mobilenet_v1_real_binary.elf')
    labels_file = r.Rlocation(f'{PKG}/labels/imagenet_labels.txt')
    manifest_file = r.Rlocation(f'{PKG}/images/val10_manifest.json')

    labels = load_imagenet_labels(labels_file)
    with open(manifest_file) as f:
        manifest = json.load(f)

    top1_hits = 0
    top5_hits = 0
    total_cycles = 0
    print(f"Verifying kernels on {len(manifest)} random ImageNet val images")
    for entry in manifest:
        gt = entry['class_index']
        npy_path = r.Rlocation(f"{PKG}/images/{entry['npy']}")
        print()
        in_top1, in_top5, cycles = run_one(elf_file, npy_path, labels, gt)
        top1_hits += in_top1
        top5_hits += in_top5
        total_cycles += cycles

    n = len(manifest)
    print(f"\nTop-1 accuracy: {top1_hits}/{n}")
    print(f"Top-5 accuracy: {top5_hits}/{n}")
    print(f"cycles taken by the simulation {total_cycles}")

    if top1_hits < MIN_TOP1 or top5_hits < MIN_TOP5:
        raise SystemExit(
            f"Kernel verification FAILED: top1={top1_hits} (need >={MIN_TOP1}), "
            f"top5={top5_hits} (need >={MIN_TOP5})")
    print("Kernel verification PASSED")


if __name__ == "__main__":
    main()
