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

"""Runs a real (ImageNet-trained) MobileNet V1 0.25 224 on the NPU simulator.

Unlike npusim_run_mobilenet.py, which drives a dummy 5-class model with random
input, this test feeds a real cat photo into a fully int8-quantized MobileNet
V1 and picks the final class out of the full 1000 ImageNet classes.
"""

from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np

EXPECTED_SHAPE = (224, 224, 3)
NUM_CLASSES = 1000  # ImageNet (ILSVRC-2012)


def load_input_from_npy(npy_path):
    """Loads an image from an .npy file and returns it as flat int8 data.

    The returned array is a 1-D ``np.int8`` array of length 224*224*3, ready
    to be written into the ``inference_input`` buffer. The model's input
    quantization is (scale=1/127.5, zero_point=0) over a [-1, 1] preprocessed
    domain, which maps uint8 pixels to ``pixel - 128``.

    Args:
        npy_path: Path to a ``.npy`` file holding a 224x224x3 image.

    Returns:
        A 1-D ``np.int8`` array of length ``224 * 224 * 3``.

    Raises:
        ValueError: If the array does not have shape (224, 224, 3).
    """
    image = np.load(npy_path)
    if image.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected input of shape {EXPECTED_SHAPE}, got {image.shape} "
            f"from {npy_path}")

    if image.dtype == np.uint8:
        image = image.astype(np.int16) - 128

    return image.astype(np.int8).reshape(-1)


def load_imagenet_labels(labels_path):
    """Returns the 1000 ImageNet class labels.

    The labels file has 1001 lines with a leading "background" entry; that
    entry is dropped so indices line up with the model's 1000-class output.
    """
    with open(labels_path) as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels[1:]


def run_real_mobilenet():
    print("Running real mobilenet on a cat image...")
    npu_sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True)
    r = runfiles.Create()
    elf_file = r.Rlocation(
        'coralnpu_hw/tests/npusim_examples/mobilenet/run_full_mobilenet_v1_real_binary.elf')
    image_file = r.Rlocation(
        'coralnpu_hw/tests/npusim_examples/mobilenet/images/cat_224x224_real.npy')
    labels_file = r.Rlocation(
        'coralnpu_hw/tests/npusim_examples/mobilenet/labels/imagenet_labels.txt')

    entry_point, symbol_map = npu_sim.get_elf_entry_and_symbol(
        elf_file, ['inference_status', 'inference_input', 'inference_output'])
    npu_sim.load_program(elf_file, entry_point)

    if symbol_map.get('inference_input'):
        input_data = load_input_from_npy(image_file)
        print(f"Writing cat image to {symbol_map['inference_input']}")
        npu_sim.write_memory(symbol_map['inference_input'], input_data)

    print("Running simulation...", flush=True)
    npu_sim.run()
    npu_sim.wait()
    print(f"cycles taken by the simulation {npu_sim.get_cycle_count()}")

    if symbol_map.get('inference_output'):
        output_data = npu_sim.read_memory(
            symbol_map['inference_output'], NUM_CLASSES)
        output_data = np.array(output_data, dtype=np.int8)
        labels = load_imagenet_labels(labels_file)
        top5 = np.argsort(output_data)[::-1][:5]
        print("Top 5 predictions:")
        for idx in top5:
            print(f"  class {idx:4d} ({labels[idx]}): {output_data[idx]}")
        max_idx = top5[0]
        print(f"Output info: Top index {max_idx} ({labels[max_idx]}) "
              f"with value {output_data[max_idx]}")

    if symbol_map.get('inference_status'):
        inference_status = npu_sim.read_memory(
            symbol_map['inference_status'], 1)[0]
        print(f"inference_status {inference_status}")


if __name__ == "__main__":
    run_real_mobilenet()
