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

"""Prepares random ImageNet validation images for :npusim_verify_val10.

Samples N ImageNet (ILSVRC-2012) classes at random (one image per class),
center-crops each to 224x224x3, and writes them as uint8 .npy files into the
``images_224x224x3/`` folder next to this script, alongside a
``val10_manifest.json``
that records each image's ground-truth class index and label.

Images are pulled from the public one-image-per-class ImageNet mirror
``EliSchwartz/imagenet-sample-images``. The files are listed in synset order,
so the position of a file in the sorted list equals its ImageNet class index,
which lines up with ``labels/imagenet_labels.txt`` (after its leading
"background" entry is dropped).

This is a host-side helper, not a Bazel target; it needs ``numpy`` and
``pillow`` and network access. Run it from anywhere:

    python3 tests/npusim_examples/mobilenet/prepare_val_images.py --seed 42 --count 10

Then rebuild/run the verifier (the ``glob`` in BUILD picks up new ``val_*``
files automatically):

    bazel run //tests/npusim_examples/mobilenet:npusim_verify_val10
"""

import argparse
import json
import os
import random
import subprocess

import numpy as np
from PIL import Image

REPO = "EliSchwartz/imagenet-sample-images"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/master"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/master/"
NUM_CLASSES = 1000  # ImageNet (ILSVRC-2012)
CROP = 224

IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "images_224x224x3")


def list_class_files():
    """Returns the 1000 sample filenames in ImageNet class-index order."""
    tree = json.loads(subprocess.check_output(["curl", "-sSL", TREE_URL]))
    files = sorted(t["path"] for t in tree["tree"]
                   if t["path"].endswith(".JPEG"))
    if len(files) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} sample images, found {len(files)}")
    return files


def center_crop_224(jpg_path):
    """Loads a JPEG, resizes short side to 224, center-crops to 224x224x3."""
    img = Image.open(jpg_path).convert("RGB")
    w, h = img.size
    s = CROP / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - CROP) // 2, (h - CROP) // 2
    img = img.crop((left, top, left + CROP, top + CROP))
    arr = np.asarray(img, dtype=np.uint8)
    if arr.shape != (CROP, CROP, 3):
        raise RuntimeError(f"Unexpected crop shape {arr.shape} for {jpg_path}")
    return arr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for the class sample (default: 42).")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of images to sample (default: 10).")
    parser.add_argument("--out", default=IMAGES_DIR,
                        help="Output images directory "
                             "(default: ./images_224x224x3).")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = list_class_files()

    rng = random.Random(args.seed)
    picks = sorted(rng.sample(range(NUM_CLASSES), args.count))

    manifest = []
    for idx in picks:
        fname = files[idx]
        label = fname[:-len(".JPEG")].split("_", 1)[1]
        jpg = os.path.join("/tmp", fname)
        subprocess.check_call(["curl", "-sSL", "-o", jpg, RAW_URL + fname])
        arr = center_crop_224(jpg)
        out_name = f"val_{idx:04d}_{label}_{CROP}x{CROP}.npy"
        np.save(os.path.join(args.out, out_name), arr)
        manifest.append(
            {"npy": out_name, "class_index": idx, "label": label})
        print(f"class {idx:4d}  {label:30s} -> {out_name}")

    manifest_path = os.path.join(args.out, "val10_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {len(manifest)} images and {manifest_path}")


if __name__ == "__main__":
    main()
