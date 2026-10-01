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
Build MobileNet V1 alpha=0.25, load the coralnpu int8 .tflite weights, classify
the cat image, compare against the .tflite reference, and print a per-stage
summary.

    python main.py --dtype fp32   float model, dequantized weights (default)
    python main.py --dtype int8   int8 model reproducing the .tflite integer arithmetic;
                                  every stage is compared with the .tflite bit for bit
"""

import argparse
import os

import torch

from mobilenet_v1_fp32 import MobileNetV1
from mobilenet_v1_int8 import MobileNetV1Int8
from utils import load_image, load_labels, load_tflite_weights, load_tflite_weights_int8, run_tflite

# data from the enclosing npusim mobilenet example (tests/npusim_examples/mobilenet)
EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_PATH = os.path.join(EXAMPLE_DIR, "labels", "imagenet_labels.txt")
TFLITE_PATH = os.path.join(EXAMPLE_DIR, "models", "mobilenet_v1_025_224_int8_real.tflite")
IMAGE_PATH = os.path.join(EXAMPLE_DIR, "images_224x224x3", "cat_224x224_real.npy")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtype", choices=["fp32", "int8"], default="fp32",
                   help="fp32: float model with dequantized weights; int8: integer model (default: fp32)")
    p.add_argument("--labels", default=LABELS_PATH)
    p.add_argument("--tflite", default=TFLITE_PATH)
    p.add_argument("--image", default=IMAGE_PATH)
    return p.parse_args()


def print_top5(probs, ref, labels, title, ref_title, probs_q=None, ref_q=None):
    """
    probs, ref: float probabilities (1000,). probs_q, ref_q: the raw int8 softmax
    outputs (1000,) they were dequantized from (int8 mode only); shown as
    prob_q so they can be compared directly with the npusim's int8 scores.
    """
    width = 48 + (12 if probs_q is not None else 0)
    print(f"\n{title:<{width}} {ref_title:<40}")
    print("-" * (width + 40))
    top5 = torch.topk(probs, k=5)
    top5_ref = torch.topk(ref, k=5)
    for (p, i), (pr, ir) in zip(zip(top5.values, top5.indices), zip(top5_ref.values, top5_ref.indices)):
        q = f" prob_q={probs_q[i].item():4d}" if probs_q is not None else ""
        qr = f" prob_q={ref_q[ir].item():4d}" if ref_q is not None else ""
        print(f"  {i.item():4d} {labels[i.item()]:<30} p={p.item():.4f}{q}   "
              f"{ir.item():4d} {labels[ir.item()]:<24} p={pr.item():.4f}{qr}")
    print(f"\nargmax agree: {probs.argmax().item() == ref.argmax().item()}   "
          f"max |p_pytorch - p_tflite|: {(probs - ref).abs().max().item():.4f}")


def stages(model):
    s = [("stem", model.stem)]
    s += [(f"block {i}", blk) for i, blk in enumerate(model.blocks, 1)]
    s += [("head", model.head)]
    return s


def fmt(shape):
    return " x ".join(str(d) for d in shape[1:])   # (1, 8, 112, 112) -> "8 x 112 x 112"


def print_stage_summary_fp32(model, image):
    """One row per stage: params, in/out shapes (batch dim dropped), dtype; then total."""
    print(f"\n{'layer':<10} {'params':>8}   {'in shape':<16} {'out shape':<16} {'dtype':<14}")
    print("-" * 70)
    total = 0
    x = image
    with torch.no_grad():
        for name, m in stages(model):
            n = sum(p.numel() for p in m.parameters())
            total += n
            dtype = next(m.parameters()).dtype
            y = m(x)
            print(f"{name:<10} {n:>8,}   {fmt(x.shape):<16} {fmt(y.shape):<16} {str(dtype):<14}")
            x = y
    print("-" * 70)
    print(f"{'total':<10} {total:>8,}")            # 470,072
    assert total == sum(p.numel() for p in model.parameters())


def print_stage_summary_int8(model, xq, interp):
    """
    Like the fp32 summary, plus a bit-exact comparison of each stage output
    with the corresponding .tflite tensor (conv ops are stored in execution
    order: stem, (dw, pw) x 13, classifier; a stage's output is its last conv).
    """
    conv_ops = [op for op in interp._get_ops_details() if op["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D")]
    ref_idx = [conv_ops[0]["outputs"][0]] + [conv_ops[2 + 2 * i]["outputs"][0] for i in range(len(model.blocks))] \
              + [conv_ops[-1]["outputs"][0]]

    def n_params(m):   # int8 weights + int32 biases
        return sum(b.numel() for name, b in m.named_buffers() if name.endswith(("weight", "bias")))

    print(f"\n{'layer':<10} {'params':>8}   {'in shape':<16} {'out shape':<16} {'dtype':<16} {'vs tflite':<20}")
    print("-" * 104)
    total, total_mismatch = 0, 0
    x = xq
    with torch.no_grad():
        for (name, m), idx in zip(stages(model), ref_idx):
            n = n_params(m)
            total += n
            y = m(x)
            ref = torch.from_numpy(interp.get_tensor(idx))
            ref = ref.permute(0, 3, 1, 2) if ref.dim() == 4 else ref            # NHWC -> NCHW
            ref = ref.reshape(y.shape)
            diff = (y.to(torch.int64) - ref.to(torch.int64)).abs()
            mismatch = int((diff > 0).sum())
            total_mismatch += mismatch
            status = "bit-exact" if mismatch == 0 else f"{mismatch} of {diff.numel()} differ, max {int(diff.max())} LSB"
            print(f"{name:<10} {n:>8,}   {fmt(x.shape):<16} {fmt(y.shape):<16} {'int8 w, int32 b':<16} {status:<32}")
            x = y
    print("-" * 104)
    print(f"{'total':<10} {total:>8,}   {'':<16} {'':<16} {'':<16} "
          f"{'all bit-exact' if total_mismatch == 0 else f'{total_mismatch} elements differ'}")


def main():
    args = parse_args()
    labels = load_labels(args.labels)              # 1000 entries ('background' dropped)
    print(f"labels: {args.labels}  ({len(labels)} classes)")

    if args.dtype == "fp32":
        model = MobileNetV1(num_classes=len(labels), alpha=0.25)
        print(f"model\n{model}")
        model.eval()                                   # inference: BN uses running stats
        interp = load_tflite_weights(model, args.tflite)
    else:
        model = MobileNetV1Int8(num_classes=len(labels), alpha=0.25)
        interp = load_tflite_weights_int8(model, args.tflite)   # builtin kernels, all tensors preserved
        print(f"model\n{model}")
        model.eval()
    print(f"\nweights: {args.tflite}")

    image = load_image(args.image, normalize=False)   # (1, 3, 224, 224) float32, raw [0, 255]
    print(f"input:   {args.image}")
    print(f"  shape {tuple(image.shape)}  dtype {image.dtype}  "
          f"min {image.min():.1f}  max {image.max():.1f}  mean {image.mean():.1f}")

    ref = torch.from_numpy(run_tflite(interp, image))   # also leaves the interpreter's tensors populated

    if args.dtype == "fp32":
        with torch.no_grad():
            probs = model(image).softmax(dim=1)[0]     # (1000,)
        print_top5(probs, ref, labels, "top-5 pytorch (fp32, weights from tflite)", "tflite int8 reference (xnnpack)")
        print_stage_summary_fp32(model, image)
    else:
        xq = model.quantize_input(image)               # int8, q = pixel - 128
        print(f"  quantized: int8, scale {float(model.stem.in_scale):.6g}, zero point {int(model.stem.in_zp)}")
        with torch.no_grad():
            logits_q = model(xq)                       # int8 (1, 1000)
            probs_q = model.softmax(logits_q)          # int8 (1, 1000), scale 1/256, zp -128
        probs = model.softmax.dequantize_output(probs_q)[0]
        ref_q = torch.from_numpy(interp.get_tensor(interp.get_output_details()[0]["index"]))   # int8 (1, 1000)
        print_top5(probs, ref, labels, "top-5 pytorch (int8, tflite arithmetic)", "tflite int8 reference (builtin)",
                   probs_q=probs_q[0], ref_q=ref_q[0])
        n_off = int((probs_q.to(torch.int64) - ref_q.to(torch.int64)).abs().gt(0).sum())
        print(f"int8 softmax output vs tflite: {'bit-exact' if n_off == 0 else f'{n_off} of {ref_q.numel()} classes differ'}")
        print_stage_summary_int8(model, xq, interp)


if __name__ == "__main__":
    main()
