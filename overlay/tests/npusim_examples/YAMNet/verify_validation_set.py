"""Run the YAMNet validation-set clips through inference and score them.

For each clip in validation-set/manifest.json:
  - log_mel front-end (log_mel.py) -> (96, 64) patches
  - average the model's per-patch probabilities over the whole clip
  - report top-5 and whether a ground-truth AudioSet index is in top-1 / top-5

Run (LiteRT interpreter lives in ~/litertenv):
    ~/litertenv/bin/python verify_validation_set.py
"""

import json
from pathlib import Path

import numpy as np

from log_mel import load_wav, waveform_to_patches

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

HERE = Path(__file__).resolve().parent
MODEL = str(HERE / "model" / "yamnet.tflite")
CLASS_MAP = HERE / "model" / "yamnet_class_map.csv"
VAL_DIR = HERE / "validation-set"


def load_labels():
    return [line.split(",", 2)[2].strip().strip('"')
            for line in CLASS_MAP.read_text().splitlines()[1:]]


def clip_scores(interp, inp, out, patches):
    """Mean int8-dequantized probability vector over all patches in a clip."""
    scale, zero = inp["quantization"]
    oscale, ozero = out["quantization"]
    acc = np.zeros(521, dtype=np.float64)
    for patch in patches:
        q = (patch / scale + zero).round().clip(-128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None])
        interp.invoke()
        y = interp.get_tensor(out["index"])[0].astype(np.int32)
        acc += oscale * (y - ozero)
    return acc / len(patches)


def main():
    labels = load_labels()
    manifest = json.loads((VAL_DIR / "manifest.json").read_text())

    interp = Interpreter(model_path=MODEL)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    top1_hits = top5_hits = 0
    for item in manifest:
        waveform, rate = load_wav(VAL_DIR / item["file"])
        patches = waveform_to_patches(waveform, rate)
        probs = clip_scores(interp, inp, out, patches)
        top5 = np.argsort(probs)[::-1][:5]

        gt = set(item["ground_truth_indices"])
        in_top1 = int(top5[0]) in gt
        in_top5 = bool(gt & set(int(i) for i in top5))
        top1_hits += in_top1
        top5_hits += in_top5

        mark = "OK " if in_top5 else "MISS"
        print(f"[{mark}] {item['esc50_category']:15s} "
              f"gt={item['ground_truth_name']!r} "
              f"(top1 {'Y' if in_top1 else 'n'}, top5 {'Y' if in_top5 else 'n'})")
        for rank, i in enumerate(top5):
            flag = "*" if int(i) in gt else " "
            print(f"        {flag} {rank+1}. {labels[int(i)]:32s} {probs[int(i)]:.3f}")

    n = len(manifest)
    print(f"\nTop-1: {top1_hits}/{n}   Top-5: {top5_hits}/{n}")


if __name__ == "__main__":
    main()
