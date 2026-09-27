"""Download a diverse ESC-50 calibration corpus for YAMNet int8 PTQ.

The shipped yamnet.tflite was PTQ-calibrated on synthetic N(-5,3) Gaussian
patches, which flattens the sigmoid head to ~0.5 for every class (useless on
real audio). Post-training quantization needs *representative* activations,
so this downloads real ESC-50 clips spread across all 50 categories,
resamples 44.1 kHz -> 16 kHz mono, and writes them where convert_yamnet.py's
--audio-dir can consume them (it requires 16 kHz wavs).

Run (needs librosa + soundfile, present in ~/litertenv):
    ~/litertenv/bin/python make_calib_corpus.py --per-category 6 --out calib16k
"""

import argparse
import io
import urllib.request
from collections import defaultdict
from pathlib import Path

import librosa
import soundfile as sf

ESC50_META = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
ESC50_AUDIO = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"
SAMPLE_RATE = 16000
HERE = Path(__file__).resolve().parent


def load_meta():
    text = urllib.request.urlopen(ESC50_META).read().decode()
    rows = []
    for line in text.splitlines()[1:]:
        f = line.split(",")
        rows.append((f[0], f[3]))  # filename, category
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=6,
                    help="clips per ESC-50 category (50 categories total)")
    ap.add_argument("--out", default="calib16k", help="output dir")
    args = ap.parse_args()

    out_dir = (HERE / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_cat = defaultdict(list)
    for fname, cat in load_meta():
        by_cat[cat].append(fname)

    n = 0
    for cat, files in sorted(by_cat.items()):
        for src in files[: args.per_category]:
            raw = urllib.request.urlopen(ESC50_AUDIO + src).read()
            wav, _ = librosa.load(io.BytesIO(raw), sr=SAMPLE_RATE, mono=True)
            sf.write(out_dir / src, wav, SAMPLE_RATE, subtype="PCM_16")
            n += 1
        print(f"  {cat:20s} {min(len(files), args.per_category)} clips")
    print(f"\nWrote {n} clips to {out_dir}")


if __name__ == "__main__":
    main()
