"""Build a small YAMNet validation set from ESC-50.

AudioSet (YAMNet's training set) is not directly downloadable -- it's a set
of YouTube clip references. ESC-50 (Piczak, github.com/karolpiczak/ESC-50)
is a free, labeled set of 5 s environmental recordings whose categories map
cleanly onto AudioSet's 521 classes, so it's a practical stand-in for a
labeled validation set.

This script picks one clip per chosen category, resamples 44.1 kHz -> 16 kHz
mono, writes 16-bit PCM .wav into validation-set/, and emits a manifest.json
with the ground-truth AudioSet class index (plus the small set of related
indices that also count as correct, since YAMNet is a scene classifier and
neighboring labels -- e.g. "Fire" vs "Crackle" -- are both defensible).

Run (needs librosa + soundfile, present in ~/litertenv):
    ~/litertenv/bin/python prepare_validation_set.py
"""

import io
import json
import urllib.request
from pathlib import Path

import librosa
import soundfile as sf

ESC50_META = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
ESC50_AUDIO = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"
SAMPLE_RATE = 16000

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "validation-set"

# ESC-50 category -> ground-truth AudioSet label(s). First index is primary
# (used for reporting); any listed index counts as a top-1/top-5 hit.
# Indices verified against model/yamnet_class_map.csv.
TARGETS = [
    ("dog",            [69],       "Dog"),
    ("rooster",        [96, 94],   "Crowing, cock-a-doodle-doo"),
    ("rain",           [283, 284], "Rain"),
    ("sea_waves",      [289],      "Waves, surf"),
    ("crackling_fire", [293, 292], "Crackle"),
    ("crying_baby",    [20],       "Baby cry, infant cry"),
    ("sneezing",       [44],       "Sneeze"),
    ("clock_tick",     [402, 401], "Tick-tock"),
    ("helicopter",     [333],      "Helicopter"),
    ("church_bells",   [196, 195], "Church bell"),
]


def load_meta():
    """Returns list of (filename, category) rows from esc50.csv."""
    text = urllib.request.urlopen(ESC50_META).read().decode()
    rows = []
    for line in text.splitlines()[1:]:  # skip header
        f = line.split(",")
        rows.append((f[0], f[3]))  # filename, category
    return rows


def main():
    OUT_DIR.mkdir(exist_ok=True)
    meta = load_meta()
    manifest = []
    for category, gt_indices, gt_name in TARGETS:
        # Deterministic: first ESC-50 clip of this category.
        src = next(f for f, c in meta if c == category)
        raw = urllib.request.urlopen(ESC50_AUDIO + src).read()
        wav, _ = librosa.load(io.BytesIO(raw), sr=SAMPLE_RATE, mono=True)
        out_name = f"{category}.wav"
        sf.write(OUT_DIR / out_name, wav, SAMPLE_RATE, subtype="PCM_16")
        manifest.append({
            "file": out_name,
            "esc50_source": src,
            "esc50_category": category,
            "ground_truth_name": gt_name,
            "ground_truth_indices": gt_indices,
        })
        print(f"  {out_name:22s} <- {src:22s} "
              f"gt={gt_name} {gt_indices}")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} clips + manifest.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
