#!/usr/bin/env python3
"""Convert Google YAMNet to a TFLite Micro-compatible INT8 model.

Strips the in-graph mel spectrogram computation (AUDIO_SPECTROGRAM / MFCC
ops are Flex-only and unsupported by TFLite Micro) and expects a
precomputed 96x64 log-mel patch as input. Output is the 521-class
AudioSet logit vector; class 20 is "Baby cry, infant cry".

One-time host-side step. Requires:

    pip install "tensorflow>=2.12" numpy requests

Then:

    ./tools/convert_yamnet.py            # writes spiffs/yamnet.tflite

Calibration data defaults to synthetic random-ish log-mel patches — good
enough for the INT8 PTQ to converge and produce a working model, but
accuracy will be slightly below the float baseline. Pass --audio-dir
DIR to calibrate against a directory of 16 kHz mono WAV clips for the
best result.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import urllib.request

import numpy as np
import tensorflow as tf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DEST = os.path.join(PROJECT_ROOT, "spiffs", "yamnet.tflite")
WORK_DIR = os.path.join(SCRIPT_DIR, ".yamnet_work")

GOOGLE_RAW = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet"
WEIGHTS_URL = "https://storage.googleapis.com/audioset/yamnet.h5"


def fetch(filename: str, url: str, dst_dir: str) -> str:
    dst = os.path.join(dst_dir, filename)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    print(f"  fetch {filename}")
    urllib.request.urlretrieve(url, dst)
    return dst


def ensure_yamnet_source() -> None:
    os.makedirs(WORK_DIR, exist_ok=True)
    fetch("params.py", f"{GOOGLE_RAW}/params.py", WORK_DIR)
    fetch("yamnet.py", f"{GOOGLE_RAW}/yamnet.py", WORK_DIR)
    fetch("features.py", f"{GOOGLE_RAW}/features.py", WORK_DIR)
    fetch("yamnet.h5", WEIGHTS_URL, WORK_DIR)
    if WORK_DIR not in sys.path:
        sys.path.insert(0, WORK_DIR)


def build_mel_patch_model(params_mod, yamnet_mod):
    """Build a Keras model that consumes a single 96x64 log-mel patch."""
    params_obj = params_mod.Params(sample_rate=16000, patch_hop_seconds=0.48)
    mel_input = tf.keras.Input(
        batch_size=1,
        shape=(params_obj.patch_frames, params_obj.patch_bands),
        dtype=tf.float32,
        name="mel_patch",
    )
    predictions, _embeddings = yamnet_mod.yamnet(mel_input, params_obj)
    model = tf.keras.Model(inputs=mel_input, outputs=predictions, name="yamnet_mel_patch")
    return model, params_obj


def log_mel_patches_from_wav(path: str, params_mod, features_mod) -> np.ndarray | None:
    """Return all (patch_frames, patch_bands) log-mel patches, or None.

    Returning *every* patch (not just the first) is important for
    calibration: the first patch of a clip is often the quiet lead-in, so
    calibrating on it alone leaves activation ranges tuned for near-silence
    and clipping on loud events.
    """
    try:
        wav_bin = tf.io.read_file(path)
        wav, sr = tf.audio.decode_wav(wav_bin, desired_channels=1)
    except Exception:
        return None
    samples = tf.squeeze(wav, axis=-1).numpy()
    if sr.numpy() != 16000:
        return None
    p = params_mod.Params(sample_rate=16000, patch_hop_seconds=0.48)
    # Google's features.py exposes ..._patches (returns log_mel, patches);
    # the bare ..._spectrogram name in the original script never existed.
    _, patches = features_mod.waveform_to_log_mel_spectrogram_patches(samples, p)
    patches = patches.numpy()  # (num_patches, patch_frames, patch_bands)
    if patches.shape[0] == 0:
        return None
    return patches


def make_representative_dataset(params_obj, audio_dir: str | None, count: int):
    patches: list[np.ndarray] = []
    if audio_dir:
        try:
            from importlib import import_module
            features_mod = import_module("features")
            params_mod = import_module("params")
        except Exception:
            features_mod = None
            params_mod = None
        wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        for w in wavs:
            if len(patches) >= count:
                break
            clip = log_mel_patches_from_wav(w, params_mod, features_mod)
            if clip is not None:
                patches.extend(clip.astype(np.float32))
        print(f"  calibration: {len(patches)} real patches from "
              f"{len(wavs)} clips in {audio_dir}")

    rng = np.random.default_rng(0)
    while len(patches) < count:
        # Realistic log-mel range is roughly -20 to +5.
        patch = rng.standard_normal(
            (params_obj.patch_frames, params_obj.patch_bands), dtype=np.float32
        ) * 3.0 - 5.0
        patches.append(patch.astype(np.float32))
    print(f"  calibration total: {len(patches)} patches ({count} target)")

    def gen():
        for p in patches[: count]:
            yield [p[np.newaxis, ...]]

    return gen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEFAULT_DEST, help="output .tflite path")
    ap.add_argument("--audio-dir", default=None,
                    help="dir of 16 kHz mono WAV files for better INT8 calibration")
    ap.add_argument("--calib-count", type=int, default=200)
    args = ap.parse_args()

    print("[1/4] ensure YAMNet source and weights...")
    ensure_yamnet_source()

    import params as params_mod
    import yamnet as yamnet_mod

    print("[2/4] build mel-patch Keras model...")
    model, params_obj = build_mel_patch_model(params_mod, yamnet_mod)
    # yamnet.h5 is Google's topological weight save; it must be loaded in
    # layer order. The original by_name=True, skip_mismatch=True silently
    # matched *nothing* (the '/' layer names don't map by name), leaving a
    # randomly-initialized model -- which is why the shipped int8 model
    # classified real audio as noise regardless of calibration.
    model.load_weights(os.path.join(WORK_DIR, "yamnet.h5"))
    model.summary(line_length=100)

    print(f"[3/4] INT8 PTQ convert (calib_count={args.calib_count})...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(
        params_obj, args.audio_dir, args.calib_count)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite = converter.convert()

    print(f"[4/4] write {len(tflite)} bytes -> {args.dest}")
    os.makedirs(os.path.dirname(args.dest), exist_ok=True)
    with open(args.dest, "wb") as f:
        f.write(tflite)
    print("done. next: idf.py -C . -B build build && idf.py -p <port> flash")
    return 0


if __name__ == "__main__":
    sys.exit(main())
