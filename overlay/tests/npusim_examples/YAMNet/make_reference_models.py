"""Export trustworthy YAMNet reference models (Phase A).

Full-integer int8 PTQ collapses YAMNet's accuracy (int8 *activations* are the
problem; see README). These two variants keep float activations and match the
float baseline (~8/10 top-5 on validation-set), so they serve as the golden
reference the NPU int8 path is scored against:

  yamnet_float.tflite          fp32 weights + activations   (~15 MB)
  yamnet_int8_dynrange.tflite  int8 weights, float acts     (~3.8 MB)

Both are built from Google's real yamnet.h5, loaded TOPOLOGICALLY (the fix in
convert_yamnet.py; by_name loading silently loaded nothing).

Run (TensorFlow lives in ~/tfconvenv; legacy Keras needed for Google's yamnet.py):
    TF_USE_LEGACY_KERAS=1 ~/tfconvenv/bin/python make_reference_models.py
"""

import os
import sys
from pathlib import Path

import tensorflow as tf

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
sys.path.insert(0, str(MODEL_DIR))
import convert_yamnet as cv  # reuse ensure_yamnet_source + build_mel_patch_model


def build_loaded_model():
    cv.ensure_yamnet_source()
    import params as params_mod
    import yamnet as yamnet_mod
    model, _ = cv.build_mel_patch_model(params_mod, yamnet_mod)
    model.load_weights(os.path.join(cv.WORK_DIR, "yamnet.h5"))  # topological
    return model


def main():
    model = build_loaded_model()

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    float_blob = conv.convert()
    (MODEL_DIR / "yamnet_float.tflite").write_bytes(float_blob)
    print(f"  yamnet_float.tflite          {len(float_blob) // 1024} KB")

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]  # int8 weights, float acts
    dyn_blob = conv.convert()
    (MODEL_DIR / "yamnet_int8_dynrange.tflite").write_bytes(dyn_blob)
    print(f"  yamnet_int8_dynrange.tflite  {len(dyn_blob) // 1024} KB")


if __name__ == "__main__":
    main()
