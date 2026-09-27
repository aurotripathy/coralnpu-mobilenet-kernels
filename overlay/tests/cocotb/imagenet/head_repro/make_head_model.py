"""Minimal int8 model reproducing MobileNet's classifier-head op chain.

The full MobileNet run wedges on the CoralNPU RTL inside node 30
(STRIDED_SLICE) -- part of the SHAPE -> STRIDED_SLICE -> PACK -> RESHAPE
chain the TFLite converter emits for Keras `Reshape` with a symbolic batch
dimension. This builds just the head, so the RTL repro takes minutes
instead of hours:

    7x7x256 int8 -> MEAN -> CONV_2D (1x1, 256->1000)
                 -> SHAPE -> STRIDED_SLICE -> PACK -> RESHAPE -> SOFTMAX

Layer structure mirrors tf.keras.applications.MobileNet's head exactly
(GlobalAveragePooling2D(keepdims=True) -> Conv2D -> Reshape -> softmax).

Run (TensorFlow lives in ~/tfconvenv):
    ~/tfconvenv/bin/python make_head_model.py
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

OUT = str(Path(__file__).resolve().parent / "head_int8.tflite")

inp = tf.keras.Input(shape=(7, 7, 256))  # batch dim left symbolic on purpose
x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(inp)
x = tf.keras.layers.Conv2D(1000, (1, 1), name="conv_preds")(x)
x = tf.keras.layers.Reshape((1000,), name="reshape_2")(x)
out = tf.keras.layers.Activation("softmax", name="predictions")(x)
model = tf.keras.Model(inp, out)


def rep_dataset():
    rng = np.random.default_rng(42)
    for _ in range(64):
        # Feature-map-like activations; range chosen to resemble the real
        # node-26 output scale (int8 dequantized values are a few units).
        yield [rng.normal(0.0, 2.0, size=(1, 7, 7, 256)).astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
open(OUT, "wb").write(tflite_model)
print("model size:", len(tflite_model))

interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
i = interp.get_input_details()[0]
o = interp.get_output_details()[0]
print("input quant:", i["quantization"], "output shape:", o["shape"].tolist())
print("nodes:")
for od in interp._get_ops_details():
    print(f"  node {od['index']:2d} {od['op_name']}")

# Golden check: fixed input -> print top-5 so the sim runs can be verified.
rng = np.random.default_rng(7)
q = rng.integers(-128, 128, size=(1, 7, 7, 256), dtype=np.int64).astype(np.int8)
interp.set_tensor(i["index"], q)
interp.invoke()
s = interp.get_tensor(o["index"])[0]
top5 = np.argsort(s)[::-1][:5]
print("golden top5:", top5.tolist(), "scores:", s[top5].tolist())
np.save(str(Path(OUT).parent / "head_input.npy"), q[0])
np.save(str(Path(OUT).parent / "head_golden.npy"), s)
