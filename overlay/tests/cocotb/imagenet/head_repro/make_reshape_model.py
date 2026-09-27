"""Smallest int8 model containing the suspect STRIDED_SLICE chain.

Just Keras Reshape (symbolic batch) + softmax, which the converter lowers to
SHAPE -> STRIDED_SLICE -> PACK -> RESHAPE -> SOFTMAX. No MEAN/CONV, so the
RTL run reaches the wedge point within a few million cycles.

Run:
    ~/tfconvenv/bin/python make_reshape_model.py
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

OUT = str(Path(__file__).resolve().parent / "reshape_int8.tflite")

inp = tf.keras.Input(shape=(1, 1, 1000))  # batch dim symbolic on purpose
x = tf.keras.layers.Reshape((1000,), name="reshape_2")(inp)
out = tf.keras.layers.Activation("softmax", name="predictions")(x)
model = tf.keras.Model(inp, out)


def rep_dataset():
    rng = np.random.default_rng(42)
    for _ in range(64):
        yield [rng.normal(0.0, 4.0, size=(1, 1, 1, 1000)).astype(np.float32)]


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

rng = np.random.default_rng(7)
q = rng.integers(-128, 128, size=(1, 1, 1, 1000), dtype=np.int64).astype(np.int8)
interp.set_tensor(i["index"], q)
interp.invoke()
s = interp.get_tensor(o["index"])[0]
top5 = np.argsort(s)[::-1][:5]
print("golden top5:", top5.tolist(), "scores:", s[top5].tolist())
np.save(str(Path(OUT).parent / "reshape_input.npy"), q[0])
np.save(str(Path(OUT).parent / "reshape_golden.npy"), s)
