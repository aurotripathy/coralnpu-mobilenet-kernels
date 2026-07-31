"""Builds a full-int8 MobileNet V1 0.25 224 (1000 ImageNet classes) tflite.

Uses Keras pretrained weights + post-training quantization, then verifies the
model classifies the repo's cat image correctly.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
CAT_NPY = str(_EXAMPLE_DIR / 'images' / 'cat_224x224_real.npy')
OUT = '/tmp/mobilenet_v1_0.25_224_int8_real.tflite'

model = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3), alpha=0.25, weights='imagenet')

cat = np.load(CAT_NPY).astype(np.float32)  # uint8 [0,255]

def rep_dataset():
    # Representative samples in the model's preprocessed domain [-1, 1].
    yield [np.expand_dims(cat / 127.5 - 1.0, 0).astype(np.float32)]
    rng = np.random.default_rng(42)
    for _ in range(63):
        img = rng.uniform(-1.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
        yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
open(OUT, 'wb').write(tflite_model)
print('model size:', len(tflite_model))

# Verify
interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print('input:', inp['shape'], inp['dtype'], 'quant:', inp['quantization'])
print('output:', out['shape'], out['dtype'], 'quant:', out['quantization'])

scale, zp = inp['quantization']
q = np.round((cat / 127.5 - 1.0) / scale + zp).clip(-128, 127).astype(np.int8)
print('exact quant vs (uint8-128) max diff:',
      np.abs(q.astype(np.int16) - (cat.astype(np.int16) - 128)).max())

interp.set_tensor(inp['index'], np.expand_dims(q, 0))
interp.invoke()
scores = interp.get_tensor(out['index'])[0]
top5 = np.argsort(scores)[::-1][:5]
print('top5 keras-class indices:', top5, 'scores:', scores[top5])

# Also try simple uint8-128 input (what the loader in the repo produces)
interp.set_tensor(inp['index'],
                  np.expand_dims((cat.astype(np.int16) - 128).astype(np.int8), 0))
interp.invoke()
scores2 = interp.get_tensor(out['index'])[0]
top5b = np.argsort(scores2)[::-1][:5]
print('top5 with uint8-128 input:', top5b, 'scores:', scores2[top5b])

# Decode labels
from tensorflow.keras.applications.mobilenet import decode_predictions
onehot = np.zeros((1, 1000), dtype=np.float32)
for i in top5:
    onehot[0, i] = scores[i]
print(decode_predictions(onehot, top=5))

# check ops used
ops = set()
for od in interp._get_ops_details():
    ops.add(od['op_name'])
print('ops:', sorted(ops))
