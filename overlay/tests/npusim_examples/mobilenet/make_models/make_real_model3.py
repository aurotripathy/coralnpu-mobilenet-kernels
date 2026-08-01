"""MobileNet V1 0.25 224 int8 (1000 classes) with preprocessing folded in.

Folds the (x/127.5 - 1) rescaling into the first conv's weights/bias so the
model takes raw [0,255] pixels. The int8 input then quantizes with
zero_point=-128 (input_offset=+128), the same regime as the repo's dummy
model, which the CoralNPU-optimized conv kernels are known to handle.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
CAT_NPY = str(_EXAMPLE_DIR / 'images_224x224x3' / 'cat_224x224_real.npy')
OUT = str(_EXAMPLE_DIR / 'models' / 'mobilenet_v1_025_224_int8_real.tflite')

model = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3), alpha=0.25, weights='imagenet')

# conv(x/127.5 - 1, W) + b == conv(x, W/127.5) + (b - sum_hwi(W))
conv1 = model.get_layer('conv1')
W = conv1.get_weights()[0]  # (h, w, in, out); conv1 has no bias
Wp = W / 127.5
bp = -W.sum(axis=(0, 1, 2))
conv1.use_bias = True
conv1_cfg = conv1.get_config()
conv1_cfg['use_bias'] = True

# Rebuild the model with a biased conv1 by cloning layer-by-layer.
inp = tf.keras.Input(shape=(224, 224, 3))
x = inp
new_conv1 = tf.keras.layers.Conv2D.from_config(conv1_cfg)
for layer in model.layers[1:]:
    if layer.name == 'conv1':
        x = new_conv1(x)
    else:
        x = layer(x)
folded = tf.keras.Model(inp, x)
new_conv1.set_weights([Wp, bp])

cat = np.load(CAT_NPY).astype(np.float32)  # raw [0,255] pixels

# Sanity check the fold in float.
ref = model(np.expand_dims(cat / 127.5 - 1.0, 0)).numpy()
got = folded(np.expand_dims(cat, 0)).numpy()
print('fold max abs diff:', np.abs(ref - got).max())

def rep_dataset():
    yield [np.expand_dims(cat, 0)]
    rng = np.random.default_rng(42)
    for _ in range(63):
        yield [rng.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(folded)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
open(OUT, 'wb').write(tflite_model)
print('model size:', len(tflite_model))

interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
i = interp.get_input_details()[0]
o = interp.get_output_details()[0]
print('input quant:', i['quantization'], 'output shape:', o['shape'].tolist())
from collections import Counter
print('ops:', Counter(od['op_name'] for od in interp._get_ops_details()))

q = (cat.astype(np.int16) - 128).astype(np.int8)
interp.set_tensor(i['index'], q[None, ...])
interp.invoke()
s = interp.get_tensor(o['index'])[0]
top5 = np.argsort(s)[::-1][:5]
print('top5:', top5.tolist(), 'scores:', s[top5].tolist())
