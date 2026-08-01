"""MobileNet V1 0.25 224 int8, with input quant (scale=1, zp=-128).

Wraps the Keras model with a Rescaling layer so the tflite input domain is
raw [0,255] pixels, giving int8 input = pixel - 128 and input_offset = +128,
the same regime as the repo's dummy model.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent
CAT_NPY = str(_EXAMPLE_DIR / 'images_224x224x3' / 'cat_224x224_real.npy')
OUT = str(_EXAMPLE_DIR / 'models' / 'mobilenet_v1_025_224_int8_real.tflite')

base = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3), alpha=0.25, weights='imagenet')

inp = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)
out = base(x)
model = tf.keras.Model(inp, out)

cat = np.load(CAT_NPY).astype(np.float32)  # uint8 pixels [0,255]

def rep_dataset():
    yield [np.expand_dims(cat, 0)]
    rng = np.random.default_rng(42)
    for _ in range(63):
        yield [rng.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
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
print('input quant:', i['quantization'], 'output shape:', o['shape'])

q = (cat.astype(np.int16) - 128).astype(np.int8)
interp.set_tensor(i['index'], q[None, ...])
interp.invoke()
s = interp.get_tensor(o['index'])[0]
top5 = np.argsort(s)[::-1][:5]
print('top5:', top5.tolist(), 'scores:', s[top5].tolist())
