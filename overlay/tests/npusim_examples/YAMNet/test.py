import sys
from pathlib import Path

import numpy as np

from log_mel import SAMPLE_RATE, load_wav, waveform_to_patches

# Short guide to the code:
# Interpreter(model_path=...) — parses the flatbuffer: graph structure, weights, quantization params. No computation.

# allocate_tensors() — sizes and allocates all intermediate tensor buffers and lets the delegate claim ops. Still no computation.

# set_tensor(inp["index"], qpatch[None]) — copies your int8 patch into the input buffer. Nothing runs yet.

# invoke() — runs the full graph: your (1, 96, 64) patch flows through all the 
# conv/depthwise/pool/dense layers to the (1, 521) output. This is where all the compute time goes.

# get_tensor(out["index"]) — copies the result out of the output buffer.


# litertenv has the lightweight LiteRT interpreter, not full tensorflow.
# Run with: ~/litertenv/bin/python test.py
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

HERE = Path(__file__).resolve().parent
MODEL = str(HERE / "model" / "yamnet.tflite")
CLASS_MAP = HERE / "model" / "yamnet_class_map.csv"

interp = Interpreter(model_path=MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

# mel_patch: float32 shape (96, 64) log-mel patch from log_mel.py.
# Input: a 16 kHz mono 16-bit .wav on argv, else a synthesized 440 Hz tone.
if len(sys.argv) > 1:
    waveform, rate = load_wav(sys.argv[1])
else:
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE  # 1 s
    waveform, rate = 0.5 * np.sin(2 * np.pi * 440.0 * t), SAMPLE_RATE
mel_patch = waveform_to_patches(waveform, rate)[0]

qpatch = (mel_patch / inp["quantization"][0] + inp["quantization"][1]).round().astype(np.int8)
interp.set_tensor(inp["index"], qpatch[None])

# the actual inference. It executes every op in the model's graph, in order, on the input you staged
interp.invoke()

y = interp.get_tensor(out["index"])[0]    # (521,) int8
probs = out["quantization"][0] * (y.astype(np.int32) - out["quantization"][1])

# Class names: skip the csv header; columns are index,mid,display_name.
labels = [line.split(",", 2)[2].strip().strip('"')
          for line in CLASS_MAP.read_text().splitlines()[1:]]
top5 = np.argsort(probs)[::-1][:5]
print("Top 5:")
for i in top5:
    print(f"  {i:3d} {labels[i]:30s} {probs[i]:.3f}")
