---
license: apache-2.0
tags:
  - yamnet
  - audio-classification
  - tflite
  - tflite-micro
  - embedded
  - esp32
  - esp32-s3
  - audioset
library_name: tflite
language:
  - en
pipeline_tag: audio-classification
---

# YAMNet INT8 TFLite-Micro — mel-patch input (521 AudioSet classes)

Drop-in INT8-quantised **YAMNet** that actually runs on microcontrollers via **TFLite Micro**. Keeps Google's full **521-class AudioSet** head (class 20 = `Baby cry, infant cry`). Input is a precomputed **96 × 64 log-mel patch**, not a raw waveform — the in-graph `AUDIO_SPECTROGRAM` / `MFCC` ops in the stock `yamnet.tflite` from TensorFlow Hub are *not* supported by TFLite Micro, which is why that model won't load on an ESP32-S3 / Cortex-M and why this one exists.

## Why this exists

The stock `yamnet.tflite` from TF Hub uses `AUDIO_SPECTROGRAM` and `MFCC` ops — these are "Flex" (Select TF ops) and are **not in TFLite Micro's op registry**. Result: `AllocateTensors` fails with `Didn't find op for builtin opcode 'AUDIO_SPECTROGRAM'`. STMicroelectronics ships an ONNX version for STM32N6 but not a TFLite variant. No public HuggingFace or community drop existed (as of 2026-04). This model fills that gap for anyone doing audio event detection on ESP32-S3, Nordic nRF52/53, STM32, etc.

## Quick facts

| Attribute | Value |
|---|---|
| Architecture | YAMNet (MobileNetV1 depthwise-separable) |
| Input | `[1, 96, 64]` float32 → quantised INT8 (per input quant params) |
| Output | `[1, 521]` INT8 probabilities (sigmoid, not raw logits) |
| Input audio format | 16 kHz mono, 25 ms window / 10 ms hop, 64 mel bands 125–7500 Hz, log-mel |
| Patch duration | 96 frames × 10 ms ≈ 0.96 s |
| Params | 3.75 M |
| Model size | ~4.0 MB INT8 |
| Tensor arena (TFLite Micro) | ~1.2 MB (tested on ESP32-S3) |
| Reference inference latency | ~500 ms / patch on ESP32-S3 @ 240 MHz via `esp-tflite-micro` with ESP-NN |
| Quantisation | Full-integer PTQ, INT8 weights + activations |
| Source weights | Google YAMNet `yamnet.h5` (Apache-2.0) from `research/audioset/yamnet` |
| Class map | [Google AudioSet `yamnet_class_map.csv`](https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv) — unchanged |

## Files

- `yamnet.tflite` — the model.
- `yamnet_class_map.csv` — index → display name, unchanged from Google.
- `convert_yamnet.py` — reproducible converter; downloads Google's weights + architecture, surgically replaces the waveform input with a mel-patch placeholder, PTQ-quantises to INT8.

## Using it — on-device (ESP-IDF / ESP32-S3)

1. Pull into a SPIFFS partition:

   ```c
   esp_vfs_spiffs_conf_t cfg = {
       .base_path = "/yamnet",
       .partition_label = "yamnet",
       .max_files = 2,
       .format_if_mount_failed = false,
   };
   esp_vfs_spiffs_register(&cfg);
   ```

2. Load via `esp-tflite-micro` with an op resolver covering: `Conv2D`, `DepthwiseConv2D`, `FullyConnected`, `AveragePool2D`, `MaxPool2D`, `Mean`, `Reshape`, `Quantize`, `Dequantize`, `Softmax`, `Logistic`, `Pad`, `Add`, `Mul`, `Relu`, `Relu6`.

3. Compute log-mel yourself (e.g. with `esp-dsp`):
   - 16 kHz mono PCM → 400-sample Hann window, 512-pt FFT, |X|² → 64-band HTK mel filterbank (125–7500 Hz) → `log(mel + 1e-10)`
   - Maintain a rolling ring of 96 frames → once every 48 new frames, flatten into a `[96, 64]` patch and quantise to INT8 using `input_tensor->params.scale` / `zero_point`.

4. Invoke the interpreter; read `output[0..520]`. `output[20]` after `scale * (raw - zero_point)` is the probability of `Baby cry, infant cry`.

### Reference implementation

The `cry-detect-01` project at [github.com/chayuto/ws-ESP32-S3-CAM](https://github.com/chayuto/ws-ESP32-S3-CAM/tree/main/projects/cry-detect-01) is a complete end-to-end reference: ES7210 microphone capture, STFT + mel extraction, TFLite Micro inference, event detector, live HTTP audio stream, SD WAV recorder, web UI. Uses this exact model.

## Using it — host (Python / TFLite)

```python
import numpy as np
import tensorflow as tf

interp = tf.lite.Interpreter(model_path="yamnet.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

# mel_patch: float32 shape (96, 64) log-mel as described above.
qpatch = (mel_patch / inp["quantization"][0] + inp["quantization"][1]).round().astype(np.int8)
interp.set_tensor(inp["index"], qpatch[None])
interp.invoke()
y = interp.get_tensor(out["index"])[0]    # (521,) int8
probs = out["quantization"][0] * (y.astype(np.int32) - out["quantization"][1])
print("baby cry:", probs[20])
```

## Quantisation notes & accuracy

Calibration used **200 synthetic log-mel patches** drawn from a Gaussian
centred to match typical log-mel statistics (`N(-5, 3)`). This is enough for
the INT8 PTQ to converge and the model to run, **but accuracy is slightly
degraded relative to the float baseline**, especially on rare classes. For
best accuracy, re-run the converter with real audio:

```bash
python convert_yamnet.py --audio-dir /path/to/16khz_mono_wavs --calib-count 500
```

Representative data from AudioSet-balanced-train, ESC-50, UrbanSound8K all
work. The `cry-detect-01` reference project also exposes a "real-world
adaptive noise floor" that scales the detection threshold relative to
learned ambient RMS — this largely compensates for any synthetic-calibration
drift in practical baby-monitor deployments.

## License

- **Model weights:** Apache-2.0, from Google's [research/audioset/yamnet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet).
- **Graph surgery & conversion script:** Apache-2.0, contributed in this repo.
- **Class map CSV:** Google, unchanged.

No modification to architecture or weights — only input tensor shape changed (to bypass unsupported TFLM ops) and quantised to INT8.

## Intended use

- Embedded audio event detection where the waveform-input TF Hub model is too heavy or uses ops unsupported by your runtime.
- Baby-cry / pet-sound / glass-break / alarm monitors on microcontrollers.
- Research baseline for further INT8 transfer-learning on AudioSet-style tasks.

## Limitations

- 521-class AudioSet labels are imbalanced; rare classes (e.g. "Hiccup", "Burping") perform poorly even in the float original. YAMNet is a **scene classifier**, not a strict event detector — expect non-zero probabilities for many classes in noisy audio. Use thresholding + temporal smoothing + adaptive noise-floor rather than raw argmax.
- Synthetic PTQ calibration → small accuracy drop vs float; retrain with real audio for production.
- Mel extractor must match exactly (25 ms/10 ms/64 mel/125-7500 Hz HTK) or accuracy suffers.

## Citation

If you use this model, cite Google's original:

```bibtex
@misc{google2017yamnet,
  title = {YAMNet: A pretrained deep net for audio event recognition},
  author = {Google Research},
  year = {2017},
  howpublished = {\url{https://github.com/tensorflow/models/tree/master/research/audioset/yamnet}}
}
```

And optionally reference this repack:

```
chayuto/yamnet-mel-int8-tflm — YAMNet INT8 TFLite-Micro with mel-patch input (2026)
```
