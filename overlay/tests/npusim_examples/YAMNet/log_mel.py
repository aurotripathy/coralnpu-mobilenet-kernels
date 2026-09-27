"""Numpy log-mel front-end for YAMNet.

Reimplements the AudioSet/YAMNet feature extraction
(tensorflow/models research/audioset/yamnet) in plain numpy, no
tensorflow/librosa/scipy needed:

    16 kHz mono float waveform
      -> 25 ms periodic-Hann window, 10 ms hop      (400 / 160 samples)
      -> 512-pt rFFT, MAGNITUDE spectrogram         (|X|, not |X|^2)
      -> 64-band HTK mel filterbank, 125-7500 Hz
      -> log(mel + 0.001)
      -> (96, 64) float32 patches, hopped every 48 frames (0.48 s)

Note: model/README.md describes the ESP32 reference front-end as |X|^2 with
log offset 1e-10, but the model's weights are Google's yamnet.h5 unchanged,
and Google trained on MAGNITUDE spectrograms with log offset 0.001
(features.py / mel_features.py). This module follows the training-time math.

One full patch needs at least 400 + 95*160 = 15600 samples (0.975 s).
"""

import wave

import numpy as np

SAMPLE_RATE = 16000
WINDOW = 400          # 25 ms
HOP = 160             # 10 ms
NFFT = 512
N_MELS = 64
FMIN_HZ = 125.0
FMAX_HZ = 7500.0
LOG_OFFSET = 0.001
PATCH_FRAMES = 96     # 0.96 s
PATCH_HOP = 48        # 0.48 s


def _hertz_to_mel(hz):
    """HTK mel scale, as used by tf.signal / AudioSet mel_features."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_filterbank():
    """Returns the (NFFT//2 + 1, N_MELS) mel weight matrix.

    Triangular filters with edges spaced uniformly in mel between FMIN_HZ
    and FMAX_HZ; the DC row is zeroed (mel_features.py convention).
    """
    spec_bins_mel = _hertz_to_mel(
        np.linspace(0.0, SAMPLE_RATE / 2.0, NFFT // 2 + 1))
    band_edges_mel = np.linspace(
        _hertz_to_mel(FMIN_HZ), _hertz_to_mel(FMAX_HZ), N_MELS + 2)
    weights = np.zeros((NFFT // 2 + 1, N_MELS))
    for i in range(N_MELS):
        lo, center, hi = band_edges_mel[i:i + 3]
        lower_slope = (spec_bins_mel - lo) / (center - lo)
        upper_slope = (hi - spec_bins_mel) / (hi - center)
        weights[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    weights[0, :] = 0.0  # never pass DC
    return weights


def waveform_to_log_mel(waveform, sample_rate=SAMPLE_RATE):
    """float waveform in [-1, 1] -> (num_frames, 64) float32 log-mel.

    Raises if the sample rate is not 16 kHz: resampling is out of scope
    here, and the model's accuracy depends on the front-end matching
    training exactly.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"YAMNet expects {SAMPLE_RATE} Hz audio, got {sample_rate}; "
            f"resample before calling.")
    waveform = np.asarray(waveform, dtype=np.float64).reshape(-1)
    num_frames = 1 + (len(waveform) - WINDOW) // HOP
    if num_frames < 1:
        raise ValueError(
            f"Need at least {WINDOW} samples (25 ms), got {len(waveform)}.")

    # Periodic Hann, matching tf.signal.hann_window / mel_features.py
    # (np.hanning is the symmetric variant -- close but not identical).
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi / WINDOW * np.arange(WINDOW))

    idx = np.arange(WINDOW) + HOP * np.arange(num_frames)[:, None]
    frames = waveform[idx] * window
    magnitude = np.abs(np.fft.rfft(frames, NFFT))
    mel = magnitude @ mel_filterbank()
    return np.log(mel + LOG_OFFSET).astype(np.float32)


def log_mel_to_patches(log_mel):
    """(num_frames, 64) -> (num_patches, 96, 64), hopped every 48 frames."""
    num_frames = log_mel.shape[0]
    if num_frames < PATCH_FRAMES:
        raise ValueError(
            f"Need >= {PATCH_FRAMES} frames for one patch, got {num_frames} "
            f"(>= {WINDOW + (PATCH_FRAMES - 1) * HOP} waveform samples).")
    starts = range(0, num_frames - PATCH_FRAMES + 1, PATCH_HOP)
    return np.stack([log_mel[s:s + PATCH_FRAMES] for s in starts])


def waveform_to_patches(waveform, sample_rate=SAMPLE_RATE):
    """float waveform in [-1, 1] -> (num_patches, 96, 64) float32."""
    return log_mel_to_patches(waveform_to_log_mel(waveform, sample_rate))


def load_wav(path):
    """Reads a 16-bit PCM mono 16 kHz .wav; returns float32 in [-1, 1].

    Stdlib-only on purpose (no scipy/soundfile in the venv). Stereo input is
    averaged to mono.
    """
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM, got "
                             f"{8 * w.getsampwidth()}-bit: {path}")
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
    return pcm.astype(np.float32) / 32768.0, rate
