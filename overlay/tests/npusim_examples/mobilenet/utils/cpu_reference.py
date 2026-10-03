# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU (host TFLite) cross-check for the npusim MobileNet drivers.

Runs the SAME int8 .tflite the simulator executes through a host TFLite
interpreter and compares the two int8 score vectors. The CPU run is integer
inference, not fp32/bf16: the model is fully quantized, so the interpreter
uses int8 kernels with int32 accumulators; the only floats involved are the
scale factors stored as tensor metadata.

Do not expect a bit-exact match. Measured on the val10 set, the simulator
(TFLM-based runtime + the optimized kernels) and the TFLite BUILTIN_REF
reference kernels disagree by 3-38 LSB per softmax-output score. This is NOT
conv-kernel error: on the simulator, upstream TFLM reference convs match the
optimized kernels to 1 LSB (cat top-1 -60 vs -61). The drift comes from
op-level implementation differences between TFLM and TFLite (fixed-point
MEAN/SOFTMAX details), amplified across the softmax. The verifier's default
tolerance (48 LSB) is an empirical envelope just above that; a broken kernel
produces bit-garbage scores that miss by 100+ LSB, which is the failure mode
this gate exists to catch.

The interpreter (ai-edge-litert / tflite-runtime / tensorflow) is NOT part of
the Bazel environment, so the in-process runner below is executed through an
external Python via subprocess:

    bazel run //tests/npusim_examples/mobilenet:npusim_verify_val10 -- \
        --cpu-check ~/litertenv/bin/python

Standalone use (with an interpreter-equipped python):

    python3 utils/cpu_reference.py --model models/<model>.tflite \
        --npy images_224x224x3/cat_224x224_real.npy

    or better yet, with the model and image paths already set:
    cd /home/coralnpu/coralnpu/tests/npusim_examples/mobilenet
    ~/litertenv/bin/python utils/cpu_reference.py \ 
    --model models/mobilenet_v1_025_224_int8_real.tflite \
    --npy images_224x224x3/cat_224x224_real.npy
"""

import argparse
import json
import os
import subprocess

import numpy as np


def _make_interpreter(tflite_path):
    """Builds a TFLite interpreter from whichever package is present.

    Forces the BUILTIN_REF op resolver where supported: the default CPU path
    (XNNPack delegate) requantizes with its own fp32 arithmetic and drifts
    several LSB from the canonical integer reference semantics (measured -46
    vs -49 on the cat top-1), so only the reference kernels give a
    deterministic, machine-independent baseline.
    """
    try:
        from ai_edge_litert import interpreter as itp
    except ImportError:
        itp = None
    if itp is None:
        try:
            from tflite_runtime import interpreter as itp
        except ImportError:
            pass
    if itp is None:
        try:
            from tensorflow import lite as itp
        except ImportError:
            raise ImportError(
                "No TFLite interpreter available. Install one, e.g.: "
                "python3 -m venv ~/litertenv && "
                "~/litertenv/bin/pip install ai-edge-litert numpy")
    resolver = getattr(itp, 'OpResolverType', None) or getattr(
        getattr(itp, 'experimental', None), 'OpResolverType', None)
    if resolver is not None:
        return itp.Interpreter(
            model_path=tflite_path,
            experimental_op_resolver_type=resolver.BUILTIN_REF)
    return itp.Interpreter(model_path=tflite_path)


def run_cpu_reference(tflite_path, npy_path):
    """Runs the int8 model on the host CPU for one image.

    Applies the same input encoding as the sim drivers (uint8 pixel - 128,
    the model's zero-point encoding) and returns the raw int8 score vector.
    Needs a TFLite interpreter package in the current Python environment.

    Args:
        tflite_path: Path to the int8 .tflite the simulator also runs.
        npy_path: Path to a 224x224x3 uint8 image .npy.

    Returns:
        A 1-D np.int8 array of the 1000 class scores.
    """
    image = np.load(npy_path)
    if image.dtype == np.uint8:
        image = image.astype(np.int16) - 128
    image = image.astype(np.int8)

    interpreter = _make_interpreter(tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if inp['dtype'] != np.int8 or out['dtype'] != np.int8:
        raise ValueError(
            f"Expected an int8-in/int8-out model, got {inp['dtype']} -> "
            f"{out['dtype']}; the CPU cross-check compares raw int8 scores.")
    interpreter.set_tensor(inp['index'], image.reshape(inp['shape']))
    interpreter.invoke()
    return interpreter.get_tensor(out['index']).reshape(-1).astype(np.int8)


def cpu_scores_via_subprocess(python_exe, script_path, tflite_path, npy_path):
    """Runs run_cpu_reference() in an external Python and returns the scores.

    Used from Bazel-run drivers, whose hermetic environment has no TFLite
    interpreter; `python_exe` must point at one that does.
    """
    # Scrub the environment: under `bazel run` PYTHONPATH points into Bazel's
    # hermetic site-packages, whose numpy shadows (and breaks) the external
    # Python's own installation.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PYTHON', 'RUNFILES'))}
    result = subprocess.run(
        [python_exe, script_path, '--model', tflite_path, '--npy', npy_path],
        capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"CPU reference run failed ({python_exe}):\n{result.stderr}")
    # Scores are the last stdout line (JSON list), so stray interpreter
    # logging above it is harmless.
    return np.array(json.loads(result.stdout.strip().splitlines()[-1]),
                    dtype=np.int8)


def compare_topk(sim_scores, cpu_scores, k=5):
    """Compares the top-1 / top-k CLASS ASSIGNMENTS of the two runs.

    Answers "do sim and CPU pick the same labels?" independent of ground
    truth and of score magnitudes. Both comparisons are tie-aware, because
    the int8 scores tie frequently (e.g. tabby == Egyptian cat) and the k-th
    slot often sits at the -128 floor shared by hundreds of classes, where
    argsort order is arbitrary:

      * top-1 agrees if either run's argmax class also scores maximal in the
        other run.
      * top-k agreement is the overlap |sim top-k  intersect  cpu top-k|.

    Returns:
        (top1_agree, topk_overlap, sim_topk, cpu_topk) with the top-k lists
        in descending-score order.
    """
    sim_top1 = int(np.argmax(sim_scores))
    cpu_top1 = int(np.argmax(cpu_scores))
    top1_agree = (cpu_scores[sim_top1] == cpu_scores.max()
                  or sim_scores[cpu_top1] == sim_scores.max())
    sim_topk = np.argsort(sim_scores)[::-1][:k].tolist()
    cpu_topk = np.argsort(cpu_scores)[::-1][:k].tolist()
    overlap = len(set(sim_topk) & set(cpu_topk))
    return top1_agree, overlap, sim_topk, cpu_topk


def compare_scores(sim_scores, cpu_scores, tol=1):
    """Compares sim vs CPU int8 score vectors.

    Compares values, not argsort order: exact ties (common in these scores)
    can legitimately order differently between environments.

    Returns:
        (ok, max_abs_diff, num_mismatched) where ok means every score agrees
        to within `tol` LSB.
    """
    diff = np.abs(sim_scores.astype(np.int32) - cpu_scores.astype(np.int32))
    max_abs_diff = int(diff.max())
    return max_abs_diff <= tol, max_abs_diff, int((diff > tol).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='int8 .tflite path')
    parser.add_argument('--npy', required=True, help='224x224x3 image .npy')
    args = parser.parse_args()
    scores = run_cpu_reference(args.model, args.npy)
    print(json.dumps(scores.tolist()))


if __name__ == '__main__':
    main()
