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

"""
Loading helpers: input image, class labels, weights from the coralnpu int8
.tflite, and a runner for that .tflite so it can be used as a reference.
"""

import os
import numpy as np
import torch


def load_image(path, normalize=False):
    """
    Load a 224x224x3 uint8 HWC .npy image and convert it to a model input:
        HWC uint8 [0, 255]  ->  NCHW float32
    normalize=False: keep raw pixel values [0, 255]. Use this with weights loaded
                     from the coralnpu int8 .tflite, which has the x/127.5 - 1
                     preprocessing folded into conv1.
    normalize=True:  scale to [-1, 1] (MobileNet V1 / TF-Slim / Keras convention).
                     Use this with the original float Keras/TF-Slim weights.
    """
    img = np.load(os.path.expanduser(path))                   # (224, 224, 3) uint8
    assert img.shape == (224, 224, 3), f"unexpected shape {img.shape}"
    x = torch.from_numpy(img).permute(2, 0, 1).float()        # (3, 224, 224) float32
    if normalize:
        x = x / 127.5 - 1.0                                   # [0, 255] -> [-1, 1]
    return x.unsqueeze(0)                                     # (1, 3, 224, 224)


def load_labels(path):
    """
    Load the 1000 ImageNet class names, one per line.
    The label file has 1001 lines with a leading 'background' entry (TF-Slim
    convention); the Keras-derived weights used by the coralnpu reference model
    have 1000 outputs, so the leading entry is dropped and output index i maps
    to labels[i]. Matches load_imagenet_labels() in npusim_run_real_mobilenet.py.
    """
    with open(os.path.expanduser(path)) as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels[1:]


def load_tflite_weights(model, path):
    """
    Load weights from the coralnpu int8 .tflite into `model` (kept float32).

    How the .tflite is laid out (see make_models/README.md in the coralnpu repo):
      * Fully int8 post-training quantized: weights are per-output-channel
        symmetric int8 (zero_point 0), biases are int32. Dequantize with
        real = scale[c] * (q - zero_point[c]).
      * BatchNorm is already folded into each conv, so every conv has a bias and
        there are no BN tensors. To keep this model's architecture unchanged the
        bias is placed in the BatchNorm2d that follows the conv, configured as
        identity + bias:  gamma=1, beta=bias, running_mean=0, running_var=1-eps
        so that bn(x) = x + bias.
      * The x/127.5 - 1 input preprocessing is folded into conv1, so the model
        must be fed raw [0, 255] pixels (load_image(..., normalize=False)).
      * TFLite weight layouts differ from PyTorch:
            conv       [Cout, kh, kw, Cin]  ->  [Cout, Cin, kh, kw]
            depthwise  [1, kh, kw, C]       ->  [C, 1, kh, kw]
            conv_preds [1000, 1, 1, 256]    ->  Linear weight [1000, 256]
      * Ops are stored in execution order: op 0 is the stem, ops 1..26 alternate
        depthwise/pointwise for blocks 1..13, then MEAN, then the classifier
        conv. Each conv op's inputs are [activation, weight, bias]. We map by
        position rather than by tensor name.

    Returns the ai_edge_litert Interpreter so the caller can run the reference
    .tflite on the same input and compare.
    """
    from ai_edge_litert.interpreter import Interpreter   # pip/uv install ai-edge-litert

    interp = Interpreter(model_path=os.path.expanduser(path))
    interp.allocate_tensors()
    details = {d["index"]: d for d in interp.get_tensor_details()}

    def dequant(idx):
        d = details[idx]
        qp = d["quantization_parameters"]
        scales = qp["scales"].astype(np.float32)
        zps = qp["zero_points"].astype(np.float32)
        q = interp.get_tensor(idx).astype(np.float32)
        shape = [1] * q.ndim
        shape[qp["quantized_dimension"]] = -1                 # broadcast along the channel axis
        return (q - zps.reshape(shape)) * scales.reshape(shape)

    def set_conv_bn(conv, bn, w, b):
        w = torch.from_numpy(np.ascontiguousarray(w))
        b = torch.from_numpy(np.ascontiguousarray(b))
        assert conv.weight.shape == w.shape, f"{tuple(conv.weight.shape)} vs {tuple(w.shape)}"
        assert bn.bias.shape == b.shape, f"{tuple(bn.bias.shape)} vs {tuple(b.shape)}"
        with torch.no_grad():
            conv.weight.copy_(w)
            bn.weight.fill_(1.0)
            bn.bias.copy_(b)
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0 - bn.eps)

    conv_ops = [op for op in interp._get_ops_details()
                if op["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D")]
    n_blocks = len(model.blocks)
    assert len(conv_ops) == 1 + 2 * n_blocks + 1, f"unexpected op count {len(conv_ops)}"

    # stem: CONV_2D [Cout, kh, kw, Cin] -> [Cout, Cin, kh, kw]
    _, w_idx, b_idx = conv_ops[0]["inputs"]
    set_conv_bn(model.stem.conv, model.stem.bn,
                dequant(w_idx).transpose(0, 3, 1, 2), dequant(b_idx))

    # blocks: (DEPTHWISE_CONV_2D, CONV_2D) pairs
    for i, blk in enumerate(model.blocks):
        dw_op, pw_op = conv_ops[1 + 2 * i], conv_ops[2 + 2 * i]
        assert dw_op["op_name"] == "DEPTHWISE_CONV_2D" and pw_op["op_name"] == "CONV_2D"
        _, w_idx, b_idx = dw_op["inputs"]                    # [1, kh, kw, C] -> [C, 1, kh, kw]
        set_conv_bn(blk.dw.conv, blk.dw.bn,
                    dequant(w_idx).transpose(3, 0, 1, 2), dequant(b_idx))
        _, w_idx, b_idx = pw_op["inputs"]                    # [Cout, 1, 1, Cin] -> [Cout, Cin, 1, 1]
        set_conv_bn(blk.pw.conv, blk.pw.bn,
                    dequant(w_idx).transpose(0, 3, 1, 2), dequant(b_idx))

    # head: 1x1 conv_preds [num_classes, 1, 1, C] -> Linear [num_classes, C]
    _, w_idx, b_idx = conv_ops[-1]["inputs"]
    w = dequant(w_idx)
    w = torch.from_numpy(np.ascontiguousarray(w.reshape(w.shape[0], w.shape[-1])))
    b = torch.from_numpy(np.ascontiguousarray(dequant(b_idx)))
    assert model.head.fc.weight.shape == w.shape, f"{tuple(model.head.fc.weight.shape)} vs {tuple(w.shape)}"
    with torch.no_grad():
        model.head.fc.weight.copy_(w)
        model.head.fc.bias.copy_(b)

    return interp


def load_tflite_weights_int8(model, path):
    """
    Load the coralnpu int8 .tflite into a mobilenet_v1_int8.MobileNetV1Int8:
    raw int8 weights, int32 biases and the quantization parameters (scale /
    zero point) of every activation, weight and bias tensor. Nothing is
    dequantized; each layer derives its integer requantization constants from
    these the way LiteRT-Micro does (see mobilenet_v1_int8.py).

    Op mapping is positional, as in load_tflite_weights: CONV_2D /
    DEPTHWISE_CONV_2D ops in execution order are stem, (dw, pw) x 13, then
    the classifier; MEAN is the head's avgpool and SOFTMAX the model's softmax.
    Weight layout transposes are the same as for the float loader.

    Returns an ai_edge_litert Interpreter built with the builtin kernels only
    (no XNNPACK delegate) and with all intermediate tensors preserved, so
    every activation can be read back with interp.get_tensor(idx) after
    invoke() and compared with the PyTorch layer outputs bit for bit.
    """
    from ai_edge_litert.interpreter import Interpreter, OpResolverType   # pip/uv install ai-edge-litert

    interp = Interpreter(model_path=os.path.expanduser(path),
                         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                         experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    details = {d["index"]: d for d in interp.get_tensor_details()}

    def quant(idx):
        """(scale, zero_point) of a per-tensor quantized tensor."""
        scale, zp = details[idx]["quantization"]
        return scale, zp

    def per_channel_scales(idx):
        qp = details[idx]["quantization_parameters"]
        assert np.all(qp["zero_points"] == 0), f"tensor {idx}: expected symmetric weights"
        return qp["scales"].astype(np.float32)

    def set_conv(layer, op, w_transpose):
        act_idx, w_idx, b_idx = op["inputs"]
        (out_idx,) = op["outputs"]
        w = np.ascontiguousarray(interp.get_tensor(w_idx).transpose(*w_transpose))   # int8
        b = interp.get_tensor(b_idx)                                                 # int32
        in_scale, in_zp = quant(act_idx)
        out_scale, out_zp = quant(out_idx)
        layer.set_quant(torch.from_numpy(w), torch.from_numpy(per_channel_scales(w_idx)),
                        torch.from_numpy(b), in_scale, in_zp, out_scale, out_zp)

    ops = interp._get_ops_details()
    conv_ops = [op for op in ops if op["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D")]
    n_blocks = len(model.blocks)
    assert len(conv_ops) == 1 + 2 * n_blocks + 1, f"unexpected op count {len(conv_ops)}"

    # stem: CONV_2D [Cout, kh, kw, Cin] -> [Cout, Cin, kh, kw]
    set_conv(model.stem, conv_ops[0], (0, 3, 1, 2))

    # blocks: (DEPTHWISE_CONV_2D, CONV_2D) pairs
    for i, blk in enumerate(model.blocks):
        dw_op, pw_op = conv_ops[1 + 2 * i], conv_ops[2 + 2 * i]
        assert dw_op["op_name"] == "DEPTHWISE_CONV_2D" and pw_op["op_name"] == "CONV_2D"
        set_conv(blk.dw, dw_op, (3, 0, 1, 2))     # [1, kh, kw, C]    -> [C, 1, kh, kw]
        set_conv(blk.pw, pw_op, (0, 3, 1, 2))     # [Cout, 1, 1, Cin] -> [Cout, Cin, 1, 1]

    # head: MEAN, then the 1x1 classifier conv [num_classes, 1, 1, C] -> [num_classes, C, 1, 1]
    (mean_op,) = [op for op in ops if op["op_name"] == "MEAN"]
    model.head.avgpool.set_quant(*quant(mean_op["inputs"][0]), *quant(mean_op["outputs"][0]))
    set_conv(model.head.fc, conv_ops[-1], (0, 3, 1, 2))

    # softmax
    (softmax_op,) = [op for op in ops if op["op_name"] == "SOFTMAX"]
    model.softmax.set_quant(*quant(softmax_op["inputs"][0]), *quant(softmax_op["outputs"][0]))

    return interp


def run_tflite(interp, image):
    """
    Run the int8 reference .tflite on the same (1, 3, 224, 224) raw-pixel float
    input used for the PyTorch model. Returns dequantized softmax probabilities
    of shape (num_classes,).
    Input quantization is scale=1, zero_point=-128, i.e. q = pixel - 128.
    """
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x = image[0].permute(1, 2, 0).numpy()                    # NCHW -> HWC
    scale, zp = inp["quantization"]
    q = np.round(x / scale + zp).clip(-128, 127).astype(np.int8)[None]
    interp.set_tensor(inp["index"], q)
    interp.invoke()
    y = interp.get_tensor(out["index"]).astype(np.float32)[0]
    scale, zp = out["quantization"]
    return (y - zp) * scale
