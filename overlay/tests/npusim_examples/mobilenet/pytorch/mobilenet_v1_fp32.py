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
MobileNet V1 (Howard et al., 2017) with width multiplier alpha, built for
inference and for matching the TF/Keras/TFLite reference exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v, divisor=8):
    """Scale a channel count and round to a multiple of 8 (MobileNet convention)."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SamePad3x3(nn.Module):
    """
    TensorFlow 'SAME' padding for a 3x3 kernel, so the model matches the
    TF/Keras/TFLite reference exactly.
      stride 1: 2 pad pixels total, split (1, 1)      -> same as PyTorch padding=1
      stride 2: 1 pad pixel total (even input), placed
                on the bottom/right only: (0, 1)      -> PyTorch padding=1 is WRONG here
    Conv2d layers that use this module must be built with padding=0.
    """
    def __init__(self, stride):
        super().__init__()
        self.pad = (1, 1, 1, 1) if stride == 1 else (0, 1, 0, 1)   # (left, right, top, bottom)
    def forward(self, x):
        return F.pad(x, self.pad)
    def extra_repr(self):
        return f"pad(l,r,t,b)={self.pad}"


class Stem(nn.Module):
    """Standard 3x3 conv, stride 2, 3 -> 32*alpha channels, + BN + ReLU6."""
    def __init__(self, alpha=0.25):
        super().__init__()
        out_ch = make_divisible(32 * alpha)  # 8 for alpha=0.25
        self.pad = SamePad3x3(stride=2)      # TF 'SAME': (0, 1, 0, 1), not symmetric 1
        self.conv = nn.Conv2d(3, out_ch, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(x))))


class DepthwiseConv3x3(nn.Module):
    """
    3x3 depthwise convolution: one 3x3 filter per input channel (groups=in_ch),
    so channel count is unchanged. Followed by BN + ReLU6 as in MobileNet V1.
    """
    def __init__(self, channels, stride=1):
        super().__init__()
        self.pad = SamePad3x3(stride)   # TF 'SAME' padding; asymmetric for stride 2
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=stride, padding=0,
            groups=channels,   # <-- this is what makes it depthwise
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU6(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(x))))


class PointwiseConv1x1(nn.Module):
    """
    1x1 pointwise convolution: mixes channels, changes channel count in_ch -> out_ch,
    leaves spatial size untouched. Followed by BN + ReLU6 as in MobileNet V1.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=1, padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    """
    MobileNet V1 building block:
        3x3 depthwise conv (stride s) + BN + ReLU6
        1x1 pointwise conv (in_ch -> out_ch) + BN + ReLU6
    Spatial size is divided by `stride`; channel count goes in_ch -> out_ch.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = DepthwiseConv3x3(channels=in_ch, stride=stride)
        self.pw = PointwiseConv1x1(in_ch=in_ch, out_ch=out_ch)
    def forward(self, x):
        return self.pw(self.dw(x))


class ClassifierHead(nn.Module):
    """
    MobileNet V1 head (inference):
        global average pool -> flatten -> fully connected
    in_ch = 1024 * alpha = 256 for alpha=0.25.
    """
    def __init__(self, in_ch, num_classes=1000):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)
    def forward(self, x):
        x = self.avgpool(x)          # [N, C, 1, 1]
        x = torch.flatten(x, 1)      # [N, C]
        return self.fc(x)            # [N, num_classes]


class MobileNetV1(nn.Module):
    """
    MobileNet V1 (Howard et al., 2017) with width multiplier alpha.

    Architecture
    ------------
    A single standard 3x3 conv (the stem, stride 2) followed by 13
    depthwise-separable blocks, then global average pooling and a fully
    connected classifier. Each depthwise-separable block is a 3x3 depthwise
    conv (which carries the stride) followed by a 1x1 pointwise conv (which
    changes the channel count). Every conv is followed by BatchNorm + ReLU6.

    Five of the blocks use stride 2, so together with the stem the input is
    downsampled by 2^6 = 64 (224 -> 7) before pooling. All 3x3 convs use
    TensorFlow 'SAME' padding (SamePad3x3), which for stride 2 pads only the
    bottom/right; this is required to match the TF/Keras/TFLite reference.

    Width multiplier
    ----------------
    BLOCK_SPECS stores the channel counts of the original alpha=1.0 network.
    At build time every count is multiplied by alpha and rounded to a
    multiple of 8, so parameters and MACs scale roughly with alpha^2.
    Alpha does not affect the 3 input channels, num_classes, kernel sizes,
    strides, or depth.

    Shapes for alpha=0.25, 224x224x3 input
    --------------------------------------
        layer        stride   channels        output (H x W)
        -----------  ------   -------------   --------------
        stem            2       3 ->   8        112 x 112
        block  1        1       8 ->  16        112 x 112
        block  2        2      16 ->  32         56 x  56
        block  3        1      32 ->  32         56 x  56
        block  4        2      32 ->  64         28 x  28
        block  5        1      64 ->  64         28 x  28
        block  6        2      64 -> 128         14 x  14
        block  7-11     1     128 -> 128         14 x  14   (5 blocks)
        block 12        2     128 -> 256          7 x   7
        block 13        1     256 -> 256          7 x   7
        avgpool         -     256                 1 x   1
        fc              -     256 -> num_classes

    Parameters: 470,072 with num_classes=1000 (257,000 of that is the fc layer),
    versus 4,231,976 for alpha=1.0.

    Args:
        num_classes: size of the final fc output.
        alpha:       width multiplier in (0, 1]. 0.25 is the smallest
                     variant published in the paper.
    """
    # (output channels at alpha=1.0, stride) for each of the 13 depthwise-separable blocks
    BLOCK_SPECS = [
        (64, 1),
        (128, 2), (128, 1),
        (256, 2), (256, 1),
        (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
        (1024, 2), (1024, 1),
    ]

    def __init__(self, num_classes=1000, alpha=0.25):
        super().__init__()
        self.stem = Stem(alpha=alpha)

        blocks = []
        in_ch = make_divisible(32 * alpha)          # stem output, 8
        for base_out_ch, stride in self.BLOCK_SPECS:
            out_ch = make_divisible(base_out_ch * alpha)      # alpha applied here, once per block
            blocks.append(DepthwiseSeparable(in_ch=in_ch, out_ch=out_ch, stride=stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.head = ClassifierHead(in_ch=in_ch, num_classes=num_classes)   # in_ch == 256

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
