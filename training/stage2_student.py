"""Project encoder and trainable network derived from the uttt.ai stage-2 PVN.

Upstream: https://github.com/arnowaczynski/utttai/blob/622dae66b18198f6f12aef0a0bccd5bd40de2d8e/utttpy/selfplay/policy_value_network.py
License: Apache-2.0 (see licenses/utttai-Apache-2.0.txt).
Changes: use inference-folded affine layers, our move ordering and optional
activation-calibrated low-rank convolutions. The compact graph directly
executes its factors and has no operation to reconstruct a dense kernel.
"""

import torch
from torch import nn
from torch.nn import functional as F


class FactoredPoint(nn.Module):
    def __init__(self, inputs, outputs, rank):
        super().__init__()
        self.reduce = nn.Conv2d(inputs, rank, 1, bias=False)
        self.expand = nn.Conv2d(rank, outputs, 1)

    def forward(self, x):
        return self.expand(self.reduce(x))


class FactoredLift(nn.Module):
    """Two channel projections and a small transposed convolution on three-by-three tiles."""

    def __init__(self, inputs, outputs, stride, input_rank, output_rank, *, broadcast_offset=False):
        super().__init__()
        self.reduce = nn.Conv2d(inputs, input_rank, 1, bias=False)
        self.lift = nn.ConvTranspose2d(input_rank, output_rank, 3, stride=stride, bias=False)
        self.expand = nn.Conv2d(output_rank, outputs, 1, bias=False)
        self.offset = nn.Parameter(torch.zeros(1, outputs, 3, 3))
        self.stride = stride
        self.broadcast_offset = broadcast_offset

    def forward(self, x):
        output = self.expand(self.lift(self.reduce(x)))
        if self.broadcast_offset:
            # Keep one 3x3 offset in the packing source graph. Repeating the
            # parameter into a 9x9 initializer can consume another 73,728 bytes.
            tiles = output.reshape(-1, self.expand.out_channels, 3, 3, 3, 3)
            return (tiles + self.offset[:, :, None, :, None, :]).reshape_as(output)
        offset = self.offset.repeat(1, 1, 3, 3) if self.stride == 3 else self.offset
        return output + offset


class Stage2Net(nn.Module):
    """Full teacher, compact float32 student, or moderate model for subsequent packing."""

    def __init__(self, compact=True, *, moderate=False):
        super().__init__()
        if moderate and not compact:
            raise ValueError("The moderate rank profile applies to a factorized student")
        self.register_buffer("order", torch.arange(81).reshape(3, 3, 3, 3).transpose(1, 2).reshape(81))

        def point(n, m, rank):
            return FactoredPoint(n, m, rank) if compact and not moderate else nn.Conv2d(n, m, 1)

        def lift(n, m, stride, i, o):
            if not compact:
                return nn.ConvTranspose2d(n, m, 3, stride)
            if moderate:
                # Retain each dense gate and both value paths exactly. The four
                # transposed kernels account for 4.13M of the original 4.99M weights.
                i, o = (160, 144) if stride == 1 else (192, 176)
            return FactoredLift(n, m, stride, i, o, broadcast_offset=moderate and stride == 3)

        layers = {
            "stem": nn.Conv2d(4, 128, 3, 3),
            "early": point(128, 128, 32), "trunk": point(128, 256, 48),
        }
        for i in range(3):
            layers[f"gate{i}"] = point(256, 512, 24)
            layers[f"pool{i}"] = nn.Conv2d(512, 512, 3, groups=512)
            layers[f"lift{i}"] = lift(512, 256, 1, 40, 32)
        layers.update({
            "policy": point(640, 256, 40), "upsample": lift(256, 256, 3, 40, 24),
            "logits": point(260, 128, 32), "logit_out": nn.Conv2d(128, 1, 1),
            "v1": point(256, 256, 24), "vpool1": nn.Conv2d(256, 256, 3, groups=256),
            "v2": point(256, 256, 24), "vpool2": nn.Conv2d(256, 256, 3, groups=256),
            "value": point(512, 128, 32), "value_out": nn.Conv2d(128, 1, 1),
        })
        self.layers = nn.ModuleDict(layers)

    def forward(self, state):
        own = state[:, :81].index_select(1, self.order)
        other = state[:, 81:162].index_select(1, self.order)
        legal = state[:, 162:243].index_select(1, self.order)
        turn = 1 + 2 * (own.sum(1, keepdim=True) - other.sum(1, keepdim=True))
        image = torch.stack((own, other, turn.expand_as(own), legal), dim=1).reshape(-1, 4, 9, 9)
        layer = self.layers
        stem = F.elu(layer["stem"](image))
        trunk = F.relu(layer["trunk"](F.elu(layer["early"](stem))))
        final = trunk
        for i in range(3):
            gate = F.elu(layer[f"gate{i}"](final))
            pooled = F.elu(layer[f"pool{i}"](gate))
            final = F.relu(final + layer[f"lift{i}"](pooled))
        context = F.elu(layer["policy"](torch.cat((stem, trunk, final), dim=1)))
        spatial = F.elu(layer["upsample"](context))
        policy = layer["logit_out"](F.elu(layer["logits"](torch.cat((image, spatial), dim=1))))
        logits = policy.clamp(-32, 32).flatten(1).index_select(1, self.order)
        v1 = F.elu(layer["vpool1"](layer["v1"](trunk)))
        v2 = F.elu(layer["vpool2"](layer["v2"](final)))
        hidden = F.elu(layer["value"](torch.cat((v1, v2), dim=1)))
        return logits, layer["value_out"](hidden).tanh().flatten(1)
