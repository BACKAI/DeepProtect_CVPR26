"""ArcFace IR-50 feature extractor and tolerant checkpoint loader."""

from collections import namedtuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(x, dim=1):
    return F.normalize(x, p=2, dim=dim)


class _Block(namedtuple("Block", ["in_channel", "depth", "stride"])):
    __slots__ = ()


def _get_blocks(num_layers=50):
    if num_layers == 50:
        return [
            [_Block(64, 64, 2)] + [_Block(64, 64, 1)] * 2,
            [_Block(64, 128, 2)] + [_Block(128, 128, 1)] * 3,
            [_Block(128, 256, 2)] + [_Block(256, 256, 1)] * 13,
            [_Block(256, 512, 2)] + [_Block(512, 512, 1)] * 2,
        ]
    raise ValueError("Only ArcFace IR-50 is supported by this loader.")


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Names mirror the common InsightFace IR-SE checkpoint layout.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.relu(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return x * scale


class IRSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if in_channels == out_channels:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            SEModule(out_channels),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class ArcFaceIRSE50(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
        )
        blocks = []
        for stage in _get_blocks(50):
            blocks.extend(IRSEBlock(b.in_channel, b.depth, b.stride) for b in stage)
        self.body = nn.Sequential(*blocks)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=False),
        )

    def forward(self, x):
        return l2_norm(self.output_layer(self.body(self.input_layer(x))))


def _unwrap_state_dict(payload):
    if isinstance(payload, nn.Module):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported ArcFace checkpoint type: {type(payload)!r}")
    for key in ("state_dict", "model", "backbone", "net"):
        if key in payload and isinstance(payload[key], dict):
            payload = payload[key]
            break
    state = {}
    for key, value in payload.items():
        key = key.removeprefix("module.")
        key = key.removeprefix("model.")
        state[key] = value
    return state


def load_arcface(path: Path, device: str = "cuda"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ArcFace checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, nn.Module):
        model = payload
    else:
        model = ArcFaceIRSE50()
        state = _unwrap_state_dict(payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 20:
            raise RuntimeError(
                "ArcFace checkpoint did not match IR-SE-50. "
                f"Missing {len(missing)} parameters and unexpected {len(unexpected)}."
            )
    return model.to(device).eval()


def arcface_input(image_01):
    """Match the crop used in the supplied implementation and ArcFace."""

    image_256 = F.interpolate(image_01, size=(256, 256), mode="bilinear", align_corners=False)
    image_218 = image_256[:, :, 19:237, 19:237]
    return F.interpolate(image_218, size=(112, 112), mode="bilinear", align_corners=True)


@torch.no_grad()
def extract_arcface(model, image_01):
    return l2_norm(model(arcface_input(image_01) * 2.0 - 1.0))


def extract_arcface_with_grad(model, image_01):
    return l2_norm(model(arcface_input(image_01) * 2.0 - 1.0))
