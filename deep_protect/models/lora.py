from typing import Iterable, List, Tuple

import torch
from torch import nn


class LoRAAffine(nn.Module):
    """Frozen StyleGAN affine transform plus a trainable low-rank update."""

    def __init__(self, base: nn.Module, rank: int = 8, scale: float = 1.0):
        super().__init__()
        if not hasattr(base, "weight"):
            raise TypeError("LoRA can only wrap a StyleGAN FullyConnectedLayer.")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        in_features = int(base.weight.shape[1])
        out_features = int(base.weight.shape[0])
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.scale = float(scale)
        nn.init.normal_(self.down.weight, std=1.0 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


def _replace_child(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def inject_lora(
    generator: nn.Module,
    blocks: Iterable[str] = ("b16", "b32"),
    rank: int = 8,
    scale: float = 1.0,
) -> List[str]:
    """Inject LoRA into conv0/conv1/ToRGB affines of selected StyleGAN blocks."""

    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    injected: List[str] = []
    for block_name in blocks:
        block = getattr(generator.synthesis, block_name, None)
        if block is None:
            raise ValueError(f"StyleGAN generator has no synthesis block {block_name!r}.")
        for layer_name in ("conv0", "conv1", "torgb"):
            layer = getattr(block, layer_name, None)
            if layer is None or not hasattr(layer, "affine"):
                continue
            path = f"synthesis.{block_name}.{layer_name}.affine"
            if isinstance(layer.affine, LoRAAffine):
                injected.append(path)
                continue
            _replace_child(generator, path, LoRAAffine(layer.affine, rank, scale))
            injected.append(path)

    if not injected:
        raise RuntimeError("No StyleGAN affine modules were selected for LoRA injection.")
    for name, parameter in generator.named_parameters():
        parameter.requires_grad_(".down.weight" in name or ".up.weight" in name)
    return injected


def lora_state_dict(generator: nn.Module):
    return {
        name: parameter.detach().cpu()
        for name, parameter in generator.named_parameters()
        if ".down.weight" in name or ".up.weight" in name
    }


def load_lora_state_dict(generator: nn.Module, state_dict, strict: bool = True):
    missing, unexpected = generator.load_state_dict(state_dict, strict=False)
    missing = [name for name in missing if ".down.weight" in name or ".up.weight" in name]
    if strict and (missing or unexpected):
        raise RuntimeError(f"LoRA checkpoint mismatch. Missing={missing}, unexpected={unexpected}")
    return missing, unexpected

