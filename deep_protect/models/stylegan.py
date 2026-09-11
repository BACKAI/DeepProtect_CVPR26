import copy
import importlib
import pickle
import sys
import types
from pathlib import Path

import torch


def load_stylegan(path: Path, device: str = "cuda"):
    """Load an official StyleGAN2-ADA FFHQ pickle.

    The official pickle refers to ``training.networks``.  This repository
    bundles the official network module at the project root, so the alias
    below keeps the pickle compatible without changing the original project.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"StyleGAN checkpoint not found: {path}")

    networks = importlib.import_module("networks")
    training_package = sys.modules.get("training")
    if training_package is None:
        training_package = types.ModuleType("training")
        sys.modules["training"] = training_package
    training_package.networks = networks
    sys.modules.setdefault("training.networks", networks)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        for key in ("G_ema", "G", "generator"):
            if key in payload:
                generator = payload[key]
                break
        else:
            raise KeyError("StyleGAN pickle must contain G_ema, G, or generator.")
    else:
        generator = payload
    return generator.to(device).eval().float()


def clone_generator(generator):
    return copy.deepcopy(generator).eval()
