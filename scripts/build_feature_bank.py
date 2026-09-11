#!/usr/bin/env python3
"""Build the N=4,605 VGGFace2-HQ feature bank used by DeepProtect."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from deep_protect.config import resolve_device
from deep_protect.feature_bank import FeatureBank, save_feature_bank
from deep_protect.models.arcface import extract_arcface, load_arcface
from deep_protect.models.e4e import E4EEncoder
from deep_protect.models.farl import FaRLTextImageEncoder
from deep_protect.utils import iter_image_paths, pil_to_tensor, seed_everything


def gallery_images(root: Path, max_identities: int):
    """Select one deterministic image per identity directory, as in the paper."""

    paths = list(iter_image_paths(root))
    grouped = {}
    for path in paths:
        relative_parent = path.parent.relative_to(root) if path.parent != root else Path(path.stem)
        identity = str(relative_parent.parts[0] if relative_parent.parts else path.stem)
        grouped.setdefault(identity, []).append(path)
    selected = [sorted(values)[0] for _, values in sorted(grouped.items())]
    return selected[:max_identities]


def parse_args():
    parser = argparse.ArgumentParser(description="Build DeepProtect's HDF5 feature bank")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help=".h5 or .npz bank path")
    parser.add_argument("--e4e", type=Path, required=True)
    parser.add_argument("--arcface", type=Path, required=True)
    parser.add_argument("--farl", type=Path, required=True)
    parser.add_argument("--max-identities", type=int, default=4605)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    device = resolve_device(args.device)
    paths = gallery_images(args.images, args.max_identities)
    if len(paths) < 2:
        raise RuntimeError("At least two gallery identities are required.")
    e4e = E4EEncoder(args.e4e, device=device)
    arcface = load_arcface(args.arcface, device=device)
    farl = FaRLTextImageEncoder(args.farl, device=device)
    latents, clips, ids, names = [], [], [], []
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        image_01 = torch.stack([pil_to_tensor(image, 1024) for image in images]).add(1).div(2).to(device)
        with torch.no_grad():
            latents.append(e4e.encode(image_01).cpu())
            clips.append(farl.encode_images(images).cpu())
            ids.append(extract_arcface(arcface, image_01).cpu())
        names.extend(str(path) for path in batch_paths)
        print(f"[{min(start + len(batch_paths), len(paths))}/{len(paths)}]")
    bank = FeatureBank(torch.cat(latents), torch.cat(clips), torch.cat(ids), names)
    save_feature_bank(args.output, bank)
    print(f"Saved {bank.size} identities to {args.output}")


if __name__ == "__main__":
    main()
