#!/usr/bin/env python3
"""Run the per-image DeepProtect optimization over an aligned image directory."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_protect.config import DeepProtectConfig, ModelPaths
from deep_protect.pipeline import DeepProtectPipeline
from deep_protect.utils import iter_image_paths, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="DeepProtect CVPR 2026 per-image training")
    parser.add_argument("--input-dir", type=Path, required=True, help="Aligned CelebA-HQ/VGGFace2-HQ source directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-bank", type=Path, required=True)
    parser.add_argument("--stylegan", type=Path, required=True)
    parser.add_argument("--e4e", type=Path, required=True)
    parser.add_argument("--arcface", type=Path, required=True)
    parser.add_argument("--farl", type=Path, required=True)
    parser.add_argument("--prompt", default="nose")
    parser.add_argument("--mode", choices=("combined", "attribute"), default="combined")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-optimization-steps", type=int, default=450)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    paths = ModelPaths(args.stylegan, args.e4e, args.arcface, args.farl, args.feature_bank)
    config = DeepProtectConfig(
        max_optimization_steps=args.max_optimization_steps,
        device=args.device,
    )
    pipeline = DeepProtectPipeline(paths, config)
    image_paths = list(iter_image_paths(args.input_dir))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found in {args.input_dir}")
    for index, image_path in enumerate(image_paths, start=1):
        output = args.output_dir / image_path.relative_to(args.input_dir).with_suffix("")
        if not args.quiet:
            print(f"[{index}/{len(image_paths)}] {image_path}")
        pipeline.protect(
            image_path,
            output,
            prompt=args.prompt,
            mode=args.mode,
            optimize=True,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
