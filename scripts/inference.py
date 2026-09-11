#!/usr/bin/env python3
"""Run DeepProtect on one aligned face image."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_protect.config import DeepProtectConfig, ModelPaths
from deep_protect.pipeline import DeepProtectPipeline
from deep_protect.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="DeepProtect CVPR 2026 inference")
    parser.add_argument("--image", type=Path, required=True, help="Aligned 1024x1024 RGB face image")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-bank", type=Path, required=True)
    parser.add_argument("--stylegan", type=Path, required=True)
    parser.add_argument("--e4e", type=Path, required=True)
    parser.add_argument("--arcface", type=Path, required=True)
    parser.add_argument("--farl", type=Path, required=True)
    parser.add_argument("--prompt", default="nose")
    parser.add_argument("--mode", choices=("combined", "attribute"), default="combined")
    parser.add_argument("--no-generator-optimization", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tau", type=float, default=0.75)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--order-regularization-weight", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-optimization-steps", type=int, default=450)
    parser.add_argument("--lpips-threshold", type=float, default=0.06)
    parser.add_argument("--watermark-epsilon", type=float, default=0.02)
    parser.add_argument("--watermark-steps", type=int, default=44)
    parser.add_argument("--watermark-step-size", type=float, default=1.0 / 255.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    paths = ModelPaths(args.stylegan, args.e4e, args.arcface, args.farl, args.feature_bank)
    config = DeepProtectConfig(
        tau=args.tau,
        candidate_count=args.candidate_count,
        order_regularization_weight=args.order_regularization_weight,
        lora_rank=args.lora_rank,
        max_optimization_steps=args.max_optimization_steps,
        lpips_threshold=args.lpips_threshold,
        watermark_epsilon=args.watermark_epsilon,
        watermark_steps=args.watermark_steps,
        watermark_step_size=args.watermark_step_size,
        device=args.device,
    )
    pipeline = DeepProtectPipeline(paths, config)
    pipeline.protect(
        args.image,
        args.output_dir,
        prompt=args.prompt,
        mode=args.mode,
        optimize=not args.no_generator_optimization,
        verbose=not args.quiet,
    )
    print(f"Saved DeepProtect artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
