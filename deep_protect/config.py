from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ModelPaths:
    """Paths to the checkpoints and the pre-computed feature bank."""

    stylegan: Path
    e4e: Path
    arcface: Path
    farl: Path
    feature_bank: Optional[Path] = None


@dataclass
class DeepProtectConfig:
    """Hyperparameters reported in the CVPR 2026 paper.

    ``middle_style_indices`` are zero based.  The paper describes StyleGAN2
    layers 3--7, hence the implementation uses indices 2--6.
    """

    tau: float = 0.75
    candidate_count: int = 30
    order_regularization_weight: float = 1.0
    middle_style_indices: Tuple[int, ...] = (2, 3, 4, 5, 6)
    lora_blocks: Tuple[str, ...] = ("b16", "b32")
    lora_rank: int = 8
    lora_scale: float = 1.0
    learning_rate: float = 3e-4
    identity_lock_weight: float = 0.1
    max_optimization_steps: int = 450
    lpips_threshold: float = 0.06
    lpips_network: str = "alex"
    watermark_epsilon: float = 0.02
    watermark_step_size: float = 1.0 / 255.0
    watermark_steps: int = 44
    image_size: int = 1024
    feature_batch_size: int = 8
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = False
    output_dir: Path = Path("outputs")

    def validate(self) -> None:
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1].")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive.")
        if self.order_regularization_weight < 0:
            raise ValueError("order_regularization_weight must be non-negative.")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.identity_lock_weight < 0:
            raise ValueError("identity_lock_weight must be non-negative.")
        if self.max_optimization_steps <= 0:
            raise ValueError("max_optimization_steps must be positive.")
        if not 0 < self.watermark_epsilon:
            raise ValueError("watermark_epsilon must be positive.")
        if self.watermark_steps <= 0:
            raise ValueError("watermark_steps must be positive.")


def resolve_device(requested: str) -> str:
    """Resolve ``cuda`` to a usable device while keeping CPU smoke tests easy."""

    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested
