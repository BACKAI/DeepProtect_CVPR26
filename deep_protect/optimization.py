from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .models.arcface import extract_arcface, extract_arcface_with_grad
from .models.lora import inject_lora, lora_state_dict
from .perceptual import lpips_scalar, make_lpips


@dataclass
class OptimizationResult:
    generator: torch.nn.Module
    lora_state: Dict[str, torch.Tensor]
    initial_image: torch.Tensor
    optimized_image: torch.Tensor
    steps: int
    final_l2: float
    final_lpips: float
    final_identity_lock: float
    injected_modules: list


def optimize_generator(
    original_generator,
    blended_w: torch.Tensor,
    source_image_01: torch.Tensor,
    identity_model,
    *,
    blocks=("b16", "b32"),
    rank: int = 8,
    lora_scale: float = 1.0,
    learning_rate: float = 3e-4,
    identity_lock_weight: float = 0.1,
    max_steps: int = 450,
    lpips_threshold: float = 0.06,
    lpips_network: str = "alex",
    device: str = "cuda",
    verbose: bool = True,
) -> OptimizationResult:
    """Optimize Eq. (4) with partial StyleGAN LoRA tuning."""

    tuned = original_generator
    injected = inject_lora(tuned, blocks=blocks, rank=rank, scale=lora_scale)
    tuned.train()
    original_generator.eval()
    source = source_image_01.to(device).clamp(0, 1)
    source_norm = source * 2.0 - 1.0
    with torch.no_grad():
        initial = tuned.synthesis(blended_w.to(device), noise_mode="const", force_fp32=True).detach()
        initial_norm = initial
        target_identity = extract_arcface(identity_model, (initial_norm + 1.0) / 2.0).detach()

    lpips_model = make_lpips(lpips_network, device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in tuned.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    final_values = (float("inf"), float("inf"), float("inf"))
    completed = 0
    for step in range(max_steps):
        generated = tuned.synthesis(blended_w.to(device), noise_mode="const", force_fp32=True)
        l2 = F.mse_loss(generated, source_norm)
        perceptual = lpips_scalar(lpips_model, generated, source_norm)
        generated_identity = extract_arcface_with_grad(identity_model, (generated + 1.0) / 2.0)
        identity_lock = 1.0 - (target_identity * generated_identity).sum(dim=1).mean()
        loss = l2 + perceptual + identity_lock_weight * identity_lock

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        completed = step + 1
        final_values = (float(l2.detach()), float(perceptual.detach()), float(identity_lock.detach()))
        if verbose and (step == 0 or completed % 25 == 0):
            print(
                f"[generator] step {completed:03d}/{max_steps} "
                f"loss={float(loss.detach()):.5f} l2={final_values[0]:.5f} "
                f"lpips={final_values[1]:.5f} id_lock={final_values[2]:.5f}"
            )
        if final_values[1] <= lpips_threshold:
            break

    with torch.no_grad():
        optimized = tuned.synthesis(blended_w.to(device), noise_mode="const", force_fp32=True).detach()
    return OptimizationResult(
        tuned,
        lora_state_dict(tuned),
        initial.detach(),
        optimized,
        completed,
        final_values[0],
        final_values[1],
        final_values[2],
        injected,
    )
