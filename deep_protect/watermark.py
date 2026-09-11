from dataclasses import dataclass

import torch

from .models.arcface import extract_arcface, extract_arcface_with_grad
from .oal_da import OrderAwareLDA
from .utils import unit_normalize


@dataclass
class WatermarkResult:
    image_01: torch.Tensor
    perturbation: torch.Tensor
    direction: torch.Tensor
    target_sign: float
    steps: int


def attribute_direction(bank, text_encoder, prompt: str, m: int = 30, lambda_reg: float = 1.0):
    """Retrieve top/bottom FaRL samples and fit v_attr in identity space."""

    text_feature = text_encoder.encode_text(prompt).detach().float().cpu().numpy()[0]
    clip_features = bank.clip_features.detach().float().cpu().numpy()
    id_features = bank.id_features.detach().float().cpu().numpy()
    scores = clip_features @ text_feature
    lda = OrderAwareLDA(lambda_reg=lambda_reg).fit(id_features, scores, m=m)
    direction = torch.from_numpy(lda.direction).to(bank.id_features.device)
    return unit_normalize(direction), lda, scores


def adversarial_watermark(
    image_01: torch.Tensor,
    identity_model,
    attribute_vector: torch.Tensor,
    epsilon: float = 0.02,
    step_size: float = 1.0 / 255.0,
    steps: int = 44,
) -> WatermarkResult:
    """Implement Eqs. (9)--(11) with sign-gradient updates in [0, 1]."""

    if image_01.ndim == 3:
        image_01 = image_01.unsqueeze(0)
    base = image_01.detach().clamp(0, 1)
    with torch.no_grad():
        source_feature = extract_arcface(identity_model, base)
    vector = unit_normalize(attribute_vector.reshape(1, -1).to(base.device))
    projection = (source_feature * vector).sum(dim=1, keepdim=True)
    target_sign = -torch.sign(projection)
    target_sign[target_sign == 0] = 1
    target = target_sign * vector

    delta = torch.zeros_like(base)
    for _ in range(steps):
        adv = (base + delta).clamp(0, 1).detach().requires_grad_(True)
        feature = extract_arcface_with_grad(identity_model, adv)
        objective = (feature * target).sum()
        gradient = torch.autograd.grad(objective, adv, only_inputs=True)[0]
        delta = (adv.detach() + step_size * gradient.sign() - base).clamp(-epsilon, epsilon)
    result = (base + delta).clamp(0, 1).detach()
    return WatermarkResult(result, delta.detach(), vector[0].detach(), float(target_sign[0].item()), steps)

