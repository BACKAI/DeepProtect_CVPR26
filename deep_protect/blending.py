from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .feature_bank import FeatureBank


@dataclass
class BlendResult:
    latent: torch.Tensor
    candidate_indices: torch.Tensor
    clip_similarities: torch.Tensor


@torch.no_grad()
def identity_blend(
    w_plus: torch.Tensor,
    image_clip_feature: torch.Tensor,
    bank: FeatureBank,
    tau: float = 0.75,
    middle_style_indices=(2, 3, 4, 5, 6),
    candidate_count: int = 30,
) -> BlendResult:
    """Apply Eq. (1)--(2): retrieve and substitute one latent per style channel."""

    if w_plus.ndim == 2:
        w_plus = w_plus.unsqueeze(0)
    if w_plus.shape[0] != 1:
        raise ValueError("identity_blend currently processes one source image at a time.")
    query = F.normalize(image_clip_feature.reshape(1, -1).float(), dim=1)
    bank_clip = F.normalize(bank.clip_features.float(), dim=1)
    similarities = (bank_clip @ query.T).squeeze(1)
    candidates = torch.nonzero(similarities >= tau, as_tuple=False).flatten()
    if candidates.numel() == 0:
        candidates = torch.argsort(similarities, descending=True)[:candidate_count]
    else:
        candidates = candidates[torch.argsort(similarities[candidates], descending=True)]
        candidates = candidates[: max(candidate_count, 1)]

    fused = w_plus.clone()
    normalized_query_styles = F.normalize(fused[:, list(middle_style_indices), :], dim=-1)
    candidate_latents = bank.latents[candidates]
    normalized_candidates = F.normalize(candidate_latents[:, list(middle_style_indices), :], dim=-1)
    for local_idx, style_idx in enumerate(middle_style_indices):
        scores = normalized_candidates[:, local_idx, :] @ normalized_query_styles[0, local_idx, :]
        best = candidates[torch.argmax(scores)]
        # Eq. (2) is substitution, not interpolation.  This is the important
        # difference from the older incomplete codebase.
        fused[:, style_idx, :] = bank.latents[best, style_idx, :]
    return BlendResult(fused, candidates, similarities[candidates])

