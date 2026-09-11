from pathlib import Path

import torch


class FaRLTextImageEncoder:
    """Load FaRL through the OpenAI CLIP-compatible API."""

    def __init__(self, checkpoint: Path, device: str = "cuda", model_name: str = "ViT-B/16"):
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "FaRL requires the CLIP-compatible package from the FaRL repository."
            ) from exc

        self.clip = clip
        self.model, self.preprocess = clip.load(model_name, device="cpu")
        checkpoint = Path(checkpoint)
        if checkpoint.is_file():
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            except TypeError:  # PyTorch < 2.6
                payload = torch.load(checkpoint, map_location="cpu")
            if isinstance(payload, dict) and "state_dict" in payload:
                payload = payload["state_dict"]
            if isinstance(payload, dict):
                payload = {k.removeprefix("module."): v for k, v in payload.items()}
                self.model.load_state_dict(payload, strict=False)
            elif isinstance(payload, torch.nn.Module):
                self.model = payload
            else:
                raise TypeError(f"Unsupported FaRL checkpoint type: {type(payload)!r}")
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode_text(self, prompt: str):
        tokens = self.clip.tokenize([prompt]).to(self.device)
        features = self.model.encode_text(tokens).float()
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def encode_images(self, images):
        """Encode a list of PIL images using the FaRL preprocessing pipeline."""

        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        features = self.model.encode_image(batch).float()
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
