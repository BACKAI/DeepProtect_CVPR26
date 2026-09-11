import json
import random
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def iter_image_paths(root: Path) -> Iterator[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
        yield root
        return
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def pil_to_tensor(image: Image.Image, size: int = 1024):
    """Convert RGB PIL image to float tensor in [-1, 1], without torchvision."""

    import torch

    image = image.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil(image) -> Image.Image:
    import torch

    if image.ndim == 4:
        image = image[0]
    image = image.detach().float().cpu().clamp(-1, 1)
    array = ((image.permute(1, 2, 0).numpy() + 1.0) * 127.5).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def unit_normalize(x, dim=-1, eps=1e-8):
    import torch

    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

