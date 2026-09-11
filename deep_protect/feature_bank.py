from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

from .utils import unit_normalize


@dataclass
class FeatureBank:
    """The N-entry bank used by both identity blending and attribute retrieval."""

    latents: torch.Tensor       # [N, 18, 512]
    clip_features: torch.Tensor # [N, D]
    id_features: torch.Tensor   # [N, 512]
    names: Optional[List[str]] = None

    def __post_init__(self):
        if self.latents.ndim != 3 or self.latents.shape[1:] != (18, 512):
            raise ValueError(f"latents must have shape [N, 18, 512], got {tuple(self.latents.shape)}")
        if self.clip_features.shape[0] != self.latents.shape[0]:
            raise ValueError("latents and clip_features must have the same N.")
        if self.id_features.shape != (self.latents.shape[0], 512):
            raise ValueError("id_features must have shape [N, 512].")
        self.latents = self.latents.float().contiguous()
        self.clip_features = unit_normalize(self.clip_features.float())
        self.id_features = unit_normalize(self.id_features.float())
        if self.names is not None and len(self.names) != self.latents.shape[0]:
            raise ValueError("names must have one entry per bank sample.")

    @property
    def size(self) -> int:
        return int(self.latents.shape[0])

    def to(self, device: str):
        return FeatureBank(
            self.latents.to(device), self.clip_features.to(device), self.id_features.to(device), self.names
        )


def _first_key(handle, keys):
    for key in keys:
        if key in handle:
            return key
    raise KeyError(f"None of the keys {keys} were found. Available keys: {list(handle.keys())}")


def load_feature_bank(path: Path, device: str = "cpu") -> FeatureBank:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature bank not found: {path}")
    names = None
    if path.suffix.lower() in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Loading HDF5 feature banks requires h5py.") from exc
        with h5py.File(path, "r") as handle:
            latent_key = _first_key(handle, ("latents", "latent_codes", "gallery_e4e"))
            clip_key = _first_key(handle, ("clip_features", "gallery_clip"))
            id_key = _first_key(handle, ("id_features", "identity_features", "gallery_f"))
            latents = torch.from_numpy(np.asarray(handle[latent_key]))
            clip_features = torch.from_numpy(np.asarray(handle[clip_key]))
            id_features = torch.from_numpy(np.asarray(handle[id_key]))
            if "names" in handle or "gallery_name" in handle:
                name_key = "names" if "names" in handle else "gallery_name"
                names = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in handle[name_key][:]]
    elif path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=True)
        latents = torch.from_numpy(payload[_first_key(payload, ("latents", "latent_codes", "gallery_e4e"))])
        clip_features = torch.from_numpy(payload[_first_key(payload, ("clip_features", "gallery_clip"))])
        id_features = torch.from_numpy(payload[_first_key(payload, ("id_features", "identity_features", "gallery_f"))])
        for key in ("names", "gallery_name"):
            if key in payload:
                names = [str(item) for item in payload[key]]
                break
    else:
        raise ValueError("Feature bank must be .h5, .hdf5, or .npz.")
    return FeatureBank(latents, clip_features, id_features, names).to(device)


def save_feature_bank(path: Path, bank: FeatureBank) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Saving HDF5 feature banks requires h5py.") from exc
        with h5py.File(path, "w") as handle:
            handle.create_dataset("latents", data=bank.latents.cpu().numpy(), compression="gzip")
            handle.create_dataset("clip_features", data=bank.clip_features.cpu().numpy(), compression="gzip")
            handle.create_dataset("id_features", data=bank.id_features.cpu().numpy(), compression="gzip")
            if bank.names is not None:
                handle.create_dataset("names", data=np.asarray(bank.names, dtype="S"))
    elif path.suffix.lower() == ".npz":
        values = dict(
            latents=bank.latents.cpu().numpy(),
            clip_features=bank.clip_features.cpu().numpy(),
            id_features=bank.id_features.cpu().numpy(),
        )
        if bank.names is not None:
            values["names"] = np.asarray(bank.names)
        np.savez_compressed(path, **values)
    else:
        raise ValueError("Feature bank must be .h5, .hdf5, or .npz.")

