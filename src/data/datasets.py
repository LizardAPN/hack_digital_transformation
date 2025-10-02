"""
Geo dataset and geodesic utility.
"""

import math
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class GeoDataset(Dataset):
    """A torch Dataset for geotagged images stored in a CSV with columns path,lat,lon."""

    def __init__(self, csv_file: str, preprocess=None, img_size: int = 224):
        df = pd.read_csv(csv_file)
        if not {"path", "lat", "lon"}.issubset(df.columns):
            raise ValueError(f"CSV {csv_file} must contain 'path', 'lat' and 'lon' columns.")
        self.df = df.reset_index(drop=True)
        self.paths: List[str] = self.df["path"].astype(str).tolist()
        self.lats: np.ndarray = self.df["lat"].astype(float).to_numpy()
        self.lons: np.ndarray = self.df["lon"].astype(float).to_numpy()
        self.preprocess = preprocess
        self.img_size = img_size
        if self.preprocess is None:
            try:
                from torchvision import transforms as T

                self.fallback = T.Compose(
                    [
                        T.Resize(int(self.img_size * 1.14)),
                        T.CenterCrop(self.img_size),
                        T.ToTensor(),
                        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                    ]
                )
            except Exception:
                self.fallback = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        if self.preprocess is not None:
            tensor = self.preprocess(image)
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
                tensor = tensor.squeeze(0)
        else:
            if self.fallback is None:
                raise RuntimeError("No preprocess provided and torchvision not available")
            tensor = self.fallback(image)
        gps = torch.tensor([self.lats[idx], self.lons[idx]], dtype=torch.float32)
        return tensor, gps, path


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
