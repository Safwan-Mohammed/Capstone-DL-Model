import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class PixelSetData(Dataset):
    def __init__(self, folder, npixel=1, sub_classes=None, norm=None, extra_feature=None):
        """
        Args:
            folder: Path to dataset (e.g., '../PRED_DATA')
            npixel: Number of pixels per parcel (default: 1)
            sub_classes: Ignored (None for binary task)
            norm: Tuple of (mean, std) for normalization
            extra_feature: None for your case
        """
        super(PixelSetData, self).__init__()
        self.folder = folder
        # Load parcel IDs from .npy files
        self.pid = []
        data_dir = os.path.join(folder, 'DATA')
        for p in os.listdir(data_dir):
            if p.endswith('.npy'):
                self.pid.append(p[:-4])  # e.g., 'lon_lat'

        if not self.pid:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")

        self.len = len(self.pid)

        # Load dates
        dates_path = os.path.join(folder, 'META', 'dates.json')
        if not os.path.exists(dates_path):
            raise FileNotFoundError(f"dates.json not found at {dates_path}")

        with open(dates_path, 'r') as file:
            date_d = json.load(file)

        # Validate that all pids have corresponding dates
        self.date_positions = []
        ref_date = np.datetime64('2019-07-01')
        for p in self.pid:
            if p not in date_d:
                raise ValueError(f"No dates found for pid {p} in dates.json")
            dates = [np.datetime64(d) for d in date_d[p]]
            date_pos = [(d - ref_date).astype('timedelta64[D]') / np.timedelta64(1, 'D') for d in dates]
            self.date_positions.append(date_pos)

        # Normalization
        self.norm = norm
        self.npixel = npixel
        self.extra = extra_feature
        if self.norm is not None:
            self.mean = torch.FloatTensor(self.norm[0]).view(1, -1, 1)  # [1, C, 1]
            self.std = torch.FloatTensor(self.norm[1]).view(1, -1, 1)
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path = os.path.join(self.folder, 'DATA', f'{self.pid[index]}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f".npy file not found: {path}")

        data = np.load(path, mmap_mode='r').astype(np.float32)  # [T, C, S]
        data = torch.from_numpy(data)  # [T, C, S]
        if self.norm is not None:
            data = (data - self.mean) / self.std
        pixels = data  # [T, C, S]
        mask = torch.ones(pixels.shape[0], pixels.shape[-1]).long()  # [T, S]
        dates = torch.FloatTensor(self.date_positions[index])  # [T]
        extra = torch.zeros(1, dtype=torch.float32)  # Placeholder tensor
        return (pixels, mask, dates, extra)