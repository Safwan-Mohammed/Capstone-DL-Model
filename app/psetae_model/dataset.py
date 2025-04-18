import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class PixelSetData(Dataset):
    def __init__(self, folder, labels='labels', npixel=64, sub_classes=None, norm=None, extra_feature=None):
        """
        Args:
            folder: Path to dataset (e.g., './SAMPLE')
            labels: Label file name ('labels' for labels.json)
            npixel: Number of pixels per parcel (1 for your case)
            sub_classes: Ignored (None for binary task)
            norm: Tuple of (mean, std) for normalization
            extra_feature: None for your case
        """
        super(PixelSetData, self).__init__()
        self.folder = folder

        # Load parcel IDs from .npy files
        self.pid = []
        for p in os.listdir(os.path.join(folder, 'DATA')):
            if p.endswith('.npy'):
                self.pid.append(p[:-4])  # e.g., '0' from '0.npy'

        # Load labels
        with open(os.path.join(folder, 'META', 'labels.json'), 'r') as file:
            d = json.loads(file.read())
        self.target = []
        for i, p in enumerate(self.pid):
            t = int(d[p])  # Direct access to flat labels.json (e.g., d['0'] = 1)
            self.target.append(t)
        self.len = len(self.target)

        # Load dates
        with open(os.path.join(folder, 'META', 'dates.json'), 'r') as file:
            date_d = json.load(file)
        self.date_positions = []
        ref_date = np.datetime64('2019-01-01')
        for p in self.pid:
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
        data = np.load(path, mmap_mode='r').astype(np.float32)  # [T, C, S]
        data = torch.from_numpy(data)  # [T, C, S]
        if self.norm is not None:
            data = (data - self.mean) / self.std
        pixels = data  # [T, C, S]
        mask = torch.ones(pixels.shape[0], pixels.shape[-1]).long()  # [T, S]
        dates = torch.FloatTensor(self.date_positions[index])  # [T]
        extra = torch.zeros(1, dtype=torch.float32)  # Placeholder tensor
        target = self.target[index]
        return (pixels, mask, dates, extra), target