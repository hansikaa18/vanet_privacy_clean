"""
models.py
Neural network architecture used by all vehicles.
Deeper than a simple MLP to give better attack detection capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttackDetector(nn.Module):
    """
    3-layer MLP with batch norm and dropout.
    Input: 4 features (pos_zscore, speed_anomaly, heading_dev, time_delta)
    Output: 2 classes (legit, attack)
    """

    def __init__(self, input_dim=4, hidden_dims=(64, 32, 16), num_classes=2, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_flat_params(self):
        return np.concatenate([p.data.cpu().numpy().flatten()
                                for p in self.parameters()])

    def set_flat_params(self, flat_params):
        idx = 0
        with torch.no_grad():
            for p in self.parameters():
                numel = p.numel()
                p.data.copy_(
                    torch.tensor(flat_params[idx:idx + numel],
                                 dtype=p.dtype).reshape(p.shape)
                )
                idx += numel

    def get_flat_grads(self):
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(p.grad.cpu().numpy().flatten())
            else:
                grads.append(np.zeros(p.numel()))
        return np.concatenate(grads)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
