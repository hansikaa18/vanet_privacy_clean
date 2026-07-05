"""
vehicle.py
Implements the Vehicle Layer from the model slides:
  - RBAC + ε check
  - Local FL training (2 epochs)
  - Gradient clipping: g_clipped = g * min(1, C / ||g||)
  - ADP noise injection (Gaussian mechanism)
  - Budget tracking
"""

import numpy as np
import torch
import torch.nn as nn
from models import AttackDetector

ROLE_TRUST = {
    "infrastructure": 0.80,
    "fleet": 0.60,
    "private": 0.50,
}

ROLE_EPS_RANGE = {
    "infrastructure": (0.10, 1.50),
    "fleet":          (0.08, 1.20),
    "private":        (0.05, 1.00),
}


class Vehicle:
    def __init__(self, vid, role="private", eps_total=1.0, is_attacker=False,
                 attack_type=0):
        self.vid = vid
        self.role = role
        self.eps_total = eps_total
        self.eps_remaining = eps_total
        self.trust_score = ROLE_TRUST.get(role, 0.50)
        self.is_attacker = is_attacker
        self.attack_type = attack_type
        self.model = AttackDetector()
        self.rounds_done = 0

        # RBAC rights
        self.role_rights = "TRAIN_ONLY"
        self.eps_allocated = 0.0
        self.clip_C = 1.0

    def rbac_eps_check(self, eps_allocated, clip_C):
        """Returns True if vehicle may participate this round.
        Per Dwork & Roth (2014), budget resets each FL round.
        """
        if self.trust_score < 0.10:
            return False
        self.eps_remaining = self.eps_total   # reset each round
        self.eps_allocated = min(eps_allocated, self.eps_remaining)
        self.clip_C = clip_C
        return True

    def local_train(self, X, y, global_flat_params, epochs=2, lr=0.01,
                    batch_size=32, class_weights=None):
        """
        Load global weights, train locally, return DP-protected gradient dict.
        Attackers inject poisoned updates according to their attack_type.
        """
        self.model.set_flat_params(global_flat_params)
        self.model.train()

        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                     weight_decay=1e-4)

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)

        n = len(X)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                out = self.model(Xt[idx])
                loss = criterion(out, yt[idx])
                loss.backward()
                optimizer.step()

        grad = self.model.get_flat_grads()

        # ── Attacker poisoning ────────────────────────────────────────────
        if self.is_attacker:
            grad = self._poison_gradient(grad)

        # ── Gradient clipping ─────────────────────────────────────────────
        norm = np.linalg.norm(grad)
        if norm > self.clip_C:
            grad = grad * (self.clip_C / norm)

        # ── ADP noise injection (Gaussian mechanism) ──────────────────────
        delta = 1e-5
        sensitivity = self.clip_C
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / (self.eps_allocated + 1e-9)
        noise = np.random.normal(0, sigma, grad.shape)
        dp_grad = grad + noise

        # Budget consumption
        self.eps_remaining = max(0.0, self.eps_remaining - self.eps_allocated)
        self.rounds_done += 1

        # Sensitivity tier tag: fraction of high-magnitude components
        tier_tag = float(np.mean(np.abs(dp_grad) > np.percentile(np.abs(dp_grad), 75)))

        return {
            "vid": self.vid,
            "gradient": dp_grad,
            "tier_tag": tier_tag,
            "eps_consumed": self.eps_allocated,
            "n_samples": len(y),
            "probationary": self.trust_score < 0.40,
            "round_number": self.rounds_done,
        }

    def _poison_gradient(self, grad):
        """Simulate different attack gradient patterns per VeReMi attack class."""
        if self.attack_type == 1:
            # Constant position: completely wrong gradient direction
            return -grad * 2.0

        elif self.attack_type == 2:
            # Constant offset: systematic drift
            offset = np.ones_like(grad) * 0.5
            return grad + offset

        elif self.attack_type == 3:
            # Random position: ghost vehicle - random noise gradient
            return np.random.normal(0, np.linalg.norm(grad), grad.shape)

        elif self.attack_type == 4:
            # Random offset: noisy manipulation - amplified noise
            return grad + np.random.normal(0, np.linalg.norm(grad) * 0.8, grad.shape)

        elif self.attack_type == 5:
            # Slow drift: gradual direction shift (increases each round)
            drift_strength = min(0.05 * self.rounds_done, 0.8)
            drift_dir = np.random.normal(0, 1, grad.shape)
            drift_dir /= (np.linalg.norm(drift_dir) + 1e-9)
            return grad * (1 - drift_strength) + drift_dir * drift_strength * np.linalg.norm(grad)

        return grad
