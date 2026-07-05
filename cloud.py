"""
cloud.py
Implements the Cloud Layer (Global Digital Twin):
  - Global FL aggregation across RSU zones (quality-weighted)
  - Convergence check (0.5% threshold, McMahan et al. 2017)
  - ε policy envelope update per role tier — this is the cloud-level threshold
    adjustment described in the paper (Section IV-A): the envelope widens when
    global convergence is strong and tightens when zone anomaly rate is high.
"""

import numpy as np
from models import AttackDetector
import torch


class CloudDT:
    def __init__(self, model_template=None):
        self.global_model = AttackDetector() if model_template is None else model_template
        self.global_flat_params = self.global_model.get_flat_params().copy()
        self.loss_history = []
        self.round_counter = 0

        # ε policy envelope per role tier (cloud sets these boundaries)
        self.eps_envelope = {
            "infrastructure": [0.10, 1.50],
            "fleet":          [0.08, 1.20],
            "private":        [0.05, 1.00],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Global FL aggregation
    # ─────────────────────────────────────────────────────────────────────────

    def aggregate_zones(self, zone_updates):
        """
        Quality-weighted aggregation across RSU zones.
        Q_RSU = (1 - anomaly_rate) × (1 - probation_rate) × avg_trust
        """
        if not zone_updates:
            return self.global_flat_params

        weights = []
        grads = []

        for z in zone_updates:
            q = ((1.0 - z["anomaly_rate"]) *
                 (1.0 - z["probation_rate"]) *
                 z["avg_trust"])
            q = max(q, 0.01)
            weights.append(q)
            grads.append(z["gradient"])

        total_w = sum(weights) + 1e-9
        agg_grad = sum(w * g for w, g in zip(weights, grads)) / total_w

        # Apply gradient update to global model (SGD step)
        lr = 0.01
        self.global_flat_params = self.global_flat_params - lr * agg_grad
        self.global_model.set_flat_params(self.global_flat_params)
        self.round_counter += 1

        return self.global_flat_params.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Convergence check
    # ─────────────────────────────────────────────────────────────────────────

    def convergence_check(self, current_loss):
        """
        Returns True if model is stagnating (improvement < 0.5%).
        Li et al. (2020) threshold.
        """
        self.loss_history.append(float(current_loss))
        if len(self.loss_history) < 2:
            return False
        prev = self.loss_history[-2]
        improvement = (prev - current_loss) / (abs(prev) + 1e-9)
        return improvement < 0.005

    def compute_loss(self, X, y):
        """Compute cross-entropy loss on held-out data."""
        self.global_model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.long)
            out = self.global_model(Xt)
            loss = torch.nn.functional.cross_entropy(out, yt)
        return float(loss.item())

    # ─────────────────────────────────────────────────────────────────────────
    # ε policy envelope update
    # ─────────────────────────────────────────────────────────────────────────

    def update_eps_envelope(self, anomaly_rate_global, convergence_stagnating):
        """
        Widen envelope if convergence strong + low anomaly.
        Tighten if high anomaly rate or stagnating.
        """
        if anomaly_rate_global < 0.05 and not convergence_stagnating:
            factor = 1.05   # slight widening — reward clean rounds
        elif anomaly_rate_global > 0.20 or convergence_stagnating:
            factor = 0.92   # tighten — more noise for suspicious rounds
        else:
            factor = 1.0

        for role in self.eps_envelope:
            lo, hi = self.eps_envelope[role]
            new_lo = float(np.clip(lo * factor, 0.01, 0.50))
            new_hi = float(np.clip(hi * factor, 0.10, 2.00))
            self.eps_envelope[role] = [new_lo, new_hi]

        return self.eps_envelope

    def get_policy_for_role(self, role):
        return tuple(self.eps_envelope.get(role, [0.05, 1.00]))
