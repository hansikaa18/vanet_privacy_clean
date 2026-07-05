"""
rsu_dt.py
Implements the RSU-DT Layer:
  - Vehicle twin profiles (norm history, cosine history, budget log, trust score)
  - α computation: α = 0.5*trust + 0.3*stab + 0.2*budget
  - Predictive ε scheduler: εᵥ = εmin + α*(εmax - εmin)
  - Multi-signal anomaly detection (norm check, cosine check, tier tag check)
  - Trust score updates + RBAC consequence
  - Weighted FedAvg aggregation
"""

import numpy as np
from collections import defaultdict


class VehicleTwin:
    """Digital twin profile maintained per vehicle at the RSU."""

    def __init__(self, vid, role, trust_score, eps_total=1.0,
                 eps_min=0.05, eps_max=1.5, clip_C=1.0):
        self.vid = vid
        self.role = role
        self.trust_score = trust_score
        self.eps_total = eps_total
        self.eps_remaining = eps_total
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.clip_C = clip_C

        # Gradient history (core of the twin profile)
        self.norm_history = []
        self.cosine_history = []
        self.tier_tag_history = []
        self.budget_log = []
        self.gradient_history = []  # stores past gradient vectors for mean-direction check
        self.prev_gradient = None

        self.rounds_participated = 0
        self.consecutive_flags = 0
        self.role_rights = "TRAIN_ONLY"

        # Probation tracking
        self.is_probationary = trust_score < 0.40


class RSUDT:
    """
    RSU-level Digital Twin optimiser.
    One instance per RSU zone.
    """

    def __init__(self, eps_min=0.05, eps_max=1.50, clip_C=1.0,
                 norm_sigma_mult=3.0, min_history=5):
        """
        norm_sigma_mult=3.0 per paper Eq. (10): upper_bound = mu_norm + 3*sigma_norm.
        The direction (cosine) check has no fixed floor in the paper — it uses an
        adaptive per-vehicle baseline (hist_mean - 3*hist_std), computed fresh
        each round in screen_gradient() from the vehicle's own cosine_history.
        """
        self.twins = {}
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.clip_C = clip_C
        self.norm_sigma_mult = norm_sigma_mult
        self.min_history = min_history
        self.privacy_ledger = []
        self.round_counter = 0

    def register_vehicle(self, vid, role, trust_score, eps_total=1.0):
        self.twins[vid] = VehicleTwin(
            vid, role, trust_score,
            eps_total=eps_total,
            eps_min=self.eps_min,
            eps_max=self.eps_max,
            clip_C=self.clip_C,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # α computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_alpha(self, twin):
        """
        Three-signal composite α ∈ [0,1]:
          α = 0.5*α_trust + 0.3*α_stab + 0.2*α_budget
        """
        alpha_trust = float(np.clip(twin.trust_score, 0.0, 1.0))

        if len(twin.norm_history) >= 3:
            mu = np.mean(twin.norm_history)
            sigma = np.std(twin.norm_history)
            alpha_stab = float(1.0 - np.tanh(sigma / (mu + 1e-9)))
        else:
            # Conservative default for new vehicles
            alpha_stab = 0.40

        alpha_budget = float(np.clip(twin.eps_remaining / (twin.eps_total + 1e-9), 0.0, 1.0))

        alpha = 0.5 * alpha_trust + 0.3 * alpha_stab + 0.2 * alpha_budget
        return float(np.clip(alpha, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Budget reset (call once per round before scheduling)
    # ─────────────────────────────────────────────────────────────────────────

    def reset_round_budget(self, twin):
        """Reset twin ε budget for the new operational cycle (Dwork & Roth 2014)."""
        twin.eps_remaining = twin.eps_total

    # ─────────────────────────────────────────────────────────────────────────
    # ε scheduler
    # ─────────────────────────────────────────────────────────────────────────

    def schedule_eps(self, twin, eps_envelope=None):
        """
        Predictive ε scheduling:
          εᵥ = εmin + α * (εmax − εmin)
        Clamped to cloud envelope and remaining budget.
        """
        if eps_envelope:
            eps_min, eps_max = eps_envelope
        else:
            eps_min, eps_max = twin.eps_min, twin.eps_max

        alpha = self.compute_alpha(twin)
        eps_v = eps_min + alpha * (eps_max - eps_min)
        eps_v = float(np.clip(eps_v, eps_min, eps_max))
        eps_v = min(eps_v, twin.eps_remaining)
        eps_v = max(eps_v, 0.01)
        return eps_v, alpha

    # ─────────────────────────────────────────────────────────────────────────
    # Anomaly detection (post-screening)
    # ─────────────────────────────────────────────────────────────────────────

    def screen_gradient(self, twin, gradient, tier_tag, eps_consumed):
        """
        Four-signal post-reception anomaly detection (paper Section V-E):

          Signal 1 — Norm check (paper Eq. 10-11):
            upper_bound = μ_norm + 3σ_norm   (3σ per paper, not 2σ)
            FLAG norm_anomaly if ||g_dp|| > upper_bound
            Activated after MIN_TWIN_HISTORY rounds of accumulated history.
            Detects: amplification, ghost vehicle injection, random offset.

          Signal 2 — Directional cosine check (paper Eq. 12-13):
            cos_sim = dot(g_cur, g_prev) / (||g_cur|| * ||g_prev||)
            FLAG direction_anomaly if cos_sim < hist_mean - 3*hist_std
            Uses per-vehicle adaptive baseline from cosine_history —
            NO fixed global cosine floor (that was an undocumented deviation).
            Detects: constant position spoofing, slow-drift Byzantine attacks.

          Signal 3 — Sensitivity tier tag check (paper Eq. 14):
            FLAG tier_anomaly if |tier_tag - μ_tier| > 0.30
            Detects: noisy sensor manipulation via high-sensitivity components.

          Signal 4 — Budget consumption check (paper Eq. 15):
            FLAG budget_anomaly if |ε_consumed - μ_budget| / μ_budget > 0.50
            Detects: unauthorised extra local training steps.

        Returns list of flag strings (empty = clean).
        """
        flags = []
        norm = float(np.linalg.norm(gradient))

        # ── Signal 1: Norm check ──────────────────────────────────────────
        if len(twin.norm_history) >= self.min_history:
            mu = np.mean(twin.norm_history)
            sigma = np.std(twin.norm_history) + 1e-9
            upper = mu + self.norm_sigma_mult * sigma
            if norm > upper:
                flags.append("norm_anomaly")

        # ── Signal 2: Cosine direction check ─────────────────────────────
        # Compares the cosine similarity between consecutive rounds against
        # the vehicle's own historical baseline (mean ± k*std).
        # Catches sudden gradient reversals (Class 1) and slow drift (Class 5).
        if twin.prev_gradient is not None:
            prev_norm = np.linalg.norm(twin.prev_gradient) + 1e-9
            curr_norm = norm + 1e-9
            cos_sim   = float(np.dot(gradient, twin.prev_gradient) / (curr_norm * prev_norm))
            cos_sim   = float(np.clip(cos_sim, -1.0, 1.0))
            twin.cosine_history.append(cos_sim)

            # Only check after history is established
            if len(twin.cosine_history) >= self.min_history:
                hist      = twin.cosine_history[-self.min_history:]
                hist_mean = float(np.mean(hist))
                hist_std  = float(np.std(hist) + 1e-3)
                # Flag if current cosine is more than 3σ below the vehicle's norm
                if cos_sim < hist_mean - 3.0 * hist_std:
                    flags.append("direction_anomaly")

        # Store gradient in history (capped at last 20 rounds to save memory)
        twin.gradient_history.append(gradient.copy())
        if len(twin.gradient_history) > 20:
            twin.gradient_history.pop(0)

        # ── Signal 3: Tier tag spike ──────────────────────────────────────
        if len(twin.tier_tag_history) >= 3:
            expected = np.mean(twin.tier_tag_history)
            if abs(tier_tag - expected) > 0.30:
                flags.append("tier_anomaly")

        # ── Signal 4: Budget consumption anomaly ─────────────────────────
        # Only check after history is established, and compare against
        # the mean of past consumption (not the scheduled amount) to avoid
        # false alarms from normal round-to-round variation.
        if len(twin.budget_log) >= self.min_history:
            expected_eps = float(np.mean(twin.budget_log[-self.min_history:]))
            if expected_eps > 1e-6:
                deviation = abs(eps_consumed - expected_eps) / (expected_eps + 1e-9)
                if deviation > 0.50:   # >50% deviation = genuinely anomalous
                    flags.append("budget_anomaly")

        # ── Update twin profile ───────────────────────────────────────────
        # Drain twin's budget tracker to match actual consumption
        twin.eps_remaining = max(0.0, twin.eps_remaining - eps_consumed)
        twin.norm_history.append(norm)
        twin.tier_tag_history.append(tier_tag)
        twin.budget_log.append(eps_consumed)
        twin.prev_gradient = gradient.copy()
        twin.rounds_participated += 1

        return flags

    # ─────────────────────────────────────────────────────────────────────────
    # Trust score update + RBAC
    # ─────────────────────────────────────────────────────────────────────────

    def update_trust(self, twin, flags):
        """Update trust score and RBAC rights based on anomaly flags.

        Trust update follows the paper's bounded formula (Section V-G,
        Eq. 20-21), which is the operative rule — it supersedes the simpler
        uncapped penalty sketched in Section V-F's prose, since V-G's cap
        (0.08/round) and floor (0.10) are what make the model's reported
        attacker-trust trajectory in Table X (0.30 -> 0.15 -> ~0 over the
        full 20 rounds, not within a handful of flagged rounds) achievable:
          Clean round:   trust_score = min(1.0, trust_score + 0.05)
          Flagged round: trust_decrement = min(0.08, 0.04 * n_flags)
                         trust_score = max(0.10, trust_score - trust_decrement)

        RBAC thresholds follow paper Table I exactly:
          >=0.70            -> TRAIN+WEIGHTED_AGG (full rights)
          0.30 <= t < 0.70  -> TRAIN_ONLY (down-weighted)
          0.15 <= t < 0.30  -> FLAGGED (RSU logs)
          <0.15             -> SUSPENDED (re-auth required)
        Note the 0.10 floor in the decrement formula sits below the 0.15
        SUSPENDED boundary, so a vehicle can still reach SUSPENDED status
        through repeated flagged rounds even though trust itself never
        drops below 0.10.
        """
        if not flags:
            twin.trust_score = min(1.0, twin.trust_score + 0.05)
            twin.consecutive_flags = 0
        else:
            trust_decrement = min(0.08, 0.04 * len(flags))
            twin.trust_score = max(0.10, twin.trust_score - trust_decrement)
            twin.consecutive_flags += 1

        # RBAC consequence table (paper Table I)
        if twin.trust_score >= 0.70:
            twin.role_rights = "TRAIN+WEIGHTED_AGG"
            twin.is_probationary = False
        elif twin.trust_score >= 0.30:
            twin.role_rights = "TRAIN_ONLY"
            twin.is_probationary = False
        elif twin.trust_score >= 0.15:
            twin.role_rights = "FLAGGED"
            twin.is_probationary = True
        else:
            twin.role_rights = "SUSPENDED"
            twin.is_probationary = True

    # ─────────────────────────────────────────────────────────────────────────
    # Weighted FedAvg aggregation
    # ─────────────────────────────────────────────────────────────────────────

    def aggregate(self, updates):
        """
        Weighted FedAvg (paper Section V-G, Eq. 18-19):
          weight_i = trust_score_i x |D_i| x penalty_i
          penalty_i: 1.0 (clean), 0.30 (one flag), 0.10 (2+ flags)

        SUSPENDED vehicles (trust < 0.15) are excluded entirely per Table I
        ("Action: Re-auth required"), rather than merely down-weighted.

        Returns aggregated gradient vector.
        """
        weights = []
        grads = []

        for u in updates:
            twin = self.twins[u["vid"]]

            # SUSPENDED vehicles are excluded from aggregation (Table I)
            if twin.role_rights == "SUSPENDED":
                continue

            n_flags = len(u.get("flags", []))
            if n_flags == 0:
                penalty = 1.0
            elif n_flags == 1:
                penalty = 0.30
            else:
                penalty = 0.10

            w = twin.trust_score * u["n_samples"] * penalty
            weights.append(max(w, 0.0))
            grads.append(u["gradient"])

        if not grads:
            # Nothing eligible this round (e.g. all suspended) — no update
            return np.zeros_like(updates[0]["gradient"]) if updates else None

        total_w = sum(weights) + 1e-9
        agg = sum(w * g for w, g in zip(weights, grads)) / total_w

        # Log to privacy ledger
        flagged_count = sum(1 for u in updates if u.get("flags"))
        avg_trust = np.mean([self.twins[u["vid"]].trust_score for u in updates])

        self.privacy_ledger.append({
            "round": self.round_counter,
            "n_vehicles": len(updates),
            "flagged": flagged_count,
            "avg_trust": avg_trust,
        })
        self.round_counter += 1

        return agg

    def get_zone_metadata(self, updates):
        """Compute zone-level metadata for cloud aggregation."""
        n = len(updates)
        if n == 0:
            return {"anomaly_rate": 0, "probation_rate": 0, "avg_trust": 0.5}

        anomaly_rate = sum(1 for u in updates if u.get("flags")) / n
        probation_rate = sum(1 for u in updates if u.get("probationary", False)) / n
        avg_trust = np.mean([self.twins[u["vid"]].trust_score for u in updates])

        return {
            "anomaly_rate": float(anomaly_rate),
            "probation_rate": float(probation_rate),
            "avg_trust": float(avg_trust),
        }
