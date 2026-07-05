"""
plot_results.py
Generates charts from saved results. Run AFTER federated_runner.py.

Usage:
    python plot_results.py

Charts produced:
    accuracy_over_rounds.png   — test accuracy per FL round
    f1_over_rounds.png         — F1 score per FL round
    privacy_utility_frontier.png — accuracy vs REAL cumulative ε consumed
    trust_and_flags.png        — flagged count + avg trust per round
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS_DIR, DATASET_NAME

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
MODEL_COLOR  = "#1D9E75"   # teal — DT-RBAC-FL-ADP
FLAG_COLOR   = "#D85A30"   # coral — flagged vehicles bar
TRUST_COLOR  = "#7F77DD"   # purple — trust score line

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         150,
})

# ── Load data ─────────────────────────────────────────────────────────────────
stats_path = os.path.join(RESULTS_DIR, "round_stats.json")
final_path = os.path.join(RESULTS_DIR, "final_results.json")

if not os.path.exists(stats_path):
    raise FileNotFoundError(
        f"{stats_path} not found.\n"
        "Run  python federated_runner.py  first.")

with open(stats_path) as f:
    round_stats = json.load(f)

with open(final_path) as f:
    final_results = json.load(f)

rounds   = [r["round"]         for r in round_stats]
accuracy = [r["your_accuracy"] for r in round_stats]
f1       = [r["your_f1"]       for r in round_stats]
flagged  = [r["flagged"]       for r in round_stats]
trusts   = [r["avg_trust"]     for r in round_stats]

# Real cumulative ε — tracked in federated_runner.py, no synthetic placeholders
cum_eps  = [r["cumulative_eps"] for r in round_stats]


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Accuracy over FL training rounds
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(rounds, accuracy,
        color=MODEL_COLOR, linewidth=2.2, marker="o", markersize=4,
        label="DT-RBAC-FL-ADP (yours)")
ax.set_xlabel("FL training round")
ax.set_ylabel("Test accuracy")
ax.set_title(f"Model accuracy over FL training rounds\n({DATASET_NAME})")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_xlim(1, max(rounds))
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "accuracy_over_rounds.png"))
plt.close(fig)
print("Saved: accuracy_over_rounds.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: F1 score over FL training rounds
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(rounds, f1,
        color=MODEL_COLOR, linewidth=2.2, marker="o", markersize=4,
        label="DT-RBAC-FL-ADP (yours)")
ax.set_xlabel("FL training round")
ax.set_ylabel("F1 score (attack class)")
ax.set_title(f"Attack detection F1 over FL training rounds\n({DATASET_NAME})")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_xlim(1, max(rounds))
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "f1_over_rounds.png"))
plt.close(fig)
print("Saved: f1_over_rounds.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Privacy-utility frontier — REAL cumulative ε vs accuracy
# ─────────────────────────────────────────────────────────────────────────────
# cumulative_eps and your_accuracy are now tracked round-by-round in
# federated_runner.py (see round_stats["cumulative_eps"]).
# This chart uses those REAL measured values — no synthetic np.linspace curves.
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cum_eps, accuracy,
        color=MODEL_COLOR, linewidth=2.2, marker="o", markersize=3,
        label="DT-RBAC-FL-ADP (yours)")

ax.set_xlabel("Cumulative privacy budget consumed (ε)")
ax.set_ylabel("Test accuracy")
ax.set_title(f"Privacy-utility frontier\n(higher & left = better) — {DATASET_NAME}")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "privacy_utility_frontier.png"))
plt.close(fig)
print("Saved: privacy_utility_frontier.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: Flagged vehicles + avg trust over rounds
# ─────────────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()

ax1.bar(rounds, flagged, color=FLAG_COLOR, alpha=0.55, label="Flagged vehicles")
ax2.plot(rounds, trusts, color=TRUST_COLOR, linewidth=2.2, marker="o",
         markersize=3, label="Avg trust score")

ax1.set_xlabel("FL training round")
ax1.set_ylabel("Flagged vehicles (count)", color=FLAG_COLOR)
ax2.set_ylabel("Average trust score",      color=TRUST_COLOR)
ax1.set_title(f"DT-RBAC-FL-ADP — flagged vehicles and trust evolution\n({DATASET_NAME})")
ax1.tick_params(axis="y", colors=FLAG_COLOR)
ax2.tick_params(axis="y", colors=TRUST_COLOR)
ax2.set_ylim(0, 1.1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "trust_and_flags.png"))
plt.close(fig)
print("Saved: trust_and_flags.png")


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
header = f"{'Model':<25} {'Accuracy':>9} {'F1':>8} {'Precision':>10} {'Recall':>8}"
print(header)
print("-" * 60)
for name, res in final_results.items():
    row = (f"{name:<25} "
           f"{res['accuracy']:>9.4f} "
           f"{res['f1']:>8.4f} "
           f"{res['precision']:>10.4f} "
           f"{res['recall']:>8.4f}")
    print(row)
print("=" * 60)
print(f"\nAll charts saved to: {RESULTS_DIR}/\n")
