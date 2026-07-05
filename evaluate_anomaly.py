"""
evaluate_anomaly.py
Measures anomaly detection accuracy using VeReMi class labels as ground truth.

Ground truth  (from VeReMi 'class' column):
    attack_type = 0      → legitimate vehicle   → should NOT be flagged
    attack_type = 1–5    → attacker vehicle      → SHOULD be flagged

Anomaly detector output (from RSU-DT screen_gradient):
    flagged = True       → detector raised an alarm
    flagged = False      → detector passed it as clean

Run AFTER federated_runner.py:
    python evaluate_anomaly.py
"""

import json
import os
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)
from config import RESULTS_DIR


def main():
    log_path = os.path.join(RESULTS_DIR, "vehicle_anomaly_log.json")
    if not os.path.exists(log_path):
        print(f"ERROR: {log_path} not found.")
        print("Please run  python federated_runner.py  first.")
        return

    with open(log_path) as f:
        log = json.load(f)

    if not log:
        print("ERROR: vehicle_anomaly_log.json is empty.")
        return

    # ── Build ground truth and predictions ───────────────────────────────
    # Each entry = one vehicle × one FL round
    y_true = []   # 1 = attacker (VeReMi class 1-5),  0 = legit (class 0)
    y_pred = []   # 1 = flagged by anomaly detector,   0 = passed clean
    rounds = []
    attack_types = []

    for entry in log:
        y_true.append(1 if entry["attack_type"] > 0 else 0)
        y_pred.append(1 if entry["flagged"] else 0)
        rounds.append(entry["round"])
        attack_types.append(entry["attack_type"])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Overall results ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Anomaly Detection — Ground Truth Evaluation")
    print("  (VeReMi class label vs RSU-DT flags)")
    print("=" * 60)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print(f"\n  Confusion matrix")
    print(f"  {'':25s}  Predicted clean  Predicted flagged")
    print(f"  {'Actual legit  (class=0)':25s}  {tn:14d}   {fp:17d}  ← false alarms")
    print(f"  {'Actual attacker (class>0)':25s}  {fn:14d}   {tp:17d}  ← correctly caught")

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)

    print(f"\n  Overall metrics (anomaly detector only)")
    print(f"  Accuracy :  {acc:.4f}  — how often detector decision matches VeReMi label")
    print(f"  Precision:  {prec:.4f}  — of flagged vehicles, how many were real attackers")
    print(f"  Recall   :  {rec:.4f}  — of real attackers, how many were caught")
    print(f"  F1 Score :  {f1:.4f}  — balance of precision and recall")

    print(f"\n  Full classification report")
    print(classification_report(y_true, y_pred,
                                 target_names=["Legitimate", "Attacker"],
                                 zero_division=0))

    # ── Per attack class breakdown ────────────────────────────────────────
    print("  Detection rate per VeReMi attack class")
    print(f"  {'Class':<10} {'Label':<30} {'Detected':<10} {'Total':<10} {'Recall'}")
    print(f"  {'-'*10} {'-'*30} {'-'*10} {'-'*10} {'-'*8}")

    class_names = {
        0: "Legitimate (no attack)",
        1: "Constant position",
        2: "Constant offset",
        3: "Random position (ghost)",
        4: "Random offset (noisy)",
        5: "Eventual stop (slow drift)",
    }

    attack_types_arr = np.array(attack_types)
    for cls in sorted(set(attack_types)):
        mask     = attack_types_arr == cls
        total    = mask.sum()
        detected = (y_pred[mask] == 1).sum() if cls > 0 else (y_pred[mask] == 0).sum()
        recall   = detected / total if total > 0 else 0.0
        label    = "correctly clean" if cls == 0 else "flagged"
        print(f"  {cls:<10} {class_names.get(cls,'Unknown'):<30} "
              f"{detected:<10} {total:<10} {recall:.3f}  ({label})")

    # ── Per round trend ───────────────────────────────────────────────────
    print(f"\n  Recall trend across FL rounds (does it improve?)")
    rounds_arr = np.array(rounds)
    unique_rounds = sorted(set(rounds))
    # Group into early / mid / late thirds
    n = len(unique_rounds)
    thirds = [unique_rounds[:n//3], unique_rounds[n//3:2*n//3], unique_rounds[2*n//3:]]
    labels_thirds = ["Early rounds", "Mid rounds  ", "Late rounds "]

    for period_label, period_rounds in zip(labels_thirds, thirds):
        if not period_rounds:
            continue
        mask      = np.isin(rounds_arr, period_rounds)
        yt        = y_true[mask]
        yp        = y_pred[mask]
        attk_mask = yt == 1
        if attk_mask.sum() == 0:
            r = 0.0
        else:
            r = (yp[attk_mask] == 1).sum() / attk_mask.sum()
        fp_rate = (yp[yt == 0] == 1).sum() / max((yt == 0).sum(), 1)
        print(f"  {period_label}  recall={r:.3f}   false-alarm rate={fp_rate:.3f}")

    print()
    print("  Interpretation guide")
    print("  Recall    > 0.70 = good  — catching most attackers")
    print("  Precision > 0.70 = good  — not crying wolf on legit vehicles")
    print("  Recall improves across rounds = expected (twin history builds up)")
    print("  Class 5 recall often lags — slow drift needs more rounds to detect")
    print()


if __name__ == "__main__":
    main()
