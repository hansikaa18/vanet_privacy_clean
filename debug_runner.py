"""
debug_runner.py  — run this to find exactly why vehicles are skipped
Usage:  python debug_runner.py
"""
import sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import numpy as np
import torch

from config import *
from preprocessing import load_dataset, engineer_features, prepare_vehicle_datasets, FEATURE_COLS
from vehicle import Vehicle
from rsu_dt import RSUDT
from cloud import CloudDT
from sklearn.model_selection import train_test_split as tts

np.random.seed(SEED)
torch.manual_seed(SEED)

print("Loading data...")
df = load_dataset(DATASET_NAME, DATA_PATH, max_vehicles=10, verbose=False)
df = engineer_features(df, verbose=False)
train_df, _ = tts(df, test_size=0.20, random_state=SEED, stratify=df["label"])
vehicle_data, scaler = prepare_vehicle_datasets(train_df.copy())
all_vids = list(vehicle_data.keys())[:5]

print(f"Using {len(all_vids)} vehicles for debug: {all_vids}")

rsu   = RSUDT(eps_min=EPS_MIN, eps_max=EPS_MAX, clip_C=CLIP_C, min_history=MIN_TWIN_HISTORY)
cloud = CloudDT()

vehicles = {}
for vid in all_vids:
    v = Vehicle(vid, role='private', eps_total=EPS_TOTAL, is_attacker=False, attack_type=0)
    vehicles[vid] = v
    rsu.register_vehicle(vid, 'private', v.trust_score, eps_total=EPS_TOTAL)

print()
print("=" * 60)
print("  Tracing Round 1 for each vehicle")
print("=" * 60)

for vid in all_vids:
    vdata = vehicle_data[vid]
    X_full, y_full = vdata["X"], vdata["y"]
    v    = vehicles[vid]
    twin = rsu.twins[vid]

    print(f"\nVehicle {vid}:")
    print(f"  v.trust_score     = {v.trust_score}")
    print(f"  v.eps_total       = {v.eps_total}")
    print(f"  v.eps_remaining   = {v.eps_remaining}")
    print(f"  twin.eps_remaining= {twin.eps_remaining}")
    print(f"  twin.eps_total    = {twin.eps_total}")

    # Step 1: reset
    rsu.reset_round_budget(twin)
    print(f"  after reset_round_budget: twin.eps_remaining = {twin.eps_remaining}")

    # Step 2: schedule
    envelope = cloud.get_policy_for_role(v.role)
    eps_v, alpha = rsu.schedule_eps(twin, eps_envelope=envelope)
    print(f"  envelope = {envelope}")
    print(f"  eps_v scheduled = {eps_v:.4f}  alpha = {alpha:.4f}")

    # Step 3: rbac check
    result = v.rbac_eps_check(eps_v, CLIP_C)
    print(f"  rbac_eps_check returned: {result}")
    print(f"  v.eps_remaining after check: {v.eps_remaining:.4f}")
    print(f"  v.eps_allocated: {v.eps_allocated:.4f}")

    if not result:
        print(f"  *** SKIPPED — reason:")
        if v.trust_score < 0.10:
            print(f"      trust_score {v.trust_score} < 0.10")
        else:
            print(f"      unknown — check rbac_eps_check logic")
        continue

    # Step 4: sample data
    n_sample = min(BEACONS_PER_ROUND, len(X_full))
    idx = np.random.choice(len(X_full), size=n_sample, replace=True)
    X_s, y_s = X_full[idx], y_full[idx]
    print(f"  X_s.shape={X_s.shape}  unique labels={np.unique(y_s).tolist()}")

    if len(np.unique(y_s)) < 2:
        print(f"  *** SKIPPED — only one class in sample")
        continue

    # Step 5: train
    update = v.local_train(X_s, y_s, cloud.global_flat_params, epochs=LOCAL_EPOCHS, lr=LR)
    grad_norm = float(np.linalg.norm(update['gradient']))
    print(f"  gradient norm = {grad_norm:.4f}")

    # Step 6: screen
    flags = rsu.screen_gradient(twin, update['gradient'], update['tier_tag'], update['eps_consumed'])
    print(f"  flags = {flags}")
    print(f"  *** SUCCESS — vehicle participated and was screened")

print()
print("=" * 60)
print("  vehicle.py rbac_eps_check source (lines 45-60):")
print("=" * 60)
with open(os.path.join(BASE, 'vehicle.py'), encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines[44:65], 45):
    print(f"  {i:3d}: {line}", end='')

print()
print("=" * 60)
print("  federated_runner.py lines around reset_round_budget:")
print("=" * 60)
with open(os.path.join(BASE, 'federated_runner.py'), encoding='utf-8') as f:
    flines = f.readlines()
for i, line in enumerate(flines):
    if any(x in line for x in ['reset_round_budget', 'rbac_eps_check', 'vehicle_anomaly_log.append', 'your_updates.append']):
        print(f"  {i+1:4d}: {line}", end='')
