"""
preprocessing.py
Loads one of 3 supported VANET datasets and engineers the 4 features used
by DT-RBAC-FL-ADP. All loaders normalise to the same internal schema:

  vehicle_id   str    — unique vehicle / node identifier
  timestamp    float  — seconds since simulation start
  x            float  — position x (metres or degrees longitude)
  y            float  — position y (metres or degrees latitude)
  speed        float  — scalar speed (m/s)
  heading      float  — heading angle in degrees
  attack_type  int    — 0 = legitimate; >0 = attacker (dataset-specific codes)
  label        int    — 0 = legitimate, 1 = attack (binary, for training)

Supported datasets (set DATASET_NAME in config.py):
  "veremi_extension"    VeReMi Extension dataset (Kamel et al. 2020)
                        CSV or JSON, file or folder
  "kaggle_maliciousnode" VANET-MaliciousNode Dataset (Kaggle, ziya07)
                        Single CSV (~5 k rows, 20 features)
  "mosaic_replay_bogus" MOSAIC Replay/Bogus dataset (Iqbal et al. 2022)
                        CSV file or folder from the Google Drive release
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["pos_zscore", "speed_anomaly", "heading_dev", "time_delta"]

# Binary label map: 0 → legitimate, anything else → attacker
# Works for VeReMi (0–5), Kaggle (0/1), and MOSAIC (0/1).
LABEL_MAP = {0: 0}   # default: unknown codes map to 1 (attacker) via the lambda below


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers shared across loaders
# ─────────────────────────────────────────────────────────────────────────────

def _collect_files(data_path, ext):
    """Return sorted list of files with given extension under data_path."""
    if os.path.isfile(data_path) and data_path.lower().endswith(f".{ext}"):
        return [data_path]
    return sorted(set(
        glob.glob(os.path.join(data_path, "**", f"*.{ext}"), recursive=True) +
        glob.glob(os.path.join(data_path, f"*.{ext}"))))


def _to_binary_label(attack_type_series):
    """Map any non-zero attack_type to label=1 (attacker)."""
    return (attack_type_series != 0).astype(int)


def _ensure_schema(df):
    """Guarantee all required columns exist with correct dtypes."""
    required = {"vehicle_id": "", "timestamp": 0.0, "x": 0.0,
                "y": 0.0, "speed": 0.0, "heading": 0.0, "attack_type": 0}
    for col, default in required.items():
        if col not in df.columns:
            df[col] = default
        else:
            if col == "vehicle_id":
                df[col] = df[col].astype(str).str.strip()
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df.reset_index(drop=True)


def _enforce_vehicle_diversity(df, verbose):
    """
    If only 1–2 unique vehicle IDs exist (common when a pseudonym column was
    picked instead of a real ID), fall back to row-chunking so the FL round
    has multiple participants.
    """
    n_vids = df["vehicle_id"].nunique()
    if n_vids <= 2:
        chunk = 500
        if verbose:
            print(f"[Preprocessing] Only {n_vids} vehicle ID(s) found — "
                  f"using row-chunking ({chunk} rows/vehicle)")
        df["vehicle_id"] = (np.arange(len(df)) // chunk).astype(str)
    return df


def _cap_vehicles_and_rows(df, max_vehicles, rows_per_vehicle=2000):
    """Keep up to max_vehicles vehicles, capped to rows_per_vehicle rows each."""
    all_vids = df["vehicle_id"].unique()
    if len(all_vids) > max_vehicles:
        df = df[df["vehicle_id"].isin(all_vids[:max_vehicles])].copy()
    parts = [grp.head(rows_per_vehicle) for _, grp in df.groupby("vehicle_id")]
    return pd.concat(parts).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Loader 1 — VeReMi Extension
# ─────────────────────────────────────────────────────────────────────────────

def _load_veremi_csv(data_path, max_vehicles, verbose):
    files = _collect_files(data_path, "csv")
    if not files:
        raise FileNotFoundError(f"No CSV files found at: {data_path}")
    if verbose:
        print(f"[VeReMi] Found {len(files)} CSV file(s)")

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"[VeReMi] Warning: skipping {f}: {e}")
    if not dfs:
        raise ValueError("No VeReMi CSV files could be read.")

    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    cols = set(df.columns)

    if verbose:
        print(f"[VeReMi] Raw columns: {list(df.columns)}")
        print(f"[VeReMi] Raw shape:   {df.shape}")

    # vehicle_id
    vid_col = next((c for c in ["sender", "senderid", "senderpseudo",
                                 "vehicle_id", "id", "node_id"] if c in cols), None)
    if vid_col:
        df["vehicle_id"] = df[vid_col].astype(str)
    else:
        # No vehicle ID column — bucket by row index into groups of 300
        # Each group acts as one vehicle's beacon sequence for FL training
        df["vehicle_id"] = (np.arange(len(df)) // 300).astype(str)
        # Cap to MAX_VEHICLES worth of groups to avoid memory issues on large files
        df = df[df["vehicle_id"].astype(int) < 200].copy()

    # attack_type — prefer attack_type over 'class'; ignore 'type' (BSM msg type)
    # Handles both integer codes AND string labels (e.g. "RandomSpeedOffset")
    atk_col = next((c for c in ["attack_type", "attacktype", "class",
                                  "misbehaviortype", "attack"] if c in cols), None)
    if atk_col:
        raw = df[atk_col]
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.isna().mean() > 0.5:
            # Majority are strings — map non-zero/non-"none"/"normal" strings to 1
            _legit = {"0", "none", "normal", "legitimate", "legit", "", "nan", "0.0"}
            df["attack_type"] = raw.astype(str).str.strip().str.lower().apply(
                lambda v: 0 if v in _legit else 1)
        else:
            df["attack_type"] = numeric.fillna(0).astype(int)
    else:
        df["attack_type"] = 0

    # timestamp
    ts_col = next((c for c in ["sendtime", "rcvtime", "timestamp", "time"] if c in cols), None)
    df["timestamp"] = pd.to_numeric(
        df[ts_col] if ts_col else pd.Series(np.arange(len(df))),
        errors="coerce").fillna(0)

    # position x — includes pos_0 style (original VeReMi)
    for cand in ["posx", "pos_x", "pos_0", "x", "position_x", "sendpos_x"]:
        if cand in cols:
            df["x"] = pd.to_numeric(df[cand], errors="coerce").fillna(0)
            break
    if "x" not in df.columns:
        df["x"] = 0.0

    # position y
    for cand in ["posy", "pos_y", "pos_1", "y", "position_y", "sendpos_y"]:
        if cand in cols:
            df["y"] = pd.to_numeric(df[cand], errors="coerce").fillna(0)
            break
    if "y" not in df.columns:
        df["y"] = 0.0

    # speed — spd_0/spd_1 (original VeReMi), spdx/spdy, or scalar
    if "spd_0" in cols and "spd_1" in cols:
        df["speed"] = np.sqrt(
            pd.to_numeric(df["spd_0"], errors="coerce").fillna(0) ** 2 +
            pd.to_numeric(df["spd_1"], errors="coerce").fillna(0) ** 2)
    elif "spdx" in cols and "spdy" in cols:
        df["speed"] = np.sqrt(
            pd.to_numeric(df["spdx"], errors="coerce").fillna(0) ** 2 +
            pd.to_numeric(df["spdy"], errors="coerce").fillna(0) ** 2)
    else:
        spd_col = next((c for c in ["spd", "speed", "vel", "velocity"] if c in cols), None)
        df["speed"] = pd.to_numeric(df[spd_col], errors="coerce").fillna(0).abs() if spd_col else 0.0

    # heading — hed_0/hed_1 (original VeReMi), hedx/hedy, or scalar
    if "hed_0" in cols and "hed_1" in cols:
        df["heading"] = np.degrees(np.arctan2(
            pd.to_numeric(df["hed_1"], errors="coerce").fillna(0),
            pd.to_numeric(df["hed_0"], errors="coerce").fillna(0)))
    elif "hedx" in cols and "hedy" in cols:
        df["heading"] = np.degrees(np.arctan2(
            pd.to_numeric(df["hedy"], errors="coerce").fillna(0),
            pd.to_numeric(df["hedx"], errors="coerce").fillna(0)))
    else:
        hed_col = next((c for c in ["heading", "angle", "dir", "yaw"] if c in cols), None)
        df["heading"] = pd.to_numeric(df[hed_col], errors="coerce").fillna(0) if hed_col else 0.0

    df = _ensure_schema(df)
    df = _enforce_vehicle_diversity(df, verbose)

    if len(df["vehicle_id"].unique()) <= 2 and "senderpseudo" in cols:
        df["vehicle_id"] = df["senderpseudo"].astype(str)
        if verbose:
            print(f"[VeReMi] Retrying with 'senderpseudo' → "
                  f"{df['vehicle_id'].nunique()} vehicles")
        df = _enforce_vehicle_diversity(df, verbose)

    df = _cap_vehicles_and_rows(df, max_vehicles)

    if verbose:
        print(f"[VeReMi] Final: {len(df):,} rows, "
              f"{df['vehicle_id'].nunique()} vehicles")
        vc = df.drop_duplicates("vehicle_id").groupby("attack_type")["vehicle_id"].count()
        print(f"[VeReMi] Attack type distribution: {vc.to_dict()}")

    return df[["vehicle_id", "timestamp", "x", "y", "speed", "heading", "attack_type"]]


def _parse_veremi_json_file(filepath):
    """Parse a single VeReMi JSON trace file."""
    records = []
    try:
        with open(filepath, "r") as f:
            data = json.loads(f.read().strip())
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            vid   = str(item.get("id", item.get("senderId", "unknown")))
            atype = int(item.get("type", item.get("attackType", 0)))
            for msg in item.get("messages", item.get("traces", [])):
                if not isinstance(msg, dict):
                    continue
                pos = msg.get("pos", [0, 0, 0])
                if isinstance(pos, dict):
                    pos = [pos.get("x", 0), pos.get("y", 0), 0]
                spd = msg.get("spd", msg.get("speed", 0))
                if isinstance(spd, list):
                    spd = float(np.linalg.norm(spd[:2]))
                records.append({
                    "vehicle_id":  vid,
                    "attack_type": atype,
                    "timestamp":   float(msg.get("rcvTime", msg.get("time", 0))),
                    "x":           float(pos[0]) if len(pos) > 0 else 0.0,
                    "y":           float(pos[1]) if len(pos) > 1 else 0.0,
                    "speed":       float(spd),
                    "heading":     float(msg.get("heading", 0)),
                })
    except Exception:
        pass
    return records


def _load_veremi_json(data_path, max_vehicles, verbose):
    files = _collect_files(data_path, "json")
    if verbose:
        print(f"[VeReMi] Found {len(files)} JSON file(s)")
    records, seen = [], set()
    for f in files:
        recs = _parse_veremi_json_file(f)
        for r in recs:
            seen.add(r["vehicle_id"])
        records.extend(recs)
        if len(seen) >= max_vehicles:
            break
    df = pd.DataFrame(records).dropna()
    df = df[df["vehicle_id"].isin(list(seen)[:max_vehicles])]
    if verbose:
        print(f"[VeReMi] Loaded {len(df):,} beacons, {df['vehicle_id'].nunique()} vehicles")
    return df[["vehicle_id", "timestamp", "x", "y", "speed", "heading", "attack_type"]]


def load_veremi_extension(data_path, max_vehicles=200, verbose=True):
    """
    Load VeReMi Extension dataset from a CSV file, CSV folder, or JSON folder.
    Download: https://veremi-dataset.github.io/
    """
    # Detect format
    if os.path.isfile(data_path) and data_path.lower().endswith(".csv"):
        fmt = "csv"
    elif _collect_files(data_path, "csv"):
        fmt = "csv"
    elif _collect_files(data_path, "json"):
        fmt = "json"
    else:
        raise FileNotFoundError(
            f"No CSV or JSON files found at: {data_path}\n"
            "Check DATA_PATH in config.py.")

    if verbose:
        print(f"[VeReMi] Format: {fmt.upper()}")

    df = (_load_veremi_csv(data_path, max_vehicles, verbose) if fmt == "csv"
          else _load_veremi_json(data_path, max_vehicles, verbose))
    return _ensure_schema(df)


# ─────────────────────────────────────────────────────────────────────────────
# Loader 2 — Kaggle VANET-MaliciousNode Dataset
# ─────────────────────────────────────────────────────────────────────────────

# Known column aliases for this dataset (lowercase). Add more if the CSV
# header differs in future versions of the Kaggle upload.
_KAGGLE_ALIASES = {
    "vehicle_id":   ["node_id", "vehicle_id", "id", "vid"],
    "timestamp":    ["timestamp", "time", "simtime", "sendtime"],
    "x":            ["position_x", "longitude", "lon", "pos_x", "x"],
    "y":            ["position_y", "latitude",  "lat", "pos_y", "y"],
    "speed":        ["speed", "velocity", "vel", "spd"],
    "heading":      ["direction", "heading", "angle", "yaw", "bearing"],
    "attack_type":  ["is_malicious", "attack_type", "attacktype", "attack",
                     "label", "malicious"],
}

# Extra numeric feature columns present in Kaggle dataset used as
# supplementary signals when spatial beacon history is unavailable
_KAGGLE_EXTRA_COLS = [
    "trust_score", "false_packet_injection", "blackhole_attack_attempts",
    "sybil_attack_attempts", "denial_of_service", "packet_drop_ratio",
    "signal_strength", "neighbor_trust_score_avg", "historical_trust_score",
]


def load_kaggle_maliciousnode(data_path, max_vehicles=200, verbose=True):
    """
    Load the Kaggle VANET-MaliciousNode Dataset.

    Download:
        https://www.kaggle.com/datasets/ziya07/vanet-maliciousnode-dataset
        (free Kaggle account required — click Download to get a single CSV)

    Column mapping used by this loader (case-insensitive):
        node_id       → vehicle_id
        timestamp     → timestamp
        longitude     → x   (or pos_x / x depending on version)
        latitude      → y   (or pos_y / y depending on version)
        speed         → speed
        heading       → heading
        attack_type   → attack_type  (0 = legitimate, 1+ = attacker)
        label         → used as attack_type if attack_type col absent

    The dataset is a single flat CSV with ~5,000 rows and 20 features.
    It does NOT have repeated beacons per vehicle in the VeReMi sense —
    each row is one network event for one node. The loader groups rows by
    node_id to produce per-vehicle time series for FL training.

    Attack types in this dataset:
        0 = Benign
        1 = False Injection (bogus speed/position data)
        2 = Sybil (multiple fake identities)
        3 = Blackhole (drops/modifies packets)
        4 = DoS (flooding)
    These differ from VeReMi's 5 classes but your anomaly detector's
    4 signals (norm, cosine, tier, budget) are attack-type agnostic —
    they detect behavioural deviation regardless of the class label.
    """
    files = _collect_files(data_path, "csv")
    if not files:
        raise FileNotFoundError(
            f"No CSV found at: {data_path}\n"
            "Download from https://www.kaggle.com/datasets/ziya07/vanet-maliciousnode-dataset\n"
            "then set DATA_PATH in config.py to the downloaded .csv file.")

    if verbose:
        print(f"[Kaggle] Reading {files[0]}")

    df = pd.read_csv(files[0], low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    cols = set(df.columns)

    if verbose:
        print(f"[Kaggle] Raw columns: {list(df.columns)}")
        print(f"[Kaggle] Raw shape:   {df.shape}")

    # Map columns using alias table
    mapped = {}
    for target, candidates in _KAGGLE_ALIASES.items():
        found = next((c for c in candidates if c in cols), None)
        mapped[target] = found

    if verbose:
        print(f"[Kaggle] Column mapping: {mapped}")

    # vehicle_id
    df["vehicle_id"] = (df[mapped["vehicle_id"]].astype(str)
                        if mapped["vehicle_id"] else
                        (np.arange(len(df)) // 50).astype(str))

    # timestamp — synthesise sequential if missing
    df["timestamp"] = (pd.to_numeric(df[mapped["timestamp"]], errors="coerce").fillna(0)
                       if mapped["timestamp"] else
                       np.arange(len(df), dtype=float))

    # x / y  (longitude / latitude or x / y)
    df["x"] = (pd.to_numeric(df[mapped["x"]], errors="coerce").fillna(0)
               if mapped["x"] else 0.0)
    df["y"] = (pd.to_numeric(df[mapped["y"]], errors="coerce").fillna(0)
               if mapped["y"] else 0.0)

    # speed
    df["speed"] = (pd.to_numeric(df[mapped["speed"]], errors="coerce").fillna(0).abs()
                   if mapped["speed"] else 0.0)

    # heading
    df["heading"] = (pd.to_numeric(df[mapped["heading"]], errors="coerce").fillna(0)
                     if mapped["heading"] else 0.0)

    # attack_type — use is_malicious directly (binary 0/1 label column)
    atk_col = mapped["attack_type"]
    if atk_col:
        df["attack_type"] = pd.to_numeric(df[atk_col], errors="coerce").fillna(0).astype(int)
    else:
        df["attack_type"] = 0

    df = _ensure_schema(df)

    # This dataset has 1 row per vehicle (snapshot, not beacon log).
    # Synthesise a short time series per vehicle by duplicating each row
    # N times with small Gaussian noise on position/speed so that
    # engineer_features() can compute meaningful per-vehicle statistics.
    SYNTH_ROWS = 20
    rng = np.random.default_rng(42)
    parts = []
    for _, row in df.iterrows():
        rows = pd.DataFrame([row] * SYNTH_ROWS).reset_index(drop=True)
        rows["x"]         += rng.normal(0, max(abs(row["x"]) * 0.001, 0.1), SYNTH_ROWS)
        rows["y"]         += rng.normal(0, max(abs(row["y"]) * 0.001, 0.1), SYNTH_ROWS)
        rows["speed"]     += rng.normal(0, max(abs(row["speed"]) * 0.05, 0.01), SYNTH_ROWS)
        rows["speed"]      = rows["speed"].abs()
        rows["heading"]   += rng.normal(0, 2.0, SYNTH_ROWS)
        rows["timestamp"]  = np.arange(SYNTH_ROWS, dtype=float)
        parts.append(rows)
    df = pd.concat(parts, ignore_index=True)

    df = _enforce_vehicle_diversity(df, verbose)
    df = _cap_vehicles_and_rows(df, max_vehicles, rows_per_vehicle=SYNTH_ROWS)

    if verbose:
        print(f"[Kaggle] Final: {len(df):,} rows (synthesised), "
              f"{df['vehicle_id'].nunique()} vehicles")
        vc = df.drop_duplicates("vehicle_id").groupby("attack_type")["vehicle_id"].count()
        print(f"[Kaggle] Attack type distribution: {vc.to_dict()}")

    return df[["vehicle_id", "timestamp", "x", "y", "speed", "heading", "attack_type"]]


# ─────────────────────────────────────────────────────────────────────────────
# Loader 3 — MOSAIC Replay/Bogus Dataset
# ─────────────────────────────────────────────────────────────────────────────

# Column aliases for the MOSAIC dataset (Iqbal et al. 2022)
# The exact column names depend on which scenario CSV you downloaded.
_MOSAIC_ALIASES = {
    "vehicle_id":  ["name", "vehicle", "sender", "vehicle_id", "node",
                    "source", "id", "vehicleid"],
    "timestamp":   ["time", "timestamp", "simtime", "t", "rcvtime", "sendtime"],
    "x":           ["longitude", "lon", "x", "pos_x", "position_x", "east"],
    "y":           ["latitude",  "lat", "y", "pos_y", "position_y", "north"],
    "speed":       ["speed", "velocity", "spd", "vel"],
    "heading":     ["heading", "direction", "angle", "yaw", "bearing",
                    "course", "head"],
    # MOSAIC attack label: 'attack_type' or 'label'. Scenario files sometimes
    # encode this as 'attack' (0 = replay, 1 = bogus, -1/NaN = legitimate).
    # We normalise: 0 → legitimate, anything else → attacker.
    "attack_type": ["attack_type", "attacktype", "attack", "label",
                    "is_attack", "malicious", "type"],
}

# MOSAIC attack class mapping — scenario files use textual or integer labels
_MOSAIC_ATTACK_MAP = {
    "none":    0, "normal":  0, "legitimate": 0, "legit": 0,
    "replay":  1, "replay_attack": 1,
    "bogus":   2, "bogus_info": 2, "false_info": 2,
}


def _parse_mosaic_attack_col(series):
    """
    Convert MOSAIC's mixed attack column to integer attack_type.
    Handles strings like 'replay', 'bogus', 'None', and integers 0/1/2.
    """
    def convert(v):
        if pd.isna(v):
            return 0
        s = str(v).strip().lower()
        if s in _MOSAIC_ATTACK_MAP:
            return _MOSAIC_ATTACK_MAP[s]
        try:
            i = int(float(s))
            return 0 if i <= 0 else i   # negative = legitimate in some files
        except (ValueError, TypeError):
            return 1  # unknown non-zero string → treat as attacker

    return series.map(convert).fillna(0).astype(int)


def load_mosaic_replay_bogus(data_path, max_vehicles=200, verbose=True):
    """
    Load the MOSAIC Replay/Bogus VANET Dataset (Iqbal et al. 2022).

    Download (free, no login):
        https://drive.google.com/drive/folders/1dDrt1K90B4zn1zi6WT3ZsZ4GstPwMLWg
        Download any or all of the scenario CSV files.

    Set DATA_PATH in config.py to either:
        - A single scenario CSV file, e.g. "replay_scenario_1.csv"
        - A folder containing multiple CSVs (all will be merged)

    Attack types used in this loader:
        0 = Legitimate
        1 = Replay attack (re-broadcasts old valid messages)
        2 = Bogus information attack (fabricated position/speed)
    These map cleanly to your model's anomaly signals:
        Replay attacks  → detected by cosine direction check (recycled
                          gradient direction) and budget check (extra steps)
        Bogus info      → detected by norm check and tier tag spike

    Column mapping (case-insensitive):
        name / vehicle / sender  → vehicle_id
        time / timestamp         → timestamp
        longitude / x            → x
        latitude / y             → y
        speed                    → speed
        heading / direction      → heading
        attack / attack_type     → attack_type
    """
    files = _collect_files(data_path, "csv")
    if not files:
        raise FileNotFoundError(
            f"No CSV files found at: {data_path}\n"
            "Download from https://drive.google.com/drive/folders/"
            "1dDrt1K90B4zn1zi6WT3ZsZ4GstPwMLWg\n"
            "then set DATA_PATH in config.py.")

    if verbose:
        print(f"[MOSAIC] Found {len(files)} CSV file(s)")

    dfs = []
    for f in files:
        try:
            chunk = pd.read_csv(f, low_memory=False)
            # Some MOSAIC files include a scenario/filename column — drop it
            dfs.append(chunk)
        except Exception as e:
            print(f"[MOSAIC] Warning: skipping {f}: {e}")
    if not dfs:
        raise ValueError("No MOSAIC CSV files could be read.")

    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    cols = set(df.columns)

    if verbose:
        print(f"[MOSAIC] Raw columns: {list(df.columns)}")
        print(f"[MOSAIC] Raw shape:   {df.shape}")

    # Map columns
    mapped = {target: next((c for c in cands if c in cols), None)
              for target, cands in _MOSAIC_ALIASES.items()}

    if verbose:
        print(f"[MOSAIC] Column mapping: {mapped}")

    # vehicle_id
    df["vehicle_id"] = (df[mapped["vehicle_id"]].astype(str).str.strip()
                        if mapped["vehicle_id"] else
                        (np.arange(len(df)) // 200).astype(str))

    # timestamp
    df["timestamp"] = (pd.to_numeric(df[mapped["timestamp"]], errors="coerce").fillna(0)
                       if mapped["timestamp"] else
                       np.arange(len(df), dtype=float))

    # x / y
    df["x"] = (pd.to_numeric(df[mapped["x"]], errors="coerce").fillna(0)
               if mapped["x"] else 0.0)
    df["y"] = (pd.to_numeric(df[mapped["y"]], errors="coerce").fillna(0)
               if mapped["y"] else 0.0)

    # speed
    df["speed"] = (pd.to_numeric(df[mapped["speed"]], errors="coerce").fillna(0).abs()
                   if mapped["speed"] else 0.0)

    # heading
    df["heading"] = (pd.to_numeric(df[mapped["heading"]], errors="coerce").fillna(0)
                     if mapped["heading"] else 0.0)

    # attack_type — handles both string ('replay', 'bogus') and integer
    if mapped["attack_type"]:
        df["attack_type"] = _parse_mosaic_attack_col(df[mapped["attack_type"]])
    else:
        # No label column found — attempt to infer from filename convention
        # e.g. "replay_scenario_1.csv" → all rows are replay attacks (type 1)
        # This is a heuristic; ideally the file should contain a label column.
        if verbose:
            print("[MOSAIC] Warning: no attack_type column found. "
                  "Inferring from filenames if possible.")
        attack_types = []
        for f in files:
            fname = os.path.basename(f).lower()
            atype = 1 if "replay" in fname else (2 if "bogus" in fname else 0)
            n_rows = len(pd.read_csv(f, low_memory=False))
            attack_types.extend([atype] * n_rows)
        df["attack_type"] = attack_types[:len(df)]

    df = _ensure_schema(df)
    df = _enforce_vehicle_diversity(df, verbose)
    df = _cap_vehicles_and_rows(df, max_vehicles)

    if verbose:
        print(f"[MOSAIC] Final: {len(df):,} rows, "
              f"{df['vehicle_id'].nunique()} vehicles")
        vc = df.drop_duplicates("vehicle_id").groupby("attack_type")["vehicle_id"].count()
        print(f"[MOSAIC] Attack type distribution: {vc.to_dict()}")

    return df[["vehicle_id", "timestamp", "x", "y", "speed", "heading", "attack_type"]]


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatcher — called by federated_runner.py
# ─────────────────────────────────────────────────────────────────────────────

_LOADERS = {
    "veremi_extension":    load_veremi_extension,
    "kaggle_maliciousnode": load_kaggle_maliciousnode,
    "mosaic_replay_bogus": load_mosaic_replay_bogus,
}


def load_dataset(dataset_name, data_path, max_vehicles=200, verbose=True):
    """
    Entry point for federated_runner.py.
    Dispatches to the correct loader based on DATASET_NAME in config.py.

    Args:
        dataset_name: one of "veremi_extension", "kaggle_maliciousnode",
                      "mosaic_replay_bogus"
        data_path:    file or folder path set in config.py
        max_vehicles: cap on number of unique vehicles to load
        verbose:      print progress messages

    Returns:
        DataFrame with columns:
            vehicle_id, timestamp, x, y, speed, heading, attack_type
    """
    if dataset_name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options: {list(_LOADERS.keys())}")

    if verbose:
        print(f"[Preprocessing] Loading dataset: {dataset_name}")
        print(f"[Preprocessing] Path: {data_path}")

    loader = _LOADERS[dataset_name]
    df = loader(data_path, max_vehicles=max_vehicles, verbose=verbose)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — dataset-agnostic (all loaders feed into this)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df, verbose=True):
    """
    Compute the 4 DT-RBAC-FL-ADP features from the normalised schema:

      pos_zscore    z-score of position jump magnitude vs vehicle history
                    (paper Feature 1 — primary discriminator across all attacks)
      speed_anomaly rolling z-score of speed vs vehicle's 5-reading window
                    (paper Feature 2 — detects random offset / noisy sensor)
      heading_dev   absolute change in heading angle between beacons
                    (paper Feature 3 — physical analogue of gradient direction)
      time_delta    absolute gap between consecutive timestamps
                    (paper Feature 4 — unnaturally dense → likely injected)

    Also adds binary 'label' column (0 = legitimate, 1 = attacker).
    Clips all features to 1st–99th percentile to remove extreme outliers.
    """
    if verbose:
        print("[Preprocessing] Engineering features...")

    parts = []
    for vid, grp in df.sort_values("timestamp").groupby("vehicle_id"):
        grp = grp.reset_index(drop=True).copy()

        # Feature 1: position jump z-score
        dists = np.sqrt(grp["x"].diff() ** 2 + grp["y"].diff() ** 2).fillna(0)
        mu, sig = dists.mean(), dists.std() + 1e-9
        grp["pos_zscore"] = (dists - mu) / sig

        # Feature 2: speed anomaly (rolling z-score over last 5 readings)
        rm  = grp["speed"].rolling(5, min_periods=1).mean()
        rs  = grp["speed"].rolling(5, min_periods=1).std().fillna(1e-9) + 1e-9
        grp["speed_anomaly"] = ((grp["speed"] - rm) / rs).abs()

        # Feature 3: heading deviation between consecutive readings
        grp["heading_dev"] = grp["heading"].diff().abs().fillna(0)

        # Feature 4: inter-reading time delta
        grp["time_delta"] = grp["timestamp"].diff().abs().fillna(0)

        parts.append(grp)

    result = pd.concat(parts).reset_index(drop=True)

    # Binary label: 0 → legitimate, any non-zero attack_type → 1
    result["label"] = (result["attack_type"] != 0).astype(int)

    # Clip outliers (1st–99th percentile per feature)
    for col in FEATURE_COLS:
        lo = result[col].quantile(0.01)
        hi = result[col].quantile(0.99)
        result[col] = result[col].clip(lo, hi)

    if verbose:
        legit  = (result["label"] == 0).sum()
        attack = (result["label"] == 1).sum()
        print(f"[Preprocessing] Features ready — legit: {legit:,}  "
              f"attack: {attack:,}  ratio: {attack / (legit + attack) * 100:.1f}%")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-vehicle dataset preparation — dataset-agnostic
# ─────────────────────────────────────────────────────────────────────────────

def prepare_vehicle_datasets(df, scaler=None, fit_scaler=True):
    """
    Scale FEATURE_COLS and split into per-vehicle dicts for the FL runner.

    Returns:
        vdata:  {vehicle_id: {"X": np.array, "y": np.array, "attack_type": int}}
        scaler: fitted StandardScaler (pass back in on test set with fit_scaler=False)
    """
    df = df.copy()
    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    else:
        df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    vdata = {}
    for vid, grp in df.groupby("vehicle_id"):
        vdata[vid] = {
            "X":           grp[FEATURE_COLS].values.astype(np.float32),
            "y":           grp["label"].values.astype(np.int64),
            "attack_type": int(grp["attack_type"].iloc[0]),
        }
    return vdata, scaler
