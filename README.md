cat > ~/Downloads/vanet_privacy_clean/README.md << 'EOF'
# Secure ITS — DT-RBAC-FL-ADP

DT-Enhanced Secure Federated Learning in VANET for smart city traffic management.
Implements the DT-RBAC-FL-ADP framework proposed in "Secure ITS for Future Cities" (Hansikaa Aggarwal, 2026).

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install kagglehub gdown
```

### 2. Download a dataset
```bash
python3 download_datasets.py
```
Enter 1, 2, or 3 when prompted. It prints the exact `DATA_PATH` to paste into `config.py`.

### 3. Set dataset in config.py
```python
DATASET_NAME = "veremi_extension"   # or "kaggle_maliciousnode"
DATA_PATH    = "/path/printed/by/download_datasets.py"
```

### 4. Run the model
```bash
python3 federated_runner.py
```

### 5. Evaluate anomaly detection
```bash
python3 evaluate_anomaly.py
```

### 6. Generate charts
```bash
python3 plot_results.py
```

Charts and metrics saved to `results/`.

---

## Supported Datasets

| # | Dataset | How to get |
|---|---------|------------|
| 1 | VeReMi Extension (Kamel et al. 2020) | Auto-downloaded via `download_datasets.py` → option 1 |
| 2 | VeReMi Original (2018) | Auto-downloaded via `download_datasets.py` → option 2 |
| 3 | Kaggle VANET-MaliciousNode | Auto-downloaded via `download_datasets.py` → option 3 (requires free Kaggle account + API token) |

To run on multiple datasets, re-run `federated_runner.py` after changing `config.py` and save results:
```bash
cp -r results/ results_veremi_extension/
cp -r results/ results_veremi_original/
cp -r results/ results_kaggle/
```

---

## Project Structure

```
secure_its/
├── config.py             ← SET YOUR DATA PATH HERE
├── requirements.txt
├── preprocessing.py      ← VeReMi loading + feature engineering
├── models.py             ← AttackDetector neural network
├── vehicle.py            ← Vehicle layer (RBAC, FL, clipping, ADP noise)
├── rsu_dt.py             ← RSU Digital Twin (α computation, ε scheduler,
│                            anomaly detection, weighted FedAvg)
├── cloud.py              ← Cloud layer (global aggregation, ε policy)
├── federated_runner.py   ← Main orchestrator
├── plot_results.py       ← Chart generation
├── data/veremi/          ← Put VeReMi JSON files here (or update config.py)
└── results/              ← Output charts and metrics
```

---

## What the model does

### Your model (DT-RBAC-FL-ADP)
- **RBAC**: Vehicles assigned roles (infrastructure/fleet/private) with pre-set trust
- **α formula**: `α = 0.5·trust + 0.3·stability + 0.2·budget`
- **ε scheduling**: `εᵥ = εmin + α·(εmax − εmin)` — personalised per vehicle per round
- **Anomaly detection**: norm check (3σ), cosine check (< 0.70), tier tag check
- **Weighted FedAvg**: `weight = trust × |D_i|`, flagged vehicles down-weighted ×0.30 / ×0.10
- **Trust updates**: +0.05 clean round, −0.10 per flag, RBAC rights change at 0.70/0.30/0.10

### Baselines
- **FL + DP**: Fixed global ε = 0.50 for all vehicles, standard FedAvg
- **DT + FL**: Twin used only for participation weighting, no DP
- **Adaptive DP-FL**: ε adapts per round based on gradient norm history, no trust scores

---

## Privacy parameters (config.py)

| Parameter | Default | Reference |
|-----------|---------|-----------|
| ε total   | 1.00    | Dwork & Roth (2014) |
| ε min     | 0.05    | — |
| ε max     | 1.50    | — |
| Clip C    | 1.0     | Abadi et al. (2016) |
| δ         | 1e-5    | Abadi et al. (2016) |

---

## Output charts

| File | Description |
|------|-------------|
| `accuracy_over_rounds.png` | Test accuracy per FL round, all 4 models |
| `f1_over_rounds.png` | Attack detection F1 per round |
| `final_comparison_bar.png` | Final accuracy/F1/precision/recall bar chart |
| `privacy_utility_frontier.png` | Privacy-utility tradeoff curve |
| `trust_and_flags.png` | Flagged vehicles + avg trust per round |

---

## Expected results

Your model should outperform baselines on:
- **F1 / recall** — because trust-weighted FedAvg removes attacker gradients
- **Late-round accuracy** — trust scores stabilise after ~round 8
- **Attack detection** — cosine check catches slow-drift attacks others miss

DT+FL will have highest raw accuracy early (no DP noise penalty) but
provides zero privacy guarantee, making the comparison unfair in a security context.
