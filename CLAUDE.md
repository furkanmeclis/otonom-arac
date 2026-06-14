# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Python version: 3.11 exactly** (TensorFlow 2.15.1 requirement)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-train.txt
```

Before training or collecting data, set `DONKEY_SIM_PATH` in [simulationconfig.py](simulationconfig.py) to point to the DonkeySim binary.

Smoke test the setup:
```powershell
python manage.py smoke --simulationconfig=simulationconfig.py
```

## Common Commands

**Data collection (3 stages per track):**
```powershell
python ai_pipeline/collect_target_point_data.py --task map --track generated_roads --simulationconfig=simulationconfig.py
python ai_pipeline/collect_target_point_data.py --task collect --collection-profile phase2_low_noise --track generated_roads --simulationconfig=simulationconfig.py
python ai_pipeline/collect_target_point_data.py --task collect --collection-profile phase3_full_noise --track generated_roads --simulationconfig=simulationconfig.py
```

**Label generation:**
```powershell
python ai_pipeline/build_target_point_labels.py --raw-roots data/sim/generated_roads --output-dir data/sim_multitrack/index
```

**Training:**
```powershell
python ai_pipeline/train.py --type target_point --manifest data/sim_multitrack/index --model models/model.keras --label-mode adaptive_v1 --config configs/model_01_pure_sim.py
```

**Evaluation (closed-loop):**
```powershell
python ai_pipeline/evaluate_target_point.py --model models/model.keras --tracks generated_roads,mini_monaco --episodes-per-track 10 --simulationconfig=simulationconfig.py
```

**Driving (real car or sim):**
```powershell
python manage.py drive --model models/model.keras --type target_point --simulationconfig=simulationconfig.py
```

**Tests:**
```powershell
pytest tests/
pytest tests/test_target_point.py  # single test file
```

Batch training scripts are in [scripts/](scripts/) as PowerShell `.ps1` files.

## Architecture

### Core Approach: Target-Point Prediction

Instead of predicting steering angle directly, the model predicts a `(target_x, target_y)` waypoint in ego-frame meters. A geometric controller then converts this to steering/throttle commands using `atan2`. This decouples geometry from vehicle dynamics for better sim-to-real transfer.

### Data Format

Raw data uses **DonkeyCar Tub V2** format (images + JSON records). The pipeline converts this to **JSONL manifests** with `(target_x, target_y)` labels computed relative to a teacher reference trajectory stored per track.

### Key Modules

| Module | Role |
|---|---|
| [manage.py](manage.py) | CLI entry point; orchestrates hardware parts (camera, AI, joystick, web UI on port 8887) |
| [config.py](config.py) | Hardware config template (771 lines) — pin mappings, camera type, sensor settings |
| [simulationconfig.py](simulationconfig.py) | Extends config.py; defines model hyperparameters, augmentation limits, sim binary path |
| [ai_pipeline/train.py](ai_pipeline/train.py) | Training orchestrator; handles GPU auto-detection and Windows CUDA DLL setup |
| [ai_pipeline/build_target_point_labels.py](ai_pipeline/build_target_point_labels.py) | Converts Tub data → JSONL manifest with ego-frame target labels |
| [ai_pipeline/collect_target_point_data.py](ai_pipeline/collect_target_point_data.py) | Multi-stage data collection (map extraction, low-noise, full-noise/recovery, rollout) |
| [ai_pipeline/evaluate_target_point.py](ai_pipeline/evaluate_target_point.py) | Closed-loop evaluation; reports lap completion rate and off-track incidents |
| [target_point/model.py](target_point/model.py) | CNN architecture — depthwise separable convolutions, ~115K params (vs 5.2M legacy) |
| [target_point/training.py](target_point/training.py) | Training loop, weighted MSE loss, sample weighting per scenario |
| [target_point/controller.py](target_point/controller.py) | Geometric heading-error → steering (tanh gain); dynamic throttle from curvature |

### Model Configurations (`configs/`)

Nine experiment configs control dataset mix, augmentation strength, and architecture variants:

| Config | Strategy |
|---|---|
| `model_01_pure_sim.py` | Baseline: pure simulation, no augmentation |
| `model_02_sim_domain_randomization.py` | Aggressive domain randomization for sim-to-real |
| `model_03_pure_real.py` | Pure real-world data (816K frames, Jetson Nano) |
| `model_04_hybrid_v1_naive_mix.py` | 70% sim / 30% real — marked failed |
| `model_05_hybrid_v2_sim_heavy.py` | 90% sim / 10% real |
| `model_06_hybrid_v3_real_heavy.py` | 30% sim / 70% real |
| `model_07_finetune.py` | Transfer learning: pretrained on sim, 6 layers frozen, 5× lower LR |
| `model_11_multitask.py` | Multi-task: joint steering + throttle prediction |
| `model_12_temporal.py` | LSTM over 5-frame sequences |

### Model I/O

- **Input**: 224×224 RGB image, cropped and normalized to [0, 1]
- **Output**: `(target_x, target_y)` in ego-frame meters
- **Loss**: Weighted MSE; weights vary by scenario (curvature, recovery, turns)
- **Augmentation**: brightness ±20%, rotation ±2.5°, horizontal flip, temporal perturbations

### Deployment Target

Jetson Nano with USB camera. The model is kept at ~115K parameters for real-time inference on embedded hardware. Tracks used: Generated Roads, Mini Monaco, Generated Track.

## Documentation (Turkish)

- [CALISTIRMA_REHBERI.md](CALISTIRMA_REHBERI.md) — Full step-by-step execution guide
- [PROJE_ACIKLAMA.md](PROJE_ACIKLAMA.md) — Detailed file-by-file reference with data flow diagram
- [MODEL_EGITIM_SURECI.md](MODEL_EGITIM_SURECI.md) — Training workflow and parameter reference
