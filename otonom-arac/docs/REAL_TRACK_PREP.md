# Real Track Readiness (Jetson Nano)

This checklist is the current runnable flow for first safe real-track tests.
Run all commands from `otonom-arac/` with PowerShell.

## 1) Train/Fine-tune (mega_dataset + quality gates + steering balance)

```powershell
$env:PYTHONPATH='ai_pipeline'
.venv\Scripts\python ai_pipeline\train.py `
  --type target_point `
  --model models\target_point_realtrack_ready.keras `
  --epochs 10 `
  --batch-size 32 `
  --device auto
```

Expected artifacts:
- `data/artifacts/target_point/experiments/<run>/dataset_quality_report.json`
- `data/artifacts/target_point/experiments/<run>/metrics.json`

## 2) Closed-loop simulator gate (before Jetson)

```powershell
$env:PYTHONPATH='ai_pipeline'
.venv\Scripts\python ai_pipeline\evaluate_target_point.py `
  --model models\target_point_realtrack_ready.keras `
  --tracks donkey-generated-roads-v0,donkey-generated-track-v0,donkey-minimonaco-track-v0 `
  --episodes-per-track 3 `
  --max-steps 1200
```

Review `closed_loop_summary.json` in the newest report directory.

## 3) Export FP16 TFLite (Jetson baseline)

```powershell
$env:PYTHONPATH='ai_pipeline'
.venv\Scripts\python ai_pipeline\target_point\export.py `
  --model models\target_point_realtrack_ready.keras `
  --output models\target_point_realtrack_ready_fp16.tflite `
  --quantize float16 `
  --benchmark `
  --simulationconfig simulationconfig.py
```

## 4) Safe first on-track run profile

Default throttle profile in `simulationconfig.py` is now conservative:
- `TARGET_POINT_BASE_THROTTLE = 0.20`
- `TARGET_POINT_MIN_THROTTLE = 0.07`

This profile was selected from a quick closed-loop sweep as the most stable candidate for first tests.

## 5) Jetson run command

Copy model + config to Jetson and start drive mode with target_point:

```powershell
python manage.py drive `
  --type target_point `
  --model models/target_point_realtrack_ready_fp16.tflite `
  --simulationconfig simulationconfig.py
```

## 6) Real track acceptance log (mandatory)

For each run, log:
- lap count
- off-track count
- manual takeover count
- longest uninterrupted autonomous duration
- average speed
- failed corner names
- recovery success/fail count

Use the same track and camera mount for all comparisons.
