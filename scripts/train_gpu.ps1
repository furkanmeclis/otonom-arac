param(
  [string]$Model = "models/sim_multitrack_v1.keras",
  [string]$LabelMode = "adaptive_v1",
  [string]$SimConfig = "simulationconfig.py"
)

$ErrorActionPreference = 'Stop'

# Collect all pip-installed NVIDIA CUDA DLL directories
$nvBase = Resolve-Path ".venv_gpu/Lib/site-packages/nvidia" -ErrorAction SilentlyContinue
if (-not $nvBase) {
  Write-Host "[error] .venv_gpu not found. Run setup first." -ForegroundColor Red
  exit 1
}
$dllDirs = Get-ChildItem $nvBase -Recurse -Filter '*.dll' |
  Select-Object -ExpandProperty DirectoryName | Sort-Object -Unique
$exeDirs = Get-ChildItem $nvBase -Recurse -Filter '*.exe' |
  Select-Object -ExpandProperty DirectoryName | Sort-Object -Unique
$env:PATH = ($dllDirs + $exeDirs -join ';') + ';' + $env:PATH
$env:TF_CPP_MIN_LOG_LEVEL = '1'
$env:PYTHONPATH = 'ai_pipeline'

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GPU TRAINING  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
Write-Host "  Model : $Model" -ForegroundColor Cyan
Write-Host "  Labels: $LabelMode" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Quick GPU sanity check
$gpuCheck = & .venv_gpu/Scripts/python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('GPU:' + str(len(gpus)))
" 2>&1 | Select-String "^GPU:"
if ($gpuCheck -match "GPU:0") {
  Write-Host "[error] GPU not visible to TF. Check CUDA setup." -ForegroundColor Red
  exit 1
}
Write-Host ("[gpu] " + $gpuCheck) -ForegroundColor Green

& .venv_gpu/Scripts/python ai_pipeline/train.py `
  --type target_point `
  --model $Model `
  --label-mode $LabelMode `
  --simulationconfig $SimConfig

Write-Host "[done] exit=$LASTEXITCODE" -ForegroundColor Cyan
