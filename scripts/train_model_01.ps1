# MODEL-01 - Pure Sim (Baseline)
# Sadece sim verisiyle egitim. External data yok.

$ErrorActionPreference = "Stop"
$StartTime = Get-Date

Set-Location "$PSScriptRoot\.."

Write-Host "============================================================"
Write-Host " MODEL-01 - Pure Sim (Baseline)"
Write-Host " Baslangic: $StartTime"
Write-Host "============================================================"

# NVIDIA DLL'lerini venv_gpu'dan bul ve PATH'e ekle (train_gpu.ps1 yontemi)
$nvBase = Resolve-Path ".venv_gpu/Lib/site-packages/nvidia" -ErrorAction SilentlyContinue
if (-not $nvBase) {
    Write-Host "HATA: .venv_gpu/nvidia paketleri bulunamadi."
    exit 1
}
$dllDirs = Get-ChildItem $nvBase -Recurse -Filter "*.dll" |
    Select-Object -ExpandProperty DirectoryName | Sort-Object -Unique
$exeDirs = Get-ChildItem $nvBase -Recurse -Filter "*.exe" |
    Select-Object -ExpandProperty DirectoryName | Sort-Object -Unique
$env:PATH = ($dllDirs + $exeDirs -join ";") + ";" + $env:PATH
$env:TF_CPP_MIN_LOG_LEVEL = "1"
$env:PYTHONPATH = "ai_pipeline"
$nvccRoot = Join-Path $nvBase "cuda_nvcc"
if (Test-Path -LiteralPath $nvccRoot) {
    $xlaFlag = "--xla_gpu_cuda_data_dir=$nvccRoot"
    if (-not $env:XLA_FLAGS -or $env:XLA_FLAGS -notmatch "xla_gpu_cuda_data_dir") {
        $env:XLA_FLAGS = if ($env:XLA_FLAGS) { "$($env:XLA_FLAGS) $xlaFlag" } else { $xlaFlag }
    }
}

Write-Host "NVIDIA DLL dizinleri PATH'e eklendi: $($dllDirs.Count) klasor"

# GPU kontrol
$gpuCheck = & .venv_gpu/Scripts/python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('GPU:' + str(len(gpus)))
" 2>&1 | Select-String "^GPU:"

if ($gpuCheck -match "GPU:0") {
    Write-Host "UYARI: GPU gorunmuyor, CPU ile devam ediliyor."
} else {
    Write-Host "GPU aktif: $gpuCheck"
}

Write-Host ""
Write-Host "Egitim basliyor..."
Write-Host "Config : configs/model_01_pure_sim.py"
Write-Host "Model  : models/model_01_pure_sim.keras"
Write-Host "Veri   : sadece sim_mega_dataset"
Write-Host ""

& .venv_gpu/Scripts/python -u ai_pipeline/train.py `
    --type=target_point `
    --model=models/model_01_pure_sim.keras `
    --simulationconfig=configs/model_01_pure_sim.py

$ExitCode = $LASTEXITCODE
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host ""
Write-Host "============================================================"
if ($ExitCode -eq 0) {
    Write-Host "MODEL-01 basariyla tamamlandi!"
} else {
    Write-Host "MODEL-01 HATA ile bitti. Exit code: $ExitCode"
}
Write-Host "Bitis  : $EndTime"
Write-Host "Sure   : $($Duration.Hours)s $($Duration.Minutes)d $($Duration.Seconds)sn"
Write-Host "============================================================"
