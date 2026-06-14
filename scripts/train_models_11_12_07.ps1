$ErrorActionPreference = 'Continue'
Set-Location "$PSScriptRoot\.."

function Initialize-GpuEnvironment {
    $nvBase = Resolve-Path '.venv_gpu/Lib/site-packages/nvidia' -ErrorAction SilentlyContinue
    if (-not $nvBase) { throw 'NVIDIA package dir not found' }
    $dllDirs = Get-ChildItem $nvBase -Recurse -Filter '*.dll' | Select -ExpandProperty DirectoryName | Sort -Unique
    $exeDirs = Get-ChildItem $nvBase -Recurse -Filter '*.exe' | Select -ExpandProperty DirectoryName | Sort -Unique
    $env:PATH = ($dllDirs + $exeDirs -join ';') + ';' + $env:PATH
    $env:TF_CPP_MIN_LOG_LEVEL = '1'
    $env:PYTHONPATH = 'ai_pipeline'
}

Initialize-GpuEnvironment

Write-Host 'Waiting for existing ai_pipeline/train.py processes to finish...'
while ($true) {
    $procs = Get-CimInstance Win32_Process -Filter 'Name="python.exe"' | Where-Object { $_.CommandLine -match 'ai_pipeline/train.py' }
    if (-not $procs) { break }
    Write-Host 'Models 5 and 6 are still running. Checking again in 30 seconds...'
    Start-Sleep -Seconds 30
}

Write-Host '================== MODEL 11 (LINEAR MULTI-TASK) =================='
$p11 = Start-Process -FilePath '.\.venv_gpu\Scripts\python.exe' -ArgumentList '-u ai_pipeline/train.py --type=linear --model=models/model_11_multitask.keras --simulationconfig=configs/model_11_multitask.py --tubs=data/datasets/mega_dataset/tubs' -PassThru -NoNewWindow
$p11.WaitForExit()

Write-Host '================== MODEL 12 (TEMPORAL RNN) =================='
$p12 = Start-Process -FilePath '.\.venv_gpu\Scripts\python.exe' -ArgumentList '-u ai_pipeline/train.py --type=rnn --model=models/model_12_temporal.keras --simulationconfig=configs/model_12_temporal.py --tubs=data/datasets/mega_dataset/tubs' -PassThru -NoNewWindow
$p12.WaitForExit()

Write-Host '================== MODEL 07 (FINE-TUNED SIM2REAL) =================='
$p07 = Start-Process -FilePath '.\.venv_gpu\Scripts\python.exe' -ArgumentList '-u ai_pipeline/train.py --type=target_point --model=models/model_07_finetune.keras --simulationconfig=configs/model_07_finetune.py' -PassThru -NoNewWindow
$p07.WaitForExit()

Write-Host 'All 3 post-models completed.'
