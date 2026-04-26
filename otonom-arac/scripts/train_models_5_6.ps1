$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\.."

function Initialize-GpuEnvironment {
    $nvBase = Resolve-Path ".venv_gpu/Lib/site-packages/nvidia" -ErrorAction SilentlyContinue
    if (-not $nvBase) {
        throw "NVIDIA package directory not found under .venv_gpu."
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
}

function Get-LatestLogLine([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return ""
    }
    $tail = Get-Content -LiteralPath $Path -Tail 50 -ErrorAction SilentlyContinue
    if (-not $tail) {
        return ""
    }
    $line = $tail | Where-Object { $_ -and $_.Trim().Length -gt 0 } | Select-Object -Last 1
    if (-not $line) {
        return ""
    }
    return $line.Trim()
}

function Read-ProcessExitCode {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [int]$MaxAttempts = 20,
        [int]$SleepMs = 250
    )

    for ($i = 0; $i -lt $MaxAttempts; $i++) {
        try {
            $Process.Refresh()
            return [int]$Process.ExitCode
        }
        catch {
            Start-Sleep -Milliseconds $SleepMs
        }
    }

    return $null
}

function Stop-ExistingTraining {
    $running = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match "python" -and $_.CommandLine -match "ai_pipeline/train.py"
    }
    foreach ($proc in $running) {
        Write-Host "Stopping existing training process PID=$($proc.ProcessId)"
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Initialize-GpuEnvironment
Stop-ExistingTraining

$runTag = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = Join-Path "data\artifacts\multi_model_first6" $runTag
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$summaryPath = Join-Path $runRoot "summary.csv"
"model_id,attempt,status,exit_code,duration_sec,model_path,config_path,out_log,err_log" | Out-File -LiteralPath $summaryPath -Encoding utf8

$models = @(
    @{
        Id = "MODEL-05"
        Config = "configs/model_05_hybrid_v2_sim_heavy.py"
        ModelPath = "models/model_05_hybrid_v2_sim_heavy.keras"
        Experiment = "model_05_hybrid_v2_sim_heavy"
        Epochs = 30
    },
    @{
        Id = "MODEL-06"
        Config = "configs/model_06_hybrid_v3_real_heavy.py"
        ModelPath = "models/model_06_hybrid_v3_real_heavy.keras"
        Experiment = "model_06_hybrid_v3_real_heavy"
        Epochs = 30
    }
)

$maxRetries = 3

Write-Host "============================================================"
Write-Host " First 6 model training pipeline started"
Write-Host " Run tag : $runTag"
Write-Host " Logs    : $runRoot"
Write-Host "============================================================"

foreach ($model in $models) {
    $modelId = [string]$model.Id
    $configPath = [string]$model.Config
    $modelPath = [string]$model.ModelPath
    $epochs = [int]$model.Epochs
    $experimentBase = [string]$model.Experiment

    if (-not (Test-Path -LiteralPath $configPath)) {
        throw "Config not found for ${modelId}: $configPath"
    }

    $success = $false
    for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
        $outLog = Join-Path $runRoot ("{0}_attempt{1}.out.log" -f $modelId, $attempt)
        $errLog = Join-Path $runRoot ("{0}_attempt{1}.err.log" -f $modelId, $attempt)
        "" | Out-File -LiteralPath $outLog -Encoding utf8
        "" | Out-File -LiteralPath $errLog -Encoding utf8

        $experimentName = "{0}_{1}_a{2}" -f $experimentBase, $runTag, $attempt
        $args = @(
            "-u",
            "ai_pipeline/train.py",
            "--type=target_point",
            "--device=gpu",
            "--model=$modelPath",
            "--simulationconfig=$configPath",
            "--epochs=$epochs",
            "--experiment-name=$experimentName"
        )

        Write-Host ""
        Write-Host "[$modelId] Attempt $attempt/$maxRetries starting..."
        $startTime = Get-Date

        $proc = Start-Process -FilePath ".\.venv_gpu\Scripts\python.exe" `
            -ArgumentList $args `
            -WorkingDirectory (Get-Location) `
            -RedirectStandardOutput $outLog `
            -RedirectStandardError $errLog `
            -PassThru

        while (-not $proc.HasExited) {
            Start-Sleep -Seconds 20
            $latest = Get-LatestLogLine -Path $outLog
            if ($latest) {
                Write-Host ("[{0}][A{1}] {2}" -f $modelId, $attempt, $latest)
            }
            $proc.Refresh()
        }

        $proc.WaitForExit()
        $exitCode = Read-ProcessExitCode -Process $proc

        $metricsPath = Join-Path "data\artifacts\target_point\experiments" ("{0}\metrics.json" -f $experimentName)
        if ($null -eq $exitCode) {
            if (Test-Path -LiteralPath $metricsPath) {
                Write-Host "[$modelId] ExitCode unavailable but metrics.json found. Treating run as success."
                $exitCode = 0
            }
            else {
                $exitCode = -999
            }
        }
        $duration = (Get-Date) - $startTime
        $status = if ($exitCode -eq 0) { "success" } else { "failed" }

        ("{0},{1},{2},{3},{4},{5},{6},{7},{8}" -f `
            $modelId, $attempt, $status, $exitCode, [int]$duration.TotalSeconds, `
            $modelPath, $configPath, $outLog, $errLog) | Add-Content -LiteralPath $summaryPath

        if ($exitCode -eq 0) {
            Write-Host "[$modelId] Attempt $attempt completed successfully in $([int]$duration.TotalMinutes)m $([int]$duration.Seconds)s."
            $success = $true
            break
        }

        Write-Host "[$modelId] Attempt $attempt failed (exit=$exitCode)."
        $lastErr = Get-LatestLogLine -Path $errLog
        if ($lastErr) {
            Write-Host "[$modelId] Last error: $lastErr"
        }

        if ($attempt -lt $maxRetries) {
            Write-Host "[$modelId] Retrying in 15 seconds..."
            Start-Sleep -Seconds 15
        }
    }

    if (-not $success) {
        throw "$modelId failed after $maxRetries attempts. Check $summaryPath"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host " First 6 model training pipeline finished successfully."
Write-Host " Summary : $summaryPath"
Write-Host "============================================================"
