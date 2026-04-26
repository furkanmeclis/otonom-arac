param(
  [string]$Root = "data/datasets/sim_mega_dataset",
  [string]$SimulationConfig = "simulationconfig.py",
  [string]$DriverModel = "models/target_point_realtrack_ready.keras",
  [int]$MapLaps = 2,
  [int]$MapMaxSteps = 3000,
  [int]$CollectMaxSteps = 1800,
  [int]$CleanMinSamples = 12000,
  [int]$LowNoiseMinSamples = 15000,
  [int]$HighNoiseMinSamples = 15000,
  [int]$RecoveryMinUsable = 6000,
  [int]$CollectEpisodesCap = 25,
  [int]$RecoveryEpisodesCap = 40,
  [double]$CleanThrottle = 0.12,
  [double]$LowNoiseThrottle = 0.13,
  [double]$HighNoiseThrottle = 0.13,
  [int]$Seed = 42
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$tracks = @(
  'donkey-generated-roads-v0',
  'donkey-generated-track-v0',
  'donkey-minimonaco-track-v0',
  'donkey-warren-track-v0',
  'donkey-warehouse-v0',
  'donkey-circuit-launch-track-v0',
  'donkey-roboracingleague-track-v0'
)

$python = '.venv\\Scripts\\python'
$entry = 'ai_pipeline\\collect_target_point_data.py'

$rootAbs = Resolve-Path . | ForEach-Object { Join-Path $_ $Root }
New-Item -ItemType Directory -Force -Path $rootAbs | Out-Null
$logDir = Join-Path $rootAbs '_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runLog = Join-Path $logDir "massive_collection_$runStamp.log"

"[start] $(Get-Date -Format o)" | Tee-Object -FilePath $runLog -Append
"[root] $rootAbs" | Tee-Object -FilePath $runLog -Append
"[driver_model] $DriverModel" | Tee-Object -FilePath $runLog -Append

$env:PYTHONPATH = 'ai_pipeline'

function Invoke-Step {
  param([string]$Cmd)
  "[cmd] $Cmd" | Tee-Object -FilePath $runLog -Append
  cmd /c "$Cmd 2>&1" | Tee-Object -FilePath $runLog -Append
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $Cmd"
  }
}

function Get-JpgCount {
  param([string]$Dir)
  if (-not (Test-Path $Dir)) { return 0 }
  return (Get-ChildItem $Dir -Recurse -Filter *.jpg -File -ErrorAction SilentlyContinue | Measure-Object).Count
}

function Test-MapDone {
  param([string]$MapRoot)
  if (-not (Test-Path $MapRoot)) { return $false }
  $meta = Get-ChildItem $MapRoot -Recurse -Filter metadata.json -File -ErrorAction SilentlyContinue | Select-Object -First 1
  return [bool]$meta
}

function Get-RecoveryUsableCount {
  param(
    [string]$RecoveryRoot,
    [string]$Track
  )
  $summary = Join-Path $RecoveryRoot 'index\\rollout_collection_summary.json'
  if (-not (Test-Path $summary)) { return 0 }
  try {
    $obj = Get-Content $summary -Raw | ConvertFrom-Json
    $prop = $obj.usable_recovery_samples_by_track.PSObject.Properties | Where-Object { $_.Name -eq $Track } | Select-Object -First 1
    if ($prop) { return [int]$prop.Value }
  } catch {
    return 0
  }
  return 0
}

foreach ($track in $tracks) {
  $slug = $track.Replace('-', '_')
  $trackRoot = Join-Path $rootAbs $slug
  $mapRoot = Join-Path $trackRoot 'maps'
  $cleanRoot = Join-Path $trackRoot 'clean'
  $lowRoot = Join-Path $trackRoot 'low_noise'
  $highRoot = Join-Path $trackRoot 'high_noise'
  $recoveryRoot = Join-Path $trackRoot 'recovery'

  New-Item -ItemType Directory -Force -Path $mapRoot,$cleanRoot,$lowRoot,$highRoot,$recoveryRoot | Out-Null

  "[track] $track" | Tee-Object -FilePath $runLog -Append

  # Phase 1 map
  if (Test-MapDone -MapRoot $mapRoot) {
    "[skip] map already exists for $track" | Tee-Object -FilePath $runLog -Append
  } else {
    Invoke-Step "$python $entry --task map --simulationconfig $SimulationConfig --track $track --laps $MapLaps --max-steps $MapMaxSteps --seed $Seed --output-root $mapRoot"
  }

  # Clean (nominal-only + low-noise profile)
  $cleanCount = Get-JpgCount -Dir $cleanRoot
  if ($cleanCount -ge $CleanMinSamples) {
    "[skip] clean complete for $track (jpg=$cleanCount target=$CleanMinSamples)" | Tee-Object -FilePath $runLog -Append
  } else {
    Invoke-Step "$python $entry --task collect --simulationconfig $SimulationConfig --maps-root $mapRoot --train-tracks $track --val-tracks $track --episodes-per-track $CollectEpisodesCap --max-steps $CollectMaxSteps --collection-profile phase2_low_noise --nominal-only-tracks $track --fixed-throttle $CleanThrottle --min-samples-per-track $CleanMinSamples --seed $Seed --output-root $cleanRoot"
  }

  # Low noise
  $lowCount = Get-JpgCount -Dir $lowRoot
  if ($lowCount -ge $LowNoiseMinSamples) {
    "[skip] low_noise complete for $track (jpg=$lowCount target=$LowNoiseMinSamples)" | Tee-Object -FilePath $runLog -Append
  } else {
    Invoke-Step "$python $entry --task collect --simulationconfig $SimulationConfig --maps-root $mapRoot --train-tracks $track --val-tracks $track --episodes-per-track $CollectEpisodesCap --max-steps $CollectMaxSteps --collection-profile phase2_low_noise --fixed-throttle $LowNoiseThrottle --min-samples-per-track $LowNoiseMinSamples --seed $Seed --output-root $lowRoot"
  }

  # High noise
  $highCount = Get-JpgCount -Dir $highRoot
  if ($highCount -ge $HighNoiseMinSamples) {
    "[skip] high_noise complete for $track (jpg=$highCount target=$HighNoiseMinSamples)" | Tee-Object -FilePath $runLog -Append
  } else {
    Invoke-Step "$python $entry --task collect --simulationconfig $SimulationConfig --maps-root $mapRoot --train-tracks $track --val-tracks $track --episodes-per-track $CollectEpisodesCap --max-steps $CollectMaxSteps --collection-profile phase3_full_noise --fixed-throttle $HighNoiseThrottle --min-samples-per-track $HighNoiseMinSamples --seed $Seed --output-root $highRoot"
  }

  # Recovery (model rollout)
  $usableRecovery = Get-RecoveryUsableCount -RecoveryRoot $recoveryRoot -Track $track
  $recoveryJpg = Get-JpgCount -Dir $recoveryRoot
  if (($usableRecovery -ge $RecoveryMinUsable) -or ($recoveryJpg -ge $RecoveryMinUsable)) {
    "[skip] recovery complete for $track (usable=$usableRecovery jpg=$recoveryJpg target=$RecoveryMinUsable)" | Tee-Object -FilePath $runLog -Append
  } else {
    Invoke-Step "$python $entry --task rollout_collect --simulationconfig $SimulationConfig --maps-root $mapRoot --train-tracks $track --episodes-per-track $RecoveryEpisodesCap --max-steps $CollectMaxSteps --driver-model $DriverModel --min-usable-recovery-samples $RecoveryMinUsable --seed $Seed --output-root $recoveryRoot"
  }

  "[track_done] $track" | Tee-Object -FilePath $runLog -Append
}

"[done] $(Get-Date -Format o)" | Tee-Object -FilePath $runLog -Append
