param(
  [string]$Root = 'data/datasets/sim_mega_dataset',
  [int]$BarWidth = 30,
  [switch]$Watch,
  [int]$RefreshSec = 5,
  [int]$CleanTarget = 12000,
  [int]$LowTarget = 15000,
  [int]$HighTarget = 15000,
  [int]$RecoveryTarget = 6000
)

$tracks = @(
  'donkey-generated-roads-v0',
  'donkey-generated-track-v0',
  'donkey-minimonaco-track-v0',
  'donkey-warren-track-v0',
  'donkey-warehouse-v0',
  'donkey-circuit-launch-track-v0',
  'donkey-roboracingleague-track-v0'
)

$phaseNames = @{
  'map'       = 'MAP'
  'clean'     = 'CLEAN'
  'low_noise' = 'LOW-NOISE'
  'high_noise'= 'HIGH-NOISE'
  'recovery'  = 'RECOVERY'
  'collect'   = 'COLLECT'
  'idle'      = 'IDLE'
}

function Get-JpgCount {
  param([string]$Dir)
  if (-not (Test-Path $Dir)) { return 0 }
  return (Get-ChildItem $Dir -Recurse -Filter *.jpg -File -ErrorAction SilentlyContinue | Measure-Object).Count
}

function Get-RecoveryUsable {
  param([string]$RecoveryRoot, [string]$Track)
  $summary = Join-Path $RecoveryRoot 'index\rollout_collection_summary.json'
  if (-not (Test-Path $summary)) { return 0 }
  try {
    $obj = Get-Content $summary -Raw | ConvertFrom-Json
    $prop = $obj.usable_recovery_samples_by_track.PSObject.Properties |
      Where-Object { $_.Name -eq $Track } | Select-Object -First 1
    if ($prop) { return [int]$prop.Value }
  } catch { return 0 }
  return 0
}

function Get-Bar {
  param([double]$Pct, [int]$Width)
  $Pct = [Math]::Max(0, [Math]::Min(100, $Pct))
  $filled = [int][Math]::Round(($Pct / 100.0) * $Width)
  $filled = [Math]::Min($filled, $Width)
  $empty  = $Width - $filled
  $bar = ('=' * $filled) + ('.' * $empty)
  return $bar
}

function Get-ActivePhase {
  param([string]$Root)
  $logDir = Join-Path $Root '_logs'
  $log = Get-ChildItem $logDir -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $log) { return 'idle' }
  $tail = Get-Content $log.FullName -Tail 100 -ErrorAction SilentlyContinue
  $cmdLine = ($tail | Where-Object { $_ -like '[cmd]*' } | Select-Object -Last 1)
  if (-not $cmdLine) { return 'idle' }
  if ($cmdLine -match '--task map') { return 'map' }
  if ($cmdLine -match '--task rollout_collect') { return 'recovery' }
  if ($cmdLine -match '--output-root .*[\\/]clean(\s|$)') { return 'clean' }
  if ($cmdLine -match '--output-root .*[\\/]low_noise(\s|$)') { return 'low_noise' }
  if ($cmdLine -match '--output-root .*[\\/]high_noise(\s|$)') { return 'high_noise' }
  return 'collect'
}

function Get-ActiveTrack {
  param([string]$Root)
  $logDir = Join-Path $Root '_logs'
  $log = Get-ChildItem $logDir -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $log) { return '?' }
  $tail = Get-Content $log.FullName -Tail 100 -ErrorAction SilentlyContinue
  $trackLine = ($tail | Where-Object { $_ -like '[track]*' } | Select-Object -Last 1)
  if ($trackLine -match '\[track\]\s+(.+)') { return $Matches[1].Trim() }
  return '?'
}

function Render {
  param([string]$Root)

  $collector = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq 'powershell.exe' -and $_.CommandLine -like '*collect_massive_sim_dataset*' }
  $pyProc = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*collect_target_point_data*' }

  $activePhase = Get-ActivePhase -Root $Root
  $activeTrack = Get-ActiveTrack -Root $Root
  $time        = Get-Date -Format 'HH:mm:ss'
  $statusIcon  = if ($collector) { 'RUNNING' } else { 'STOPPED' }

  $sep = '=' * 72

  Write-Host $sep -ForegroundColor DarkGray
  Write-Host ("  SIM MEGA DATASET COLLECTOR  [{0}]  {1}" -f $statusIcon, $time) -ForegroundColor Cyan
  if ($collector) {
    $phaseName = if ($phaseNames[$activePhase]) { $phaseNames[$activePhase] } else { $activePhase.ToUpper() }
    Write-Host ("  Phase: {0,-12}  Track: {1}" -f $phaseName, $activeTrack) -ForegroundColor Yellow
    if ($pyProc) {
      Write-Host ("  Python PID: {0}" -f ($pyProc | Select-Object -First 1).ProcessId) -ForegroundColor DarkGray
    }
  } else {
    Write-Host "  No collector process found." -ForegroundColor DarkRed
  }
  Write-Host $sep -ForegroundColor DarkGray
  Write-Host ""

  $header = "{0,-36} {1,-32} {2,6}  {3}" -f "TRACK", "OVERALL", "%", "MAP|CLEAN|LOW|HIGH|REC"
  Write-Host $header -ForegroundColor White
  Write-Host ("-" * 90) -ForegroundColor DarkGray

  $totalPhases = 5
  $sumOverall  = 0.0

  foreach ($track in $tracks) {
    $slug        = $track.Replace('-', '_')
    $trackRoot   = Join-Path $Root $slug
    $mapRoot     = Join-Path $trackRoot 'maps'
    $cleanRoot   = Join-Path $trackRoot 'clean'
    $lowRoot     = Join-Path $trackRoot 'low_noise'
    $highRoot    = Join-Path $trackRoot 'high_noise'
    $recoveryRoot= Join-Path $trackRoot 'recovery'

    $mapDone     = [bool](Get-ChildItem $mapRoot -Recurse -Filter metadata.json -File -ErrorAction SilentlyContinue | Select-Object -First 1)
    $mapPct      = if ($mapDone) { 100.0 } else { 0.0 }

    $cleanCount  = Get-JpgCount -Dir $cleanRoot
    $lowCount    = Get-JpgCount -Dir $lowRoot
    $highCount   = Get-JpgCount -Dir $highRoot
    $recovUsable = Get-RecoveryUsable -RecoveryRoot $recoveryRoot -Track $track
    $recovJpg    = Get-JpgCount -Dir $recoveryRoot
    $recovSignal = [Math]::Max($recovUsable, $recovJpg)

    $cleanPct    = [Math]::Min(100.0, 100.0 * $cleanCount   / [Math]::Max(1, $CleanTarget))
    $lowPct      = [Math]::Min(100.0, 100.0 * $lowCount     / [Math]::Max(1, $LowTarget))
    $highPct     = [Math]::Min(100.0, 100.0 * $highCount    / [Math]::Max(1, $HighTarget))
    $recovPct    = [Math]::Min(100.0, 100.0 * $recovSignal  / [Math]::Max(1, $RecoveryTarget))

    $overall     = ($mapPct + $cleanPct + $lowPct + $highPct + $recovPct) / $totalPhases
    $sumOverall  += $overall

    $bar         = Get-Bar -Pct $overall -Width $BarWidth
    $phaseSummary = ("{0,3:N0}|{1,3:N0}|{2,3:N0}|{3,3:N0}|{4,3:N0}" -f $mapPct,$cleanPct,$lowPct,$highPct,$recovPct)

    $isActive    = ($track -eq $activeTrack)
    $color       = if ($overall -ge 100) { 'Green' } elseif ($isActive) { 'Yellow' } else { 'White' }

    $line = ("{0,-36} [{1}] {2,5:N1}%  {3}" -f $slug, $bar, $overall, $phaseSummary)
    Write-Host $line -ForegroundColor $color

    # detail counts line
    $detail = ("  clean={0,6} low={1,6} high={2,6} rec_jpg={3,5} rec_usable={4,5}" -f $cleanCount,$lowCount,$highCount,$recovJpg,$recovUsable)
    Write-Host $detail -ForegroundColor DarkGray
  }

  $globalPct = $sumOverall / [Math]::Max(1, $tracks.Count)
  $gbar      = Get-Bar -Pct $globalPct -Width ($BarWidth + 8)

  Write-Host ""
  Write-Host ("-" * 90) -ForegroundColor DarkGray
  $gline = ("  GLOBAL PROGRESS  [{0}]  {1,5:N1}%" -f $gbar, $globalPct)
  $gcolor = if ($globalPct -ge 100) { 'Green' } elseif ($globalPct -gt 0) { 'Cyan' } else { 'White' }
  Write-Host $gline -ForegroundColor $gcolor
  Write-Host $sep -ForegroundColor DarkGray
  Write-Host ""
  Write-Host "  Targets: clean=$CleanTarget  low=$LowTarget  high=$HighTarget  recovery=$RecoveryTarget" -ForegroundColor DarkGray
  Write-Host ("  Refresh every {0}s  |  Ctrl+C to stop" -f $RefreshSec) -ForegroundColor DarkGray
}

if (-not (Test-Path $Root)) {
  Write-Host "root not found: $Root" -ForegroundColor Red
  exit 1
}

if ($Watch) {
  while ($true) {
    Clear-Host
    Render -Root $Root
    Start-Sleep -Seconds $RefreshSec
  }
} else {
  Render -Root $Root
}
