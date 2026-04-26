param(
  [string]$Root = 'data/datasets/sim_mega_dataset'
)

$ErrorActionPreference = 'Stop'

Write-Output "[time] $(Get-Date -Format o)"
Write-Output "[root] $Root"

if (-not (Test-Path $Root)) {
  Write-Output '[status] root_not_found'
  exit 0
}

$log = Get-ChildItem (Join-Path $Root '_logs') -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if ($log) {
  Write-Output "[latest_log] $($log.FullName)"
  Get-Content $log.FullName -Tail 25
}

$ps = Get-CimInstance Win32_Process |
  Where-Object { $_.Name -eq 'powershell.exe' -and $_.CommandLine -like '*collect_massive_sim_dataset.ps1*' }

if ($ps) {
  $ps | ForEach-Object { Write-Output ("[collector_ps] pid={0}" -f $_.ProcessId) }
} else {
  Write-Output '[collector_ps] none'
}

$py = Get-CimInstance Win32_Process |
  Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*collect_target_point_data.py*' }

if ($py) {
  $py | ForEach-Object { Write-Output ("[collector_py] pid={0}" -f $_.ProcessId) }
} else {
  Write-Output '[collector_py] none'
}

$tracks = Get-ChildItem $Root -Directory | Where-Object { $_.Name -ne '_logs' }
foreach ($t in $tracks) {
  $jpg = (Get-ChildItem $t.FullName -Recurse -Filter *.jpg -File -ErrorAction SilentlyContinue | Measure-Object).Count
  $json = (Get-ChildItem $t.FullName -Recurse -Filter *.json -File -ErrorAction SilentlyContinue | Measure-Object).Count
  $csv = (Get-ChildItem $t.FullName -Recurse -Filter *.csv -File -ErrorAction SilentlyContinue | Measure-Object).Count
  Write-Output ("[track] {0} jpg={1} json={2} csv={3}" -f $t.Name, $jpg, $json, $csv)
}
