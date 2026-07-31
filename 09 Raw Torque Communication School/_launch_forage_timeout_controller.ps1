$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$controller = Join-Path $scriptDir "train_until_forage_timeout_curriculum.py"
$manifest = Join-Path $scriptDir "forage_timeout_curriculum_manifest.json"
$targetRoot = Join-Path $scriptDir "rllib_checkpoints_v9_forage_timeout_curriculum"
$stdoutLog = Join-Path $scriptDir "forage_timeout_curriculum_controller.out.log"
$stderrLog = Join-Path $scriptDir "forage_timeout_curriculum_controller.err.log"
$pidFile = Join-Path $scriptDir "forage_timeout_curriculum_controller.pid"

$env:MPLBACKEND = 'Agg'

$argList = @(
    '-u',
    $controller,
    '--manifest-path', $manifest,
    '--target-root', $targetRoot,
    '--device', 'cuda',
    '--eval-device', 'cuda',
    '--max-wall-clock-hours', '12',
    '--resume-existing'
)

$proc = Start-Process -FilePath $python -ArgumentList $argList -WorkingDirectory $scriptDir -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
Set-Content -Path $pidFile -Value $proc.Id -Encoding ascii
Write-Output "launched_controller_pid=$($proc.Id)"
