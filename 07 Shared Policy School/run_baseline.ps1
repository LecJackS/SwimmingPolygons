param(
    [string]$PythonExecutable = (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"),
    [string]$Device = "cuda",
    [int]$TrainIterations = 100,
    [int]$CheckpointEveryIterations = 5,
    [int]$NumEnvRunners = 4,
    [int]$NumEnvsPerRunner = 2,
    [int]$EvalReportEpisodes = 10
)

$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$checkpointRoot = Join-Path $scriptDir "rllib_checkpoints_baseline_v7_school"
$pidPath = Join-Path $scriptDir "baseline_v7_school.pid"
$stdoutPath = Join-Path $scriptDir "baseline_v7_school.out.log"
$stderrPath = Join-Path $scriptDir "baseline_v7_school.err.log"

if (-not (Test-Path $PythonExecutable)) {
    throw "Python executable not found: $PythonExecutable"
}

if (Test-Path $pidPath) {
    $existingPidText = (Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($existingPidText) {
        $existingPid = [int]$existingPidText
        $existingProcess = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($existingProcess) {
            throw "A baseline run is already active with PID $existingPid."
        }
    }
    Remove-Item $pidPath -Force
}

foreach ($path in @($stdoutPath, $stderrPath)) {
    if (Test-Path $path) {
        Remove-Item $path -Force
    }
}

$argumentList = @(
    "agent.py",
    "--device", $Device,
    "--train-iterations", $TrainIterations,
    "--checkpoint-every-iterations", $CheckpointEveryIterations,
    "--num-env-runners", $NumEnvRunners,
    "--num-envs-per-runner", $NumEnvsPerRunner,
    "--eval-report-episodes", $EvalReportEpisodes,
    "--checkpoint-root", $checkpointRoot
)

$process = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList $argumentList `
    -WorkingDirectory $scriptDir `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

Set-Content -Path $pidPath -Value $process.Id -Encoding ascii

Write-Host "PID=$($process.Id)"
Write-Host "STDOUT=$stdoutPath"
Write-Host "STDERR=$stderrPath"
Write-Host "CHECKPOINT_ROOT=$checkpointRoot"
