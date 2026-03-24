param(
    [string]$PythonExecutable = (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"),
    [string]$ManifestPath = (Join-Path $PSScriptRoot "target_training_manifest_4x_rerun.json"),
    [string]$TargetRoot = (Join-Path $PSScriptRoot "rllib_checkpoints_target_v8_4x_rerun"),
    [string]$Device = "cuda",
    [string]$EvalDevice = "cuda",
    [int]$PollIntervalSeconds = 120
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$childScript = Join-Path $scriptDir "run_target_watchdog_child.ps1"
$pidPath = Join-Path $scriptDir "target_training_watchdog.pid"
$stdoutPath = Join-Path $scriptDir "target_training_watchdog.out.log"
$stderrPath = Join-Path $scriptDir "target_training_watchdog.err.log"

if (-not (Test-Path $PythonExecutable)) { throw "Python executable not found: $PythonExecutable" }
if (-not (Test-Path $childScript)) { throw "Child launcher not found: $childScript" }

if (Test-Path $pidPath) {
    $existingPidText = (Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($existingPidText) {
        $existingPid = [int]$existingPidText
        $existingProcess = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($existingProcess) { throw "A target-training watchdog is already active with PID $existingPid." }
    }
    Remove-Item $pidPath -Force
}

foreach ($path in @($stdoutPath, $stderrPath)) { if (Test-Path $path) { Remove-Item $path -Force } }

function Quote-Arg {
    param([string]$Text)
    if ($null -eq $Text) { return '""' }
    if ($Text.Contains(" ") -or $Text.Contains('"')) { return '"' + $Text.Replace('"', '\"') + '"' }
    return $Text
}

$childArgumentList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $childScript,
    "-PythonExecutable", $PythonExecutable,
    "-ManifestPath", $ManifestPath,
    "-TargetRoot", $TargetRoot,
    "-Device", $Device,
    "-EvalDevice", $EvalDevice,
    "-PollIntervalSeconds", [string]$PollIntervalSeconds,
    "-StdoutPath", $stdoutPath,
    "-StderrPath", $stderrPath
)
$argumentString = ($childArgumentList | ForEach-Object { Quote-Arg ([string]$_) }) -join " "

$process = Start-Process -FilePath "powershell.exe" -ArgumentList $argumentString -WorkingDirectory $scriptDir -WindowStyle Minimized -PassThru
Set-Content -Path $pidPath -Value $process.Id -Encoding ascii
Write-Host "PID=$($process.Id)"
Write-Host "STDOUT=$stdoutPath"
Write-Host "STDERR=$stderrPath"
Write-Host "PIDFILE=$pidPath"
