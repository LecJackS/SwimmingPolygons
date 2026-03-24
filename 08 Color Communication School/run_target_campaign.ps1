param(
    [string]$PythonExecutable = (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"),
    [string]$Device = "cuda",
    [string]$EvalDevice = "cuda",
    [switch]$ForceClean
)

$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$childScript = Join-Path $scriptDir "run_target_campaign_child.ps1"
$pidPath = Join-Path $scriptDir "target_training_campaign.pid"
$stdoutPath = Join-Path $scriptDir "target_training_campaign.out.log"
$stderrPath = Join-Path $scriptDir "target_training_campaign.err.log"

if (-not (Test-Path $PythonExecutable)) {
    throw "Python executable not found: $PythonExecutable"
}
if (-not (Test-Path $childScript)) {
    throw "Child launcher not found: $childScript"
}

if (Test-Path $pidPath) {
    $existingPidText = (Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($existingPidText) {
        $existingPid = [int]$existingPidText
        $existingProcess = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($existingProcess) {
            throw "A target-training campaign is already active with PID $existingPid."
        }
    }
    Remove-Item $pidPath -Force
}

foreach ($path in @($stdoutPath, $stderrPath)) {
    if (Test-Path $path) {
        Remove-Item $path -Force
    }
}

function Quote-Arg {
    param([string]$Text)
    if ($null -eq $Text) {
        return '""'
    }
    if ($Text.Contains(" ") -or $Text.Contains('"')) {
        return '"' + $Text.Replace('"', '\"') + '"'
    }
    return $Text
}

$childArgumentList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $childScript,
    "-PythonExecutable", $PythonExecutable,
    "-Device", $Device,
    "-EvalDevice", $EvalDevice,
    "-StdoutPath", $stdoutPath,
    "-StderrPath", $stderrPath
)
if ($ForceClean.IsPresent) {
    $childArgumentList += "-ForceClean"
}

$argumentString = ($childArgumentList | ForEach-Object {
    Quote-Arg ([string]$_)
}) -join " "

$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $argumentString `
    -WorkingDirectory $scriptDir `
    -WindowStyle Minimized `
    -PassThru

Set-Content -Path $pidPath -Value $process.Id -Encoding ascii

Write-Host "PID=$($process.Id)"
Write-Host "LAUNCHER=$childScript"
Write-Host "STDOUT=$stdoutPath"
Write-Host "STDERR=$stderrPath"
Write-Host "PIDFILE=$pidPath"
