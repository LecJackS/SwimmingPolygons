param(
    [string]$PythonExecutable = (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"),
    [string]$Device = "cuda",
    [int]$TrainIterations = 200,
    [int]$CheckpointEveryIterations = 10,
    [int]$NumEnvRunners = 8,
    [int]$NumEnvsPerRunner = 2,
    [int]$LightEvalEpisodes = 4,
    [int]$TimeLimit = 300,
    [double]$Gamma = 0.9,
    [double]$LearningRate = 0.001,
    [double]$EntropyCoeff = 0.01,
    [int]$TrainBatchSize = 8000,
    [int]$MinibatchSize = 1024,
    [int]$NumEpochs = 6
)

$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$childScript = Join-Path $scriptDir "run_baseline_child.ps1"
$checkpointRoot = Join-Path $scriptDir "rllib_checkpoints_baseline_v8_color_comm"
$pidPath = Join-Path $scriptDir "baseline_v8_color_comm.pid"
$stdoutPath = Join-Path $scriptDir "baseline_v8_color_comm.out.log"
$stderrPath = Join-Path $scriptDir "baseline_v8_color_comm.err.log"

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
    "-TrainIterations", "$TrainIterations",
    "-CheckpointEveryIterations", "$CheckpointEveryIterations",
    "-NumEnvRunners", "$NumEnvRunners",
    "-NumEnvsPerRunner", "$NumEnvsPerRunner",
    "-LightEvalEpisodes", "$LightEvalEpisodes",
    "-TimeLimit", "$TimeLimit",
    "-Gamma", "$Gamma",
    "-LearningRate", "$LearningRate",
    "-EntropyCoeff", "$EntropyCoeff",
    "-TrainBatchSize", "$TrainBatchSize",
    "-MinibatchSize", "$MinibatchSize",
    "-NumEpochs", "$NumEpochs",
    "-CheckpointRoot", $checkpointRoot,
    "-StdoutPath", $stdoutPath,
    "-StderrPath", $stderrPath
)

$argumentString = ($childArgumentList | ForEach-Object {
    $text = [string]$_
    Quote-Arg $text
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
Write-Host "CHECKPOINT_ROOT=$checkpointRoot"
