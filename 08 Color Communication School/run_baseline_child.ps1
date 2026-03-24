param(
    [string]$PythonExecutable,
    [string]$Device,
    [int]$TrainIterations,
    [int]$CheckpointEveryIterations,
    [int]$NumEnvRunners,
    [int]$NumEnvsPerRunner,
    [int]$LightEvalEpisodes,
    [int]$TimeLimit,
    [double]$Gamma,
    [double]$LearningRate,
    [double]$EntropyCoeff,
    [int]$TrainBatchSize,
    [int]$MinibatchSize,
    [int]$NumEpochs,
    [string]$CheckpointRoot,
    [string]$StdoutPath,
    [string]$StderrPath
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
$env:PYTHONUNBUFFERED = "1"

$header = "[{0}] baseline_child_started python={1} device={2} iterations={3} time_limit={4}" -f (Get-Date -Format s), $PythonExecutable, $Device, $TrainIterations, $TimeLimit
Add-Content -Path $StdoutPath -Value $header -Encoding utf8

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

$pythonArgs = @(
    "-u",
    "agent.py",
    "--device", $Device,
    "--train-iterations", "$TrainIterations",
    "--checkpoint-every-iterations", "$CheckpointEveryIterations",
    "--num-env-runners", "$NumEnvRunners",
    "--num-envs-per-runner", "$NumEnvsPerRunner",
    "--light-eval-episodes", "$LightEvalEpisodes",
    "--time-limit", "$TimeLimit",
    "--gamma", "$Gamma",
    "--learning-rate", "$LearningRate",
    "--entropy-coeff", "$EntropyCoeff",
    "--train-batch-size", "$TrainBatchSize",
    "--minibatch-size", "$MinibatchSize",
    "--num-epochs", "$NumEpochs",
    "--checkpoint-root", $CheckpointRoot
)
$pythonArgumentString = ($pythonArgs | ForEach-Object { Quote-Arg ([string]$_) }) -join " "

$oldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $process = Start-Process `
        -FilePath $PythonExecutable `
        -ArgumentList $pythonArgumentString `
        -WorkingDirectory $scriptDir `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath
    $exitCode = $process.ExitCode
} finally {
    $ErrorActionPreference = $oldErrorActionPreference
}

$footer = "[{0}] baseline_child_finished exit_code={1}" -f (Get-Date -Format s), $exitCode
if ($exitCode -eq 0) {
    Add-Content -Path $StdoutPath -Value $footer -Encoding utf8
} else {
    Add-Content -Path $StderrPath -Value $footer -Encoding utf8
}
exit $exitCode
