param(
    [string]$PythonExecutable,
    [string]$Device,
    [string]$PolicyStack,
    [string]$TrainingPhase,
    [string]$WarmstartMotionCheckpoint,
    [int]$TrainIterations,
    [int]$CheckpointEveryIterations,
    [int]$NumEnvRunners,
    [int]$NumEnvsPerRunner,
    [int]$LightEvalEpisodes,
    [int]$TimeLimit,
    [string]$RewardMode,
    [int]$HistoryLength,
    [double]$ActivationTimeConstant,
    [double]$MotionEpsilonStart,
    [double]$MotionEpsilonEnd,
    [int]$MotionEpsilonDecayIterations,
    [double]$MessageEpsilon,
    [double]$JointPassiveStiffness,
    [double]$JointSoftLimitStartRatio,
    [double]$JointSoftLimitStiffness,
    [double]$JointSoftLimitDamping,
    [double]$BodyLinearDrag,
    [double]$SwimAssistStartWeight,
    [int]$SwimAssistMinIterations,
    [double]$SwimAssistDisableForwardVelocity,
    [double]$SwimAssistDisableJointLimitOccupancy,
    [double]$SwimAssistDisableNegativeForwardFrac,
    [int]$SwimAssistDisableConsecutiveEvals,
    [int]$SwimAssistFadeEvals,
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
$cmdExe = $env:ComSpec
if ([string]::IsNullOrWhiteSpace($cmdExe)) {
    $cmdExe = "cmd.exe"
}

$header = "[{0}] baseline_child_started python={1} device={2} policy_stack={3} training_phase={4} iterations={5} time_limit={6} reward_mode={7}" -f (Get-Date -Format s), $PythonExecutable, $Device, $PolicyStack, $TrainingPhase, $TrainIterations, $TimeLimit, $RewardMode
Add-Content -LiteralPath $StdoutPath -Value $header -Encoding ascii

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
    "--policy-stack", $PolicyStack,
    "--training-phase", $TrainingPhase,
    "--train-iterations", "$TrainIterations",
    "--checkpoint-every-iterations", "$CheckpointEveryIterations",
    "--num-env-runners", "$NumEnvRunners",
    "--num-envs-per-runner", "$NumEnvsPerRunner",
    "--light-eval-episodes", "$LightEvalEpisodes",
    "--time-limit", "$TimeLimit",
    "--reward-mode", $RewardMode,
    "--history-length", "$HistoryLength",
    "--activation-time-constant", "$ActivationTimeConstant",
    "--motion-epsilon-start", "$MotionEpsilonStart",
    "--motion-epsilon-end", "$MotionEpsilonEnd",
    "--motion-epsilon-decay-iterations", "$MotionEpsilonDecayIterations",
    "--message-epsilon", "$MessageEpsilon",
    "--joint-passive-stiffness", "$JointPassiveStiffness",
    "--joint-soft-limit-start-ratio", "$JointSoftLimitStartRatio",
    "--joint-soft-limit-stiffness", "$JointSoftLimitStiffness",
    "--joint-soft-limit-damping", "$JointSoftLimitDamping",
    "--body-linear-drag", "$BodyLinearDrag",
    "--swim-assist-start-weight", "$SwimAssistStartWeight",
    "--swim-assist-min-iterations", "$SwimAssistMinIterations",
    "--swim-assist-disable-forward-velocity", "$SwimAssistDisableForwardVelocity",
    "--swim-assist-disable-joint-limit-occupancy", "$SwimAssistDisableJointLimitOccupancy",
    "--swim-assist-disable-negative-forward-frac", "$SwimAssistDisableNegativeForwardFrac",
    "--swim-assist-disable-consecutive-evals", "$SwimAssistDisableConsecutiveEvals",
    "--swim-assist-fade-evals", "$SwimAssistFadeEvals",
    "--gamma", "$Gamma",
    "--learning-rate", "$LearningRate",
    "--entropy-coeff", "$EntropyCoeff",
    "--train-batch-size", "$TrainBatchSize",
    "--minibatch-size", "$MinibatchSize",
    "--num-epochs", "$NumEpochs",
    "--checkpoint-root", $CheckpointRoot
)
if (-not [string]::IsNullOrWhiteSpace($WarmstartMotionCheckpoint)) {
    $pythonArgs += @("--warmstart-motion-checkpoint", $WarmstartMotionCheckpoint)
}
$pythonArgumentString = ($pythonArgs | ForEach-Object { Quote-Arg ([string]$_) }) -join " "
$pythonCommand = ('"{0}" {1} 1>> "{2}" 2>> "{3}"' -f $PythonExecutable, $pythonArgumentString, $StdoutPath, $StderrPath)

$exitCode = 1
try {
    & $cmdExe /d /c $pythonCommand
    $exitCode = $LASTEXITCODE
} catch {
    $errorText = ($_ | Out-String).Trim()
    Add-Content -LiteralPath $StderrPath -Value $errorText -Encoding ascii
    $exitCode = 1
}

$footer = "[{0}] baseline_child_finished exit_code={1}" -f (Get-Date -Format s), $exitCode
if ($exitCode -eq 0) {
    Add-Content -LiteralPath $StdoutPath -Value $footer -Encoding ascii
} else {
    Add-Content -LiteralPath $StderrPath -Value $footer -Encoding ascii
}
exit $exitCode
