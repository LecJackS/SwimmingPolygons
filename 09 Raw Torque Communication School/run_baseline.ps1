param(
    [string]$PythonExecutable = (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"),
    [string]$Device = "cuda",
    [string]$PolicyStack = "old",
    [string]$TrainingPhase = "forage_full",
    [string]$WarmstartMotionCheckpoint = "",
    [string]$CheckpointRoot = "",
    [int]$TrainIterations = 200,
    [int]$CheckpointEveryIterations = 20,
    [int]$NumEnvRunners = 8,
    [int]$NumEnvsPerRunner = 2,
    [int]$LightEvalEpisodes = 2,
    [int]$TimeLimit = 300,
    [string]$RewardMode = "forage",
    [int]$HistoryLength = 8,
    [double]$ActivationTimeConstant = 0.12,
    [double]$MotionEpsilonStart = 0.25,
    [double]$MotionEpsilonEnd = 0.0,
    [int]$MotionEpsilonDecayIterations = 60,
    [double]$MessageEpsilon = 0.0,
    [double]$JointPassiveStiffness = 10.0,
    [double]$JointSoftLimitStartRatio = 0.70,
    [double]$JointSoftLimitStiffness = 18.0,
    [double]$JointSoftLimitDamping = 2.0,
    [double]$BodyLinearDrag = 1.0,
    [double]$SwimAssistStartWeight = 0.35,
    [int]$SwimAssistMinIterations = 40,
    [double]$SwimAssistDisableForwardVelocity = 0.03,
    [double]$SwimAssistDisableJointLimitOccupancy = 0.35,
    [double]$SwimAssistDisableNegativeForwardFrac = 0.45,
    [int]$SwimAssistDisableConsecutiveEvals = 2,
    [int]$SwimAssistFadeEvals = 2,
    [double]$Gamma = 0.97,
    [double]$LearningRate = 0.0003,
    [double]$EntropyCoeff = 0.01,
    [int]$TrainBatchSize = 16000,
    [int]$MinibatchSize = 2048,
    [int]$NumEpochs = 6,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$childScript = Join-Path $scriptDir "run_baseline_child.ps1"
$powerShellExe = Join-Path $PSHOME "powershell.exe"
if (-not (Test-Path -LiteralPath $powerShellExe)) {
    $powerShellExe = "powershell.exe"
}

if ([string]::IsNullOrWhiteSpace($CheckpointRoot)) {
    if ($PolicyStack -eq "new" -and $TrainingPhase -eq "locomotion_teacher") {
        $runStem = "baseline_v9_newstack_locomotion_teacher"
        $checkpointRoot = Join-Path $scriptDir "rllib_checkpoints_v9_newstack_locomotion_teacher"
    } elseif ($PolicyStack -eq "new" -and ($TrainingPhase -eq "locomotion_propulsion_easy")) {
        $runStem = "baseline_v9_newstack_locomotion_propulsion_easy_limitpressure"
        $checkpointRoot = Join-Path $scriptDir "rllib_checkpoints_v9_newstack_locomotion_propulsion_easy_limitpressure"
    } elseif ($PolicyStack -eq "new" -and ($TrainingPhase -eq "locomotion_propulsion_robust")) {
        $runStem = "baseline_v9_newstack_locomotion_propulsion_robust_limitpressure"
        $checkpointRoot = Join-Path $scriptDir "rllib_checkpoints_v9_newstack_locomotion_propulsion_robust_limitpressure"
    } elseif ($PolicyStack -eq "new" -and ($TrainingPhase -eq "locomotion_self" -or $TrainingPhase -eq "locomotion_only")) {
        $runStem = "baseline_v9_newstack_locomotion_self"
        $checkpointRoot = Join-Path $scriptDir "rllib_checkpoints_v9_newstack_locomotion_self"
    } elseif ($PolicyStack -eq "new") {
        $runStem = "baseline_v9_newstack_forage_warmstart_limitpressure"
        $checkpointRoot = Join-Path $scriptDir "rllib_checkpoints_v9_newstack_forage_warmstart_limitpressure"
    } else {
        $runStem = "baseline_v9_muscle_activation_comm"
        $checkpointRoot = Join-Path $scriptDir "rllib_checkpoints_baseline_v9_muscle_activation_comm"
    }
} else {
    if ($PolicyStack -eq "new" -and $TrainingPhase -eq "locomotion_teacher") {
        $runStem = "baseline_v9_newstack_locomotion_teacher"
    } elseif ($PolicyStack -eq "new" -and ($TrainingPhase -eq "locomotion_propulsion_easy")) {
        $runStem = "baseline_v9_newstack_locomotion_propulsion_easy_limitpressure"
    } elseif ($PolicyStack -eq "new" -and ($TrainingPhase -eq "locomotion_propulsion_robust")) {
        $runStem = "baseline_v9_newstack_locomotion_propulsion_robust_limitpressure"
    } elseif ($PolicyStack -eq "new" -and ($TrainingPhase -eq "locomotion_self" -or $TrainingPhase -eq "locomotion_only")) {
        $runStem = "baseline_v9_newstack_locomotion_self"
    } elseif ($PolicyStack -eq "new") {
        $runStem = "baseline_v9_newstack_forage_warmstart_limitpressure"
    } else {
        $runStem = "baseline_v9_muscle_activation_comm"
    }
    $checkpointRoot = $CheckpointRoot
}

$pidPath = Join-Path $scriptDir ("{0}.pid" -f $runStem)
$stdoutPath = Join-Path $scriptDir ("{0}.out.log" -f $runStem)
$stderrPath = Join-Path $scriptDir ("{0}.err.log" -f $runStem)

$canonicalTrainingPhase = if ($TrainingPhase -eq "locomotion_only") { "locomotion_self" } else { $TrainingPhase }
if (-not $PSBoundParameters.ContainsKey("MotionEpsilonStart")) {
    $MotionEpsilonStart = if ($canonicalTrainingPhase -eq "forage_full") { 0.25 } else { 0.0 }
}
if (-not $PSBoundParameters.ContainsKey("MotionEpsilonEnd")) {
    $MotionEpsilonEnd = if ($canonicalTrainingPhase -eq "forage_full") { 0.0 } else { 0.0 }
}
if (-not $PSBoundParameters.ContainsKey("MotionEpsilonDecayIterations")) {
    $MotionEpsilonDecayIterations = if ($canonicalTrainingPhase -eq "forage_full") { 60 } else { 1 }
}

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Python executable not found: $PythonExecutable"
}
if (-not (Test-Path -LiteralPath $childScript)) {
    throw "Child launcher not found: $childScript"
}

if (Test-Path -LiteralPath $pidPath) {
    $existingPidText = Get-Content -LiteralPath $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existingPidText) {
        $existingPid = [int]$existingPidText
        $existingProcess = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($existingProcess) {
            throw "A baseline run is already active with PID $existingPid."
        }
    }
    Remove-Item -LiteralPath $pidPath -Force
}

foreach ($pathToClear in @($stdoutPath, $stderrPath)) {
    if (Test-Path -LiteralPath $pathToClear) {
        Remove-Item -LiteralPath $pathToClear -Force
    }
}

$childInvocationArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $childScript,
    "-PythonExecutable", $PythonExecutable,
    "-Device", $Device,
    "-PolicyStack", $PolicyStack,
    "-TrainingPhase", $TrainingPhase,
    "-TrainIterations", "$TrainIterations",
    "-CheckpointEveryIterations", "$CheckpointEveryIterations",
    "-NumEnvRunners", "$NumEnvRunners",
    "-NumEnvsPerRunner", "$NumEnvsPerRunner",
    "-LightEvalEpisodes", "$LightEvalEpisodes",
    "-TimeLimit", "$TimeLimit",
    "-RewardMode", $RewardMode,
    "-HistoryLength", "$HistoryLength",
    "-ActivationTimeConstant", "$ActivationTimeConstant",
    "-MotionEpsilonStart", "$MotionEpsilonStart",
    "-MotionEpsilonEnd", "$MotionEpsilonEnd",
    "-MotionEpsilonDecayIterations", "$MotionEpsilonDecayIterations",
    "-MessageEpsilon", "$MessageEpsilon",
    "-JointPassiveStiffness", "$JointPassiveStiffness",
    "-JointSoftLimitStartRatio", "$JointSoftLimitStartRatio",
    "-JointSoftLimitStiffness", "$JointSoftLimitStiffness",
    "-JointSoftLimitDamping", "$JointSoftLimitDamping",
    "-BodyLinearDrag", "$BodyLinearDrag",
    "-SwimAssistStartWeight", "$SwimAssistStartWeight",
    "-SwimAssistMinIterations", "$SwimAssistMinIterations",
    "-SwimAssistDisableForwardVelocity", "$SwimAssistDisableForwardVelocity",
    "-SwimAssistDisableJointLimitOccupancy", "$SwimAssistDisableJointLimitOccupancy",
    "-SwimAssistDisableNegativeForwardFrac", "$SwimAssistDisableNegativeForwardFrac",
    "-SwimAssistDisableConsecutiveEvals", "$SwimAssistDisableConsecutiveEvals",
    "-SwimAssistFadeEvals", "$SwimAssistFadeEvals",
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
if (-not [string]::IsNullOrWhiteSpace($WarmstartMotionCheckpoint)) {
    $childInvocationArgs += @("-WarmstartMotionCheckpoint", $WarmstartMotionCheckpoint)
}

function Quote-PowerShellArg {
    param([string]$Text)
    if ($null -eq $Text) {
        return '""'
    }
    if ($Text.Contains(" ") -or $Text.Contains('"')) {
        return '"' + $Text.Replace('"', '\"') + '"'
    }
    return $Text
}

$childStartProcessArgs = $childInvocationArgs | ForEach-Object { Quote-PowerShellArg ([string]$_) }

function Show-LaunchInfo {
    param([string]$Mode, [string]$PidText)
    if ($PidText) {
        Write-Host "PID=$PidText"
    }
    Write-Host "MODE=$Mode"
    Write-Host "CHILD=$childScript"
    Write-Host "POWERSHELL=$powerShellExe"
    Write-Host "PYTHON=$PythonExecutable"
    Write-Host "STDOUT=$stdoutPath"
    Write-Host "STDERR=$stderrPath"
    Write-Host "CHECKPOINT_ROOT=$checkpointRoot"
}

if ($Foreground) {
    Show-LaunchInfo -Mode "foreground" -PidText ""
    & $powerShellExe @childInvocationArgs
    $exitCode = $LASTEXITCODE
    Write-Host "EXIT_CODE=$exitCode"
    exit $exitCode
}

$process = Start-Process `
    -FilePath $powerShellExe `
    -ArgumentList $childStartProcessArgs `
    -WorkingDirectory $scriptDir `
    -WindowStyle Minimized `
    -PassThru

Start-Sleep -Seconds 8
$activeProcess = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
if (-not $activeProcess) {
    if (Test-Path -LiteralPath $pidPath) {
        Remove-Item -LiteralPath $pidPath -Force
    }
    Write-Host "Detached child exited before health check."
    if (Test-Path -LiteralPath $stderrPath) {
        Write-Host "STDERR tail:"
        Get-Content -LiteralPath $stderrPath -Tail 40
    }
    throw "Baseline child exited before the initial health check."
}

Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Show-LaunchInfo -Mode "detached" -PidText "$($process.Id)"
