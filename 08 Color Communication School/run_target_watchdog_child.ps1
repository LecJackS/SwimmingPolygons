param(
    [string]$PythonExecutable,
    [string]$ManifestPath,
    [string]$TargetRoot,
    [string]$Device,
    [string]$EvalDevice,
    [int]$PollIntervalSeconds,
    [string]$StdoutPath,
    [string]$StderrPath
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
$env:PYTHONUNBUFFERED = "1"
Add-Content -Path $StdoutPath -Value ("[{0}] target_watchdog_child_started python={1} manifest={2}" -f (Get-Date -Format s), $PythonExecutable, $ManifestPath) -Encoding utf8

function Quote-Arg {
    param([string]$Text)
    if ($null -eq $Text) { return '""' }
    if ($Text.Contains(" ") -or $Text.Contains('"')) { return '"' + $Text.Replace('"', '\"') + '"' }
    return $Text
}

$pythonArgs = @(
    "-u",
    "campaign_watchdog.py",
    "--python-executable", $PythonExecutable,
    "--manifest-path", $ManifestPath,
    "--target-root", $TargetRoot,
    "--device", $Device,
    "--eval-device", $EvalDevice,
    "--poll-interval-seconds", [string]$PollIntervalSeconds
)
$pythonArgumentString = ($pythonArgs | ForEach-Object { Quote-Arg ([string]$_) }) -join " "

$oldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $process = Start-Process -FilePath $PythonExecutable -ArgumentList $pythonArgumentString -WorkingDirectory $scriptDir -NoNewWindow -Wait -PassThru -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
    $exitCode = $process.ExitCode
} finally {
    $ErrorActionPreference = $oldErrorActionPreference
}

$footer = "[{0}] target_watchdog_child_finished exit_code={1}" -f (Get-Date -Format s), $exitCode
if ($exitCode -eq 0) { Add-Content -Path $StdoutPath -Value $footer -Encoding utf8 } else { Add-Content -Path $StderrPath -Value $footer -Encoding utf8 }
exit $exitCode
