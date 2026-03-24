param(
    [string]$PythonExecutable,
    [string]$Device,
    [string]$EvalDevice,
    [string]$StdoutPath,
    [string]$StderrPath,
    [switch]$ForceClean
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
$env:PYTHONUNBUFFERED = "1"

$header = "[{0}] target_campaign_child_started python={1} device={2} eval_device={3}" -f (Get-Date -Format s), $PythonExecutable, $Device, $EvalDevice
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
    "train_until_target.py",
    "--device", $Device,
    "--eval-device", $EvalDevice
)
if ($ForceClean.IsPresent) {
    $pythonArgs += "--force-clean"
}

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

$footer = "[{0}] target_campaign_child_finished exit_code={1}" -f (Get-Date -Format s), $exitCode
if ($exitCode -eq 0) {
    Add-Content -Path $StdoutPath -Value $footer -Encoding utf8
} else {
    Add-Content -Path $StderrPath -Value $footer -Encoding utf8
}
exit $exitCode
