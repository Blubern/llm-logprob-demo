$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$utf8Encoding = [System.Text.UTF8Encoding]::new()
[Console]::InputEncoding = $utf8Encoding
[Console]::OutputEncoding = $utf8Encoding
$OutputEncoding = $utf8Encoding
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$appFile = Join-Path $projectRoot 'logprob_demo.py'
$envFile = Join-Path $projectRoot '.env'
$requirementsFile = Join-Path $projectRoot 'requirements.txt'
$venvDir = Join-Path $projectRoot '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

if (-not (Test-Path $appFile)) {
    throw "App file not found: $appFile"
}

function Find-PythonInterpreter {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return $pythonCommand.Source
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        return $pyLauncher.Source
    }

    return $null
}

function Invoke-Python {
    param(
        [string]$PythonExe,
        [string[]]$Arguments,
        [switch]$AllowFailure,
        [switch]$Quiet
    )

    $commandArguments = if ([System.IO.Path]::GetFileName($PythonExe).Equals('py.exe', [System.StringComparison]::OrdinalIgnoreCase)) {
        @('-3') + $Arguments
    }
    else {
        $Arguments
    }

    if ($Quiet) {
        & $PythonExe @commandArguments *> $null
    }
    else {
        & $PythonExe @commandArguments
    }

    $exitCode = $LASTEXITCODE
    if (-not $AllowFailure -and $exitCode -ne 0) {
        throw "Python command failed with exit code ${exitCode}: $($Arguments -join ' ')"
    }

    return $exitCode
}

if (-not (Test-Path $envFile)) {
    throw "No .env file found in the project root: $envFile"
}

$pythonExe = if (Test-Path $venvPython) {
    $venvPython
}
else {
    $bootstrapPython = Find-PythonInterpreter
    if (-not $bootstrapPython) {
        throw 'No Python installation found. Install Python 3.10+ first.'
    }

    Write-Host "Creating virtual environment in: $venvDir"
    [void](Invoke-Python -PythonExe $bootstrapPython -Arguments @('-m', 'venv', $venvDir))

    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment creation did not produce: $venvPython"
    }

    $venvPython
}

Write-Host "Using Python: $pythonExe"
Write-Host "Project root: $projectRoot"

if (Test-Path $requirementsFile) {
    Write-Host 'Requirements file found:' $requirementsFile

    $streamlitAvailable = (Invoke-Python -PythonExe $pythonExe -Arguments @('-c', 'import streamlit') -AllowFailure -Quiet) -eq 0
    if (-not $streamlitAvailable) {
        Write-Host 'Installing dependencies from requirements.txt...'
        [void](Invoke-Python -PythonExe $pythonExe -Arguments @('-m', 'pip', 'install', '-r', $requirementsFile))
    }
}

Push-Location $projectRoot
try {
    [void](Invoke-Python -PythonExe $pythonExe -Arguments (@('-m', 'streamlit', 'run', $appFile) + $args))
}
finally {
    Pop-Location
}