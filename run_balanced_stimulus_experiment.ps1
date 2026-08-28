$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Experiment = Join-Path $ProjectRoot "balanced_stimulus_experiment"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Missing virtual environment: $PythonExe"
}

& $PythonExe (Join-Path $Experiment "prepare.py")
if ($LASTEXITCODE -ne 0) { throw "Manifest/cache preparation failed." }
& $PythonExe (Join-Path $Experiment "train_msnet.py") --stimulus-count all --fold all --epochs 50 --batch-size 8 --seed 42
if ($LASTEXITCODE -ne 0) { throw "MSNet training failed." }
& $PythonExe (Join-Path $Experiment "run_fixed_protocol.py")
if ($LASTEXITCODE -ne 0) { throw "Fixed behavioral/fusion evaluation failed." }
& $PythonExe (Join-Path $Experiment "validate.py")
if ($LASTEXITCODE -ne 0) { throw "Experiment validation failed." }
