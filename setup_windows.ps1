[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating virtual environment at $VenvDir"
    python -m venv $VenvDir
}

if (-not $SkipInstall) {
    & $VenvPython -m pip install --upgrade pip setuptools wheel

    if ($CpuOnly) {
        & $VenvPython -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
    }
    else {
        & $VenvPython -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    }

    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-windows.txt")
}

& $VenvPython (Join-Path $ProjectRoot "tools\prepare_local_notebook.py")
& $VenvPython -m ipykernel install --sys-prefix --name rvif-ems --display-name "RVIF EMS (Windows GPU)"
& $VenvPython (Join-Path $ProjectRoot "tools\verify_local_setup.py")

Write-Host ""
Write-Host "Setup complete. Start Jupyter with:"
Write-Host "  .\.venv\Scripts\python.exe -m jupyter lab"
Write-Host "Then open notebook_windows.ipynb and select 'RVIF EMS (Windows GPU)'."
