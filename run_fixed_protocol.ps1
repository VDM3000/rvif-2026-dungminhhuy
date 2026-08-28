$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Missing virtual environment: $PythonExe. Run setup_windows.ps1 first."
}

& $PythonExe "$ProjectRoot\category_experiments\run_fixed_protocol.py" --category all
if ($LASTEXITCODE -ne 0) { throw "Fixed protocol experiment failed." }
& $PythonExe "$ProjectRoot\category_experiments\validate_fixed_protocol.py"
if ($LASTEXITCODE -ne 0) { throw "Fixed protocol validation failed." }
