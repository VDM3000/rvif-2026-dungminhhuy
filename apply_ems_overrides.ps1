$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
& $Python (Join-Path $ProjectRoot "tools\apply_ems_overrides.py")
if ($LASTEXITCODE -ne 0) { throw "Applying EMS compatibility overrides failed" }
