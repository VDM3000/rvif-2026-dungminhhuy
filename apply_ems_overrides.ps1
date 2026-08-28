$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EmsRepo = Join-Path $ProjectRoot "EMS_repo"
$Overrides = Join-Path $ProjectRoot "ems_overrides"

if (-not (Test-Path -LiteralPath (Join-Path $EmsRepo "MSNet.py"))) {
    throw "EMS submodule is missing. Run: git submodule update --init --recursive"
}

foreach ($name in @("MSNet.py", "train.py", "val.py", "test.py")) {
    Copy-Item -LiteralPath (Join-Path $Overrides $name) -Destination (Join-Path $EmsRepo $name) -Force
}

Write-Host "Applied Windows/category EMS overrides to $EmsRepo"
