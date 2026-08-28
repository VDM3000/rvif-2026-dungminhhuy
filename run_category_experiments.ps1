param(
    [ValidateSet("all", "manipulated", "natural", "social", "synthetic")]
    [string]$Category = "all",
    [int]$Epochs = 50,
    [int]$BatchSize = 8,
    [int]$BootstrapIterations = 5000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Missing virtual environment: $PythonExe. Run setup_windows.ps1 first."
}

& $PythonExe "$ProjectRoot\category_experiments\prepare_cache.py" --category $Category
& $PythonExe "$ProjectRoot\category_experiments\train_msnet.py" --category $Category --fold all --epochs $Epochs --batch-size $BatchSize
& $PythonExe "$ProjectRoot\category_experiments\run_behavior_fusion.py" --category $Category

if ($Category -eq "all") {
    & $PythonExe "$ProjectRoot\category_experiments\analyze_results.py" --bootstrap-iterations $BootstrapIterations
}
