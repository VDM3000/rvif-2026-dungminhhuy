$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$runs = 'outputs/reviewer_experiments'
$baseline = "$runs/baseline_2026-11-22_seed42/improve_behavioral"
$runRoot = "$runs/ems_recipe_211_2026-11-23"
$cache = 'work/ems_211_cache'
$behavior = @('--behavior-features', "$baseline/subject_behavior_features.csv",
              '--feature-sets', "$baseline/feature_sets_clean.json")

function Invoke-Python {
    & $python -u @args
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}

Invoke-Python -c "import torch, timm, cv2, sklearn, matplotlib, openpyxl; assert torch.cuda.is_available(), 'CUDA is required'; print(torch.__version__, torch.cuda.get_device_name(0))"
if (-not (Test-Path "$cache/manifest.json")) {
    if (-not (Test-Path "$cache/fixations_ready")) {
        Invoke-Python EMS_repo/data_process/split_fix.py --dataset_dir EMS
        New-Item -ItemType Directory -Path $cache -Force | Out-Null
        New-Item -ItemType File -Path "$cache/fixations_ready" -Force | Out-Null
    }
    Invoke-Python -m reviewer_experiments.prepare_cache --output $cache
}

foreach ($fold in 'Set_0', 'Set_1', 'Set_2', 'Set_3') {
    Invoke-Python -m reviewer_experiments.train_211 --cache-dir $cache @behavior --output-root "$runRoot/msnet" --eval-fold $fold --ems-checkpoint-selection --require-cuda
}

if (-not (Test-Path "$runRoot/baseline_comparison/summary_metrics.csv")) {
    Invoke-Python -m reviewer_experiments.summarize_211 --workbook EMS/Train_Valid.xlsx --late-root "$runRoot/msnet" --output "$runRoot/baseline_comparison"
}

foreach ($variant in 'fixed', 'joint') {
    foreach ($fold in 'Set_0', 'Set_1', 'Set_2', 'Set_3') {
        if (-not (Test-Path "$runRoot/$variant/$fold/completed.json")) {
            Invoke-Python -m reviewer_experiments.run_211 --msnet-root "$runRoot/msnet" @behavior --output-root $runRoot --variant $variant --eval-fold $fold
        }
    }
}
