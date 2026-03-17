param(
    [string]$ProjectRoot = "E:\HetGAT\HetGAT-HRL",
    [string]$ConfigPath = "E:\HetGAT\HetGAT-HRL\code_and_models\configs\calibrated_short_v1.yaml",
    [int]$Iterations = 200,
    [int]$EpisodesPerIter = 2,
    [int]$Seed = 0,
    [int]$CurriculumWarmup = 50
)

$ErrorActionPreference = "Stop"

$CodeRoot = Join-Path $ProjectRoot "code_and_models"
$TrainScript = Join-Path $CodeRoot "tools\train_true_ppo_allocator.py"
$EvalScript = Join-Path $CodeRoot "tools\eval_zero_shot_L.py"
$ResultsRoot = Join-Path $ProjectRoot "training_results"
$env:PYTHONPATH = $CodeRoot

function Get-LatestRunDirByPrefix {
    param([string]$Prefix)
    if (-not (Test-Path $ResultsRoot)) { return $null }
    $dirs = Get-ChildItem -Path $ResultsRoot -Directory | Where-Object { $_.Name -like "$Prefix*" } | Sort-Object LastWriteTime -Descending
    foreach ($d in $dirs) {
        $metrics = Join-Path $d.FullName "metrics.csv"
        $ckpt = Join-Path $d.FullName "allocator_final.pt"
        if ((Test-Path $metrics) -and (Test-Path $ckpt)) { return $d.FullName }
    }
    return $null
}

function Run-Train {
    param(
        [string]$RunName,
        [string]$ExtraArgs = "",
        [bool]$ReuseExisting = $false
    )
    if ($ReuseExisting) {
        $existing = Get-LatestRunDirByPrefix -Prefix $RunName
        if ($existing -ne $null) {
            Write-Host "[Reuse] $RunName -> $existing"
            return $existing
        }
    }

    $cmd = @(
        "python", $TrainScript,
        "--config", $ConfigPath,
        "--iterations", $Iterations,
        "--episodes-per-iter", $EpisodesPerIter,
        "--seed", $Seed,
        "--curriculum-warmup-iters", $CurriculumWarmup,
        "--run-name", $RunName
    )
    if ($ExtraArgs.Trim().Length -gt 0) {
        $cmd += $ExtraArgs.Split(" ")
    }
    Write-Host "[Run] $($cmd -join ' ')"
    # Keep logs visible, but do not leak them into function return value.
    & $cmd[0] $cmd[1..($cmd.Length-1)] | Out-Host

    $outDir = Get-LatestRunDirByPrefix -Prefix $RunName
    if ($outDir -eq $null) {
        throw "Failed to resolve output dir for run: $RunName"
    }
    return $outDir
}

function Eval-Checkpoint {
    param(
        [string]$Checkpoint,
        [string]$Label,
        [bool]$UseHetGat,
        [bool]$EnableRthMask,
        [string]$OutCsv
    )
    $cmd = @(
        "python", $EvalScript,
        "--config", $ConfigPath,
        "--checkpoints", $Checkpoint,
        "--labels", $Label,
        "--episodes", "10",
        "--scenario", "C",
        "--seed", $Seed,
        "--output-csv", $OutCsv,
        "--use-hetgat", $UseHetGat.ToString().ToLower(),
        "--enable-rth-mask", $EnableRthMask.ToString().ToLower()
    )
    Write-Host "[Eval] $($cmd -join ' ')"
    & $cmd[0] $cmd[1..($cmd.Length-1)]
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$bundleDir = Join-Path $ResultsRoot "nature_bundle_$stamp"
New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null

# RUN 0 (Perfect): try reusing existing canonical i200 run first.
$run0Dir = Get-LatestRunDirByPrefix -Prefix "ll_ppo_curriculum_i200"
if ($run0Dir -eq $null) {
    $run0Dir = Run-Train -RunName "nature_run0_perfect_i200" -ReuseExisting:$true
} else {
    Write-Host "[Reuse] RUN0_Perfect from existing ll_ppo_curriculum_i200 -> $run0Dir"
}

# RUN 1 (Baseline_MAPPO): --use-hetgat False
$run1Dir = Run-Train -RunName "nature_run1_baseline_mappo_i200" -ExtraArgs "--use-hetgat false" -ReuseExisting:$false

# RUN 2 (Ablation_No_RTH): --enable-rth-mask False
$run2Dir = Run-Train -RunName "nature_run2_ablation_no_rth_i200" -ExtraArgs "--enable-rth-mask false" -ReuseExisting:$false

# RUN 3 (Ablation_No_Curr): --curriculum-warmup-iters 0
$cmd3 = @(
    "python", $TrainScript,
    "--config", $ConfigPath,
    "--iterations", $Iterations,
    "--episodes-per-iter", $EpisodesPerIter,
    "--seed", $Seed,
    "--curriculum-warmup-iters", "0",
    "--run-name", "nature_run3_ablation_no_curr_i200"
)
Write-Host "[Run] $($cmd3 -join ' ')"
& $cmd3[0] $cmd3[1..($cmd3.Length-1)] | Out-Host
$run3Dir = Get-LatestRunDirByPrefix -Prefix "nature_run3_ablation_no_curr_i200"
if ($run3Dir -eq $null) { throw "Failed to resolve RUN3 output dir." }

$run0Ckpt = Join-Path $run0Dir "allocator_final.pt"
$run1Ckpt = Join-Path $run1Dir "allocator_final.pt"
$run2Ckpt = Join-Path $run2Dir "allocator_final.pt"
$run3Ckpt = Join-Path $run3Dir "allocator_final.pt"

# Zero-shot L evaluation for RUN0 and RUN1 (10 episodes each), merged as L_scale_metrics.csv
$tmp0 = Join-Path $bundleDir "L_scale_run0_tmp.csv"
$tmp1 = Join-Path $bundleDir "L_scale_run1_tmp.csv"
$lscaleCsv = Join-Path $bundleDir "L_scale_metrics.csv"

Eval-Checkpoint -Checkpoint $run0Ckpt -Label "RUN0_Perfect" -UseHetGat $true -EnableRthMask $true -OutCsv $tmp0
Eval-Checkpoint -Checkpoint $run1Ckpt -Label "RUN1_Baseline_MAPPO" -UseHetGat $false -EnableRthMask $true -OutCsv $tmp1

$rows = @()
if (Test-Path $tmp0) { $rows += Import-Csv $tmp0 }
if (Test-Path $tmp1) { $rows += Import-Csv $tmp1 }
$rows | Export-Csv -Path $lscaleCsv -NoTypeInformation -Encoding UTF8

$manifest = [ordered]@{
    bundle_dir = $bundleDir
    generated_at = (Get-Date).ToString("s")
    config = $ConfigPath
    runs = [ordered]@{
        RUN0_Perfect = [ordered]@{
            run_dir = $run0Dir
            checkpoint = $run0Ckpt
            metrics_csv = (Join-Path $run0Dir "metrics.csv")
            use_hetgat = $true
            enable_rth_mask = $true
            curriculum_warmup_iters = $CurriculumWarmup
        }
        RUN1_Baseline_MAPPO = [ordered]@{
            run_dir = $run1Dir
            checkpoint = $run1Ckpt
            metrics_csv = (Join-Path $run1Dir "metrics.csv")
            use_hetgat = $false
            enable_rth_mask = $true
            curriculum_warmup_iters = $CurriculumWarmup
        }
        RUN2_Ablation_No_RTH = [ordered]@{
            run_dir = $run2Dir
            checkpoint = $run2Ckpt
            metrics_csv = (Join-Path $run2Dir "metrics.csv")
            use_hetgat = $true
            enable_rth_mask = $false
            curriculum_warmup_iters = $CurriculumWarmup
        }
        RUN3_Ablation_No_Curr = [ordered]@{
            run_dir = $run3Dir
            checkpoint = $run3Ckpt
            metrics_csv = (Join-Path $run3Dir "metrics.csv")
            use_hetgat = $true
            enable_rth_mask = $true
            curriculum_warmup_iters = 0
        }
    }
    l_scale_metrics_csv = $lscaleCsv
}

$manifestPath = Join-Path $bundleDir "nature_run_manifest.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host ""
Write-Host "======================================================="
Write-Host "Nature experiment bundle created:"
Write-Host "  $bundleDir"
Write-Host "Manifest:"
Write-Host "  $manifestPath"
Write-Host "L-scale metrics:"
Write-Host "  $lscaleCsv"
Write-Host "======================================================="
