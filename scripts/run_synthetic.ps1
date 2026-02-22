param(
    [ValidateSet("single", "core")]
    [string]$Mode = "single",

    [string]$Script = "compare_b_o.py"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Run-One([string]$scriptName) {
    $scriptPath = Join-Path $repoRoot (Join-Path "experiments/synthetic" $scriptName)
    if (!(Test-Path $scriptPath)) {
        throw "Synthetic script not found: $scriptPath"
    }
    Write-Host "Running $scriptName ..."
    python $scriptPath
}

if ($Mode -eq "single") {
    Run-One $Script
}
else {
    $coreScripts = @(
        "compare_b_o.py",
        "diff_eta.py",
        "dist_shift_all.py",
        "dist_shift_high.py",
        "mask.py",
        "vary_dim_loc.py"
    )

    foreach ($s in $coreScripts) {
        Run-One $s
    }
}

Write-Host "Synthetic run finished. Outputs are under outputs/synthetic/."
