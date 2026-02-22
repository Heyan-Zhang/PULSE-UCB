param(
    [string]$Dataset = "taobao",
    [string]$DataRoot = "data",
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$env:NCB_DATASET = $Dataset
$env:NCB_DATASET_PATH = Join-Path (Join-Path $repoRoot $DataRoot) $Dataset
$env:NCB_BASE_DATA_PATH = Join-Path $env:NCB_DATASET_PATH "preprocess"

Write-Host "Dataset: $env:NCB_DATASET"
Write-Host "Dataset path: $env:NCB_DATASET_PATH"
Write-Host "Preprocess path: $env:NCB_BASE_DATA_PATH"

python experiments/real/preprocess.py
python experiments/real/autoencoder_train.py --seed $Seed
python experiments/real/generate_embeddings.py
python experiments/real/prepare_final_data.py
python experiments/real/pretrain_inference_model.py
python experiments/real/run_online_evaluation.py

Write-Host "Pipeline finished. Outputs in outputs/real/."
