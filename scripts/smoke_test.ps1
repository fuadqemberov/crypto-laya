# End-to-end engineering check on a small real sample. Makes NO claim about predictive quality.
#   powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1 [-Device cuda|cpu|auto]
param([string]$Device = "auto", [string]$Dataset = "smoke")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
function Step($title, [scriptblock]$cmd) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "step failed: $title" }
}
Step "unit tests"            { uv run pytest -q }
Step "hardware"              { uv run python -m laya_crypto.cli hardware }
Step "download sample"       { uv run python -m laya_crypto.cli download --symbols BTCUSDT ETHUSDT SOLUSDT --start 2025-03-01 --end 2025-07-31 --timeframes 5m 1h }
Step "validate"              { uv run python -m laya_crypto.cli validate --timeframe 5m }
Step "dataset"               { uv run python -m laya_crypto.cli dataset --name $Dataset --symbols BTCUSDT ETHUSDT SOLUSDT --force }
Step "baselines"             { uv run python -m laya_crypto.cli baselines --dataset $Dataset }
Step "LAYA smoke fine-tune"  { uv run python -m laya_crypto.cli train --profile smoke --dataset $Dataset --device $Device --overwrite }
Step "select best"           { uv run python -m laya_crypto.cli select-best --dataset $Dataset --profile smoke }
Step "backtest lightgbm"     { uv run python -m laya_crypto.cli backtest --dataset $Dataset --model lightgbm --split test }
Step "comparison report"     { uv run python -m laya_crypto.cli evaluate --dataset $Dataset }
Step "predict (cached data)" { uv run python -m laya_crypto.cli predict --symbol BTCUSDT --offline }
Write-Host "`nSmoke test finished. Reports: reports\$Dataset\comparison.md" -ForegroundColor Green
