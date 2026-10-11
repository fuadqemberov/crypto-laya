# One command: data -> dataset -> baselines -> pick the best model on validation -> dashboard -> paper bot.
# Every step is safe to re-run: finished steps are skipped or only fetch what is new.
#   powershell -ExecutionPolicy Bypass -File scripts\start_bot.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\start_bot.ps1 -Symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT -Start 2023-01-01 -Dataset laptop4
# Paper trading only: simulated account, live public data, no orders, no API keys.
param(
    [string]$Dataset = "laptop",
    [string[]]$Symbols = @("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    [string]$Start = "2024-01-01"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
function Step($title, [scriptblock]$cmd) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "step failed: $title" }
}

Step "1/5 download market data (only new files)" {
    uv run python -m laya_crypto.cli download --symbols @Symbols --start $Start --timeframes 5m 1h
}
if (Test-Path "data\datasets\$Dataset\manifest.json") {
    Write-Host "`n=== 2/5 dataset '$Dataset' already exists, reusing it ===" -ForegroundColor Cyan
} else {
    Step "2/5 build dataset '$Dataset'" { uv run python -m laya_crypto.cli dataset --name $Dataset --symbols @Symbols }
}
if (Test-Path "reports\$Dataset\lightgbm.json") {
    Write-Host "`n=== 3/5 baselines already trained, reusing them ===" -ForegroundColor Cyan
} else {
    Step "3/5 train and compare models" { uv run python -m laya_crypto.cli baselines --dataset $Dataset }
}
Step "4/5 pick the best model on validation and configure the bot" {
    uv run python -m laya_crypto.cli paper setup --dataset $Dataset
}
Write-Host "`n=== 5/5 dashboard + paper bot ===" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; `$env:Path = '$env:USERPROFILE\.local\bin;' + `$env:Path; uv run python -m laya_crypto.cli dashboard"
Start-Sleep -Seconds 8
Start-Process "http://127.0.0.1:8765"
Write-Host "Paper bot running (simulated money). Stop: Ctrl+C here, or 'Stop bot' in the dashboard's Paper tab." -ForegroundColor Green
uv run python -m laya_crypto.cli paper run
