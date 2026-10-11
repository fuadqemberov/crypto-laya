# Paper-trading bot (simulated account, live public data, no orders, no API keys).
#   powershell -ExecutionPolicy Bypass -File scripts\paper_bot.ps1 [-Name default]
param([string]$Name = "default")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
uv run python -m laya_crypto.cli paper run --name $Name
