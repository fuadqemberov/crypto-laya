# Local dashboard on http://127.0.0.1:8765 (localhost only).
param([int]$Port = 8765)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
Start-Process "http://127.0.0.1:$Port"
uv run python -m laya_crypto.cli dashboard --host 127.0.0.1 --port $Port
