# One-time setup on Windows: uv, Python 3.12 environment (CUDA PyTorch), hardware report, base LAYA checkpoint.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python package manager) ..."
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv --version

Write-Host "Creating .venv with Python 3.12 and installing locked dependencies (CUDA 12.6 PyTorch on Windows) ..."
uv python install 3.12
uv sync --python 3.12 --extra dev
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

Write-Host "Inspecting hardware ..."
uv run python -m laya_crypto.cli hardware
if ($LASTEXITCODE -ne 0) { throw "hardware inspection failed" }

Write-Host "Downloading and verifying the official LAYA base checkpoint (~850 MB, once) and running it ..."
uv run python -m laya_crypto.cli laya-demo
if ($LASTEXITCODE -ne 0) { throw "LAYA demo failed" }

Write-Host "`nSetup complete. Next: scripts\smoke_test.ps1"
