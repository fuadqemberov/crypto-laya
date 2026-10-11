# Train a LAYA profile. Interrupt with Ctrl+C or `python -m laya_crypto.cli stop`, continue with -Resume.
#   powershell -ExecutionPolicy Bypass -File scripts\train.ps1 -Profile research -Dataset main
#   powershell -ExecutionPolicy Bypass -File scripts\train.ps1 -Profile deep -Dataset main -Resume
param(
    [Alias("Profile")][ValidateSet("smoke", "research", "deep")][string]$TrainingProfile = "research",
    [Parameter(Mandatory = $true)][string]$Dataset,
    [switch]$Resume,
    [string]$Device = "cuda",
    [switch]$AllowCpu
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$cliArgs = @("run", "python", "-m", "laya_crypto.cli", "train", "--profile", $TrainingProfile, "--dataset", $Dataset, "--device", $Device)
if ($Resume) { $cliArgs += "--resume" }
if ($AllowCpu) { $cliArgs += "--allow-cpu" }
& uv @cliArgs
if ($LASTEXITCODE -ne 0) { throw "training failed (see logs\laya_crypto.jsonl)" }
uv run python -m laya_crypto.cli select-best --dataset $Dataset --profile $TrainingProfile
uv run python -m laya_crypto.cli evaluate --dataset $Dataset
