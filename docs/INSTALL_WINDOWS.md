# Installing on Windows (clean machine)

Everything runs natively on Windows 10/11 x64. WSL2 is **not** required: every dependency
(PyTorch CUDA wheels, LAYA, LightGBM, XGBoost, FastAPI) ships native Windows wheels.

## 1. Prerequisites

1. **NVIDIA driver** (only if you have an NVIDIA GPU): install the current Game Ready or Studio
   driver from nvidia.com. You do **not** need the CUDA Toolkit: the PyTorch wheel bundles its
   CUDA runtime. Check in PowerShell:

   ```powershell
   nvidia-smi
   ```

   The top-right `CUDA Version` must be **12.6 or higher** for the cu126 wheels this project pins
   on Windows. cu126 runs on GTX 10xx (Pascal, e.g. GTX 1050/1060/1080) through RTX 40xx. An
   **RTX 50xx** needs cu128: in `pyproject.toml` change `pytorch-cu126` to `pytorch-cu128` under
   `[tool.uv.sources]`, then `uv lock` and `uv sync`. AMD/Intel GPUs: training runs on CPU only (see
   [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md)).

2. **Git**:

   ```powershell
   winget install --id Git.Git -e
   ```

3. **uv** (installs Python 3.12 and all packages; no separate Python install needed):

   ```powershell
   powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Open a new PowerShell window afterwards so `uv` is on `PATH`.

## 2. Get the project

```powershell
New-Item -ItemType Directory -Force D:\Workspace | Out-Null
git clone https://github.com/fuadqemberov/crypto-laya.git D:\Workspace\laya-crypto-trainer
cd D:\Workspace\laya-crypto-trainer
```

Plan for disk space on that drive: ~2 GB environment, ~1 GB base checkpoint, data (a few GB per
year for 100 symbols at 5m with metrics), and **~12 GB per LAYA training run** (resume state with
optimizer moments + checkpoints). See [CHECKPOINT_RESUME.md](CHECKPOINT_RESUME.md).

## 3. Install

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

This runs, in order:

```powershell
uv python install 3.12
uv sync --python 3.12 --extra dev            # virtual env in .venv, versions from uv.lock
uv run python -m laya_crypto.cli hardware    # reports\hardware_report.json + .md
uv run python -m laya_crypto.cli laya-demo   # downloads + verifies the base checkpoint, runs it once
```

`uv sync` installs PyTorch from the `cu126` index on Windows (see `[tool.uv.sources]` in
`pyproject.toml`). The hardware report checks that the installed build really has kernels for your
GPU (compute capability vs. the build's arch list, then a real kernel launch); a GPU that is visible
but unsupported is reported and the CPU is used instead of crashing.

Verify CUDA is really usable (the hardware report does a real matrix multiply on the GPU):

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Virtual environment without uv (alternative)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
```

This path does not use `uv.lock`, so versions can drift; prefer `uv sync`.

## 5. First run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1 -Device cuda
```

Then continue with [DATA_PIPELINE.md](DATA_PIPELINE.md) and [TRAINING_GUIDE.md](TRAINING_GUIDE.md).

## Notes

- Hugging Face rate-limits anonymous downloads from busy IPs. If the base checkpoint download
  returns HTTP 429, wait a few minutes, or put a free read token in `.env` (`HF_TOKEN=...`) and
  run `$env:HF_TOKEN = (Get-Content .env | Select-String HF_TOKEN).ToString().Split("=")[1]` first.
- Long paths: if `git clone` fails on long paths, run `git config --global core.longpaths true`.
- Antivirus real-time scanning of `data\` slows Parquet writes considerably; consider excluding the
  project folder.
