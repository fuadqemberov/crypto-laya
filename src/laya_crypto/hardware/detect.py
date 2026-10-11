"""Inspect the local machine and choose a safe training resource profile.

Nothing here is assumed: each field is measured, or recorded as `None` with the reason it could
not be measured. The GPU vendor is read from the OS, not inferred from "gaming PC".
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

from ..config import HardwareConfig, load_hardware_config
from ..utils.io import atomic_write_text, write_json


def _run(cmd: list[str], timeout: float = 20) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _powershell(script: str) -> str | None:
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if not exe:
        return None
    return _run([exe, "-NoProfile", "-NonInteractive", "-Command", script], timeout=40)


def os_info() -> dict[str, Any]:
    info = {"system": platform.system(), "release": platform.release(), "version": platform.version(),
            "machine": platform.machine(), "python": sys.version.split()[0],
            "python_executable": sys.executable}
    if platform.system() == "Windows":
        info["windows_build"] = platform.version()
        caption = _powershell("(Get-CimInstance Win32_OperatingSystem).Caption")
        if caption:
            info["windows_edition"] = caption
    return info


def cpu_info() -> dict[str, Any]:
    name = platform.processor() or None
    if platform.system() == "Windows":
        name = _powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name") or name
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    freq = psutil.cpu_freq()
    return {"model": name, "architecture": platform.machine(), "physical_cores": psutil.cpu_count(logical=False),
            "logical_threads": psutil.cpu_count(logical=True),
            "max_mhz": round(freq.max, 0) if freq and freq.max else None}


def memory_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {"total_gb": round(vm.total / 2**30, 2), "available_gb": round(vm.available / 2**30, 2)}


def os_gpus() -> list[dict[str, Any]]:
    """Display adapters as the OS reports them, any vendor."""
    gpus: list[dict[str, Any]] = []
    if platform.system() == "Windows":
        raw = _powershell("Get-CimInstance Win32_VideoController | Select-Object Name,AdapterCompatibility,"
                          "DriverVersion,AdapterRAM | ConvertTo-Json -Compress")
        if raw:
            try:
                data = json.loads(raw)
                for g in data if isinstance(data, list) else [data]:
                    gpus.append({"name": g.get("Name"), "vendor": g.get("AdapterCompatibility"),
                                 "driver": g.get("DriverVersion"),
                                 # AdapterRAM is a 32-bit field: wrong above 4 GB, kept only as a hint
                                 "adapter_ram_hint_gb": round((g.get("AdapterRAM") or 0) / 2**30, 2)})
            except json.JSONDecodeError:
                pass
    else:
        raw = _run(["lspci"]) if shutil.which("lspci") else None
        for line in (raw or "").splitlines():
            if "VGA" in line or "3D controller" in line:
                gpus.append({"name": line.split(":", 2)[-1].strip(), "vendor": None})
    return gpus


def nvidia_smi() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "reason": "nvidia-smi not found on PATH"}
    q = "name,memory.total,memory.used,memory.free,driver_version,compute_cap,utilization.gpu,temperature.gpu"
    raw = _run([exe, f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if raw is None:
        return {"available": False, "reason": "nvidia-smi failed (driver not loaded?)"}
    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None
        gpus.append({"name": parts[0], "memory_total_mb": num(parts[1]), "memory_used_mb": num(parts[2]),
                     "memory_free_mb": num(parts[3]), "driver": parts[4], "compute_capability": parts[5],
                     "utilization_pct": num(parts[6]), "temperature_c": num(parts[7])})
    header = _run([exe]) or ""
    cuda_driver = None
    for token in header.split("|"):
        if "CUDA Version" in token:
            cuda_driver = token.split("CUDA Version:")[-1].strip().split()[0]
    return {"available": True, "gpus": gpus, "driver_cuda_version": cuda_driver}


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as e:  # noqa: BLE001 - report any import failure
        return {"importable": False, "error": repr(e)}
    info: dict[str, Any] = {"importable": True, "version": torch.__version__,
                            "built_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        devs = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            devs.append({"index": i, "name": p.name, "total_gb": round(p.total_memory / 2**30, 2),
                         "free_gb": round(free / 2**30, 2), "capability": f"{p.major}.{p.minor}",
                         "multiprocessors": p.multi_processor_count,
                         "bf16_supported": bool(torch.cuda.is_bf16_supported())})
        info["devices"] = devs
        from ..utils.device import cuda_status

        ok, detail = cuda_status()  # capability vs. build arch list, then a real kernel launch
        info["cuda_matmul_ok"] = ok
        info["arch_list"] = torch.cuda.get_arch_list()
        if not ok:
            info["cuda_error"] = detail
    else:
        info["reason"] = ("PyTorch was built without CUDA" if torch.version.cuda is None
                          else "CUDA build present but no usable device/driver")
    mps = getattr(torch.backends, "mps", None)
    info["mps_available"] = bool(mps and mps.is_available())
    return info


def tools_info() -> dict[str, Any]:
    out = {}
    for tool, args in {"git": ["--version"], "uv": ["--version"], "nvcc": ["--version"]}.items():
        exe = shutil.which(tool)
        out[tool] = ((_run([exe, *args]) or "").splitlines() or [None])[-1] if exe else None
    return out


def wsl_info() -> dict[str, Any]:
    if platform.system() != "Windows":
        return {"applicable": False, "inside_wsl": "microsoft" in platform.release().lower()}
    exe = shutil.which("wsl")
    if not exe:
        return {"applicable": True, "installed": False}
    status = _run([exe, "--status"])
    return {"applicable": True, "installed": status is not None,
            "status": status.replace("\x00", "") if status else None}


def disk_info(path: str | Path = ".") -> dict[str, Any]:
    path = Path(path).resolve()
    du = shutil.disk_usage(path)
    info = {"path": str(path), "total_gb": round(du.total / 2**30, 1), "free_gb": round(du.free / 2**30, 1)}
    # Measured sequential write/read of 128 MB in the project directory.
    try:
        size = 128 * 2**20
        block = os.urandom(2**20)
        with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
            name = f.name
            t = time.perf_counter()
            for _ in range(size // len(block)):
                f.write(block)
            f.flush()
            os.fsync(f.fileno())
            info["seq_write_mb_s"] = round(size / 2**20 / (time.perf_counter() - t), 0)
        t = time.perf_counter()
        with open(name, "rb") as f:
            while f.read(2**22):
                pass
        info["seq_read_mb_s_cached"] = round(size / 2**20 / (time.perf_counter() - t), 0)
        os.unlink(name)
    except OSError as e:
        info["benchmark_error"] = repr(e)
    return info


def select_profile(report: dict[str, Any], cfg: HardwareConfig | None = None) -> dict[str, Any]:
    """Map measured resources to training settings. Effective batch stays 64 in every tier."""
    cfg = cfg or load_hardware_config()
    t = report.get("torch", {})
    if t.get("cuda_available") and t.get("cuda_matmul_ok", True) and t.get("devices"):
        dev = t["devices"][0]
        vram = dev["total_gb"] - cfg.reserve_vram_gb
        cap = float(dev["capability"])
        # Turing GTX 16xx have slow fp16; everything else from capability 7.0 benefits from AMP.
        slow_fp16 = "GTX 16" in dev["name"]
        amp = cap >= 7.0 and not slow_fp16
        if vram < cfg.low_vram_gb:
            tier, micro, accum, ckpt, freeze = "low_vram", 2, 32, True, vram < 5
        elif vram < cfg.high_vram_gb:
            tier, micro, accum, ckpt, freeze = "medium_vram", 4, 16, True, False
        else:
            tier, micro, accum, ckpt, freeze = "high_vram", 8, 8, vram < 22, False
        return {"tier": tier, "device": "cuda", "micro_batch": micro, "grad_accum": accum, "amp": amp,
                "gradient_checkpointing": ckpt, "freeze_encoder_suggested": freeze,
                "dataloader_workers": min(4, max(0, (report["cpu"]["logical_threads"] or 2) // 4)),
                "vram_budget_gb": round(vram, 2),
                "notes": ["fp16 disabled: GTX 16xx fp16 is slower than fp32"] if slow_fp16 else []}
    return {"tier": "cpu", "device": "cpu", "micro_batch": 2, "grad_accum": 4, "amp": False,
            "gradient_checkpointing": False, "freeze_encoder_suggested": True, "dataloader_workers": 0,
            "notes": [n for n in ["No usable CUDA device: CPU is for smoke tests only. Full runs need --allow-cpu.",
                                  t.get("cuda_error") or t.get("reason")] if n]}


def inspect(project_dir: str | Path = ".") -> dict[str, Any]:
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "os": os_info(), "cpu": cpu_info(), "memory": memory_info(), "os_gpus": os_gpus(),
              "nvidia": nvidia_smi(), "torch": torch_info(), "tools": tools_info(), "wsl": wsl_info(),
              "disk": disk_info(project_dir)}
    report["profile"] = select_profile(report)
    return report


def to_markdown(r: dict[str, Any]) -> str:
    t, p = r["torch"], r["profile"]
    lines = ["# Hardware report", "", f"Generated {r['generated_utc']} (measured on this machine).", "",
             "| Item | Value |", "|---|---|",
             f"| OS | {r['os'].get('windows_edition') or r['os']['system']} {r['os']['release']} "
             f"(build {r['os']['version']}) |",
             f"| CPU | {r['cpu']['model']} - {r['cpu']['physical_cores']} cores / "
             f"{r['cpu']['logical_threads']} threads ({r['cpu']['architecture']}) |",
             f"| RAM | {r['memory']['total_gb']} GB total, {r['memory']['available_gb']} GB available |"]
    for g in r["os_gpus"]:
        lines.append(f"| Display adapter | {g.get('name')} ({g.get('vendor') or 'vendor n/a'}) |")
    if r["nvidia"].get("available"):
        for g in r["nvidia"]["gpus"]:
            lines.append(f"| NVIDIA GPU | {g['name']}, {g['memory_total_mb']:.0f} MB VRAM "
                         f"({g['memory_used_mb']:.0f} MB in use), CC {g['compute_capability']}, "
                         f"driver {g['driver']}, driver CUDA {r['nvidia'].get('driver_cuda_version')} |")
    else:
        lines.append(f"| NVIDIA | not available: {r['nvidia'].get('reason')} |")
    lines.append(f"| PyTorch | {t.get('version')} (CUDA build {t.get('built_cuda')}), "
                 f"cuda_available={t.get('cuda_available')} |")
    for d in t.get("devices", []):
        lines.append(f"| torch device {d['index']} | {d['name']} {d['total_gb']} GB, CC {d['capability']}, "
                     f"bf16={d['bf16_supported']} |")
    lines += [f"| Tools | git: {r['tools'].get('git')}; uv: {r['tools'].get('uv')}; nvcc: {r['tools'].get('nvcc')} |",
              f"| WSL | {r['wsl']} |",
              f"| Disk | {r['disk']['free_gb']} / {r['disk']['total_gb']} GB free at {r['disk']['path']}; "
              f"seq write {r['disk'].get('seq_write_mb_s')} MB/s |",
              "", "## Selected training profile", "", "```json", json.dumps(p, indent=2), "```", ""]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], out_dir: str | Path = "reports") -> tuple[Path, Path]:
    out = Path(out_dir)
    write_json(out / "hardware_report.json", report)
    atomic_write_text(out / "hardware_report.md", to_markdown(report))
    return out / "hardware_report.json", out / "hardware_report.md"
