"""Whether CUDA is *usable*, not just visible.

`torch.cuda.is_available()` is True for a GPU the installed PyTorch build has no kernels for (e.g. a
GTX 1050, compute capability 6.1, under the cu128 wheels, which start at 7.5). Using it then fails
at the first kernel launch. This module checks the device's capability against the build's arch
list and runs one real kernel.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def cuda_status() -> tuple[bool, str]:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return False, f"torch not importable: {e!r}"
    if not torch.cuda.is_available():
        return False, "no CUDA device visible to PyTorch" if torch.version.cuda else "PyTorch built without CUDA"
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    archs = torch.cuda.get_arch_list()
    if archs and f"sm_{major}{minor}" not in archs and not any(
            a.startswith("compute_") and int(a.split("_")[1]) <= major * 10 + minor for a in archs):
        return False, (f"{name} has compute capability {major}.{minor}, but this PyTorch build "
                       f"(CUDA {torch.version.cuda}) only has kernels for {', '.join(archs)}. "
                       "Use the cu126 PyTorch wheels (the project default on Windows) for GTX 10xx cards.")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = torch.ones(8, 8, device="cuda")
            ok = bool(torch.isfinite(x @ x).all().item())
    except Exception as e:  # noqa: BLE001
        return False, f"CUDA kernel launch failed on {name}: {e!r}"
    return ok, name


def cuda_usable() -> bool:
    return cuda_status()[0]


def default_device() -> str:
    return "cuda" if cuda_usable() else "cpu"
