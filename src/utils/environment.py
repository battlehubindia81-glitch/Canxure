"""Environment detection for Codespaces, Colab, CPU, RAM, and optional GPU libraries."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


@dataclass(frozen=True)
class EnvironmentInfo:
    """Structured runtime environment description."""

    python: str
    platform: str
    codespaces: bool
    colab: bool
    cpu_cores: int
    ram_gb: float
    disk_free_gb: float
    gpu_available: bool
    cuda_available: bool
    gpu_name: str | None
    vram_gb: float | None
    packages: dict[str, bool]


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def detect_environment() -> EnvironmentInfo:
    """Detect runtime hardware and dependency availability.

    Purpose: produce a reproducible environment record for experiments.
    Inputs: none. Input shape: not applicable.
    Outputs: EnvironmentInfo scalar dataclass. Output shape: one record.
    Data types: strings, booleans, integers, floats, mapping.
    Exceptions: none expected; unavailable optional packages are reported false.
    Assumptions: GPU details are best-effort and may be unavailable without PyTorch.
    """
    colab = _package_available("google.colab") or "COLAB_RELEASE_TAG" in os.environ
    codespaces = os.getenv("CODESPACES", "").lower() in {"1", "true", "yes"}
    ram_gb = round((psutil.virtual_memory().total if psutil else 0) / 1024**3, 2)
    disk_free_gb = round(shutil.disk_usage(os.getcwd()).free / 1024**3, 2)
    cuda_available = False
    gpu_name = None
    vram_gb = None
    if _package_available("torch"):
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / 1024**3, 2)
    packages = {
        "torch": _package_available("torch"),
        "torch_geometric": _package_available("torch_geometric"),
        "deepchem": _package_available("deepchem"),
        "rdkit": _package_available("rdkit"),
        "lightgbm": _package_available("lightgbm"),
    }
    return EnvironmentInfo(
        python=platform.python_version(),
        platform="Google Colab" if colab else platform.system(),
        codespaces=codespaces,
        colab=colab,
        cpu_cores=os.cpu_count() or 1,
        ram_gb=ram_gb,
        disk_free_gb=disk_free_gb,
        gpu_available=cuda_available,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        packages=packages,
    )


def print_environment_report() -> EnvironmentInfo:
    """Print and return the current environment report.

    Purpose: make CPU/GPU/package status explicit at startup.
    Inputs: none. Input shape: not applicable.
    Outputs: EnvironmentInfo. Output shape: one record.
    Data types: dataclass containing primitive values.
    Exceptions: none expected.
    Assumptions: optional libraries may be absent in lightweight CI/Codespaces.
    """
    info = detect_environment()
    print("=" * 52)
    print("PRECISION ONCOLOGY DIGITAL TWIN")
    print("ENVIRONMENT")
    print("=" * 52)
    print(f"Python:               {info.python}")
    print(f"Platform:             {info.platform}")
    print(f"Codespaces:           {'YES' if info.codespaces else 'NO'}")
    print(f"Google Colab:         {'YES' if info.colab else 'NO'}")
    print(f"CPU cores:            {info.cpu_cores}")
    print(f"RAM:                  {info.ram_gb} GB")
    print(f"Disk free:            {info.disk_free_gb} GB")
    print(f"GPU:                  {info.gpu_name or 'NONE'}")
    print(f"CUDA:                 {'YES' if info.cuda_available else 'NO'}")
    for package, available in info.packages.items():
        print(f"{package:22}{'AVAILABLE' if available else 'UNAVAILABLE'}")
    print("=" * 52)
    return info


def environment_dict() -> dict[str, Any]:
    return asdict(detect_environment())
