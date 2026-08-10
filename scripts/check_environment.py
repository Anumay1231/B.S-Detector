#!/usr/bin/env python
"""
scripts/check_environment.py
-----------------------------

Environment diagnostic for the speaker-verification project.

Reports installed library versions, performs basic import tests for every
required dependency, reports CUDA/GPU availability, and runs a small
CPU tensor operation (plus a CUDA tensor operation if a GPU is available).

This script does NOT download the ECAPA-TDNN model or any dataset. It only
verifies that the local Python environment is set up correctly.

Usage:
    python scripts/check_environment.py
"""

from __future__ import annotations

import platform
import sys


def _print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def check_versions() -> dict:
    """Import every required dependency and report its version.

    Returns a dict mapping package name -> version string ("FAILED" if the
    import failed). Exits the process with a non-zero status only after
    reporting the full report if any import fails (see main()).
    """
    _print_header("Package versions")

    versions: dict[str, str] = {}

    print(f"Python           : {platform.python_version()} ({sys.executable})")
    versions["python"] = platform.python_version()

    checks = [
        ("torch", "torch", "__version__"),
        ("torchaudio", "torchaudio", "__version__"),
        ("speechbrain", "speechbrain", "__version__"),
        ("numpy", "numpy", "__version__"),
        ("scipy", "scipy", "__version__"),
        ("scikit-learn", "sklearn", "__version__"),
        ("matplotlib", "matplotlib", "__version__"),
        ("soundfile", "soundfile", "__version__"),
    ]

    for label, module_name, version_attr in checks:
        try:
            module = __import__(module_name)
            version = getattr(module, version_attr, "unknown")
            print(f"{label:<17}: {version}")
            versions[label] = str(version)
        except Exception as exc:  # noqa: BLE001 - want to report any import failure
            print(f"{label:<17}: FAILED to import ({exc})")
            versions[label] = "FAILED"

    return versions


def check_cuda() -> bool:
    """Report CUDA / GPU availability and details.

    Returns True if CUDA is available, False otherwise. This is expected
    to report False in a CPU-only environment (e.g. this development
    sandbox) and True on a machine with a working NVIDIA GPU + driver
    (e.g. the developer's local machine with an RTX 3060).
    """
    _print_header("CUDA / GPU")

    import torch

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available   : {cuda_available}")

    if cuda_available:
        print(f"CUDA version     : {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Device count     : {device_count}")
        for i in range(device_count):
            print(f"GPU {i}            : {torch.cuda.get_device_name(i)}")
    else:
        print(
            "No CUDA-capable GPU detected in this environment. "
            "This is expected in a CPU-only sandbox/CI environment; "
            "on a machine with an NVIDIA GPU (e.g. RTX 3060) and a "
            "matching driver, this should report True."
        )

    return cuda_available


def check_tensor_ops(cuda_available: bool) -> None:
    """Create small tensors and run a basic operation, on CPU (always)
    and on CUDA (only if available)."""
    _print_header("Tensor operation test")

    import torch

    a = torch.rand(4, 4)
    b = torch.rand(4, 4)
    c = a @ b
    print(f"CPU matmul       : OK (result shape {tuple(c.shape)})")

    if cuda_available:
        try:
            a_cuda = a.to("cuda")
            b_cuda = b.to("cuda")
            c_cuda = a_cuda @ b_cuda
            print(f"CUDA matmul      : OK (result shape {tuple(c_cuda.shape)})")
        except Exception as exc:  # noqa: BLE001
            print(f"CUDA matmul      : FAILED ({exc})")
    else:
        print("CUDA matmul      : skipped (no CUDA device available)")


def main() -> int:
    print("Speaker Verification — Environment Diagnostic")
    print("=" * 47)

    versions = check_versions()
    failed = [name for name, v in versions.items() if v == "FAILED"]

    if failed:
        print()
        print(f"FAILED imports: {', '.join(failed)}")
        return 1

    cuda_available = check_cuda()
    check_tensor_ops(cuda_available)

    _print_header("Summary")
    print("All required packages imported successfully.")
    print(f"CUDA available   : {cuda_available}")
    print(
        "Device in use    : "
        + ("cuda" if cuda_available else "cpu")
    )
    print()
    print("Environment check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
