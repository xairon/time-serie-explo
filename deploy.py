#!/usr/bin/env python3
"""
Deployment script for time-serie-explo.
Detects hardware (CPU/CUDA/XPU) and sets up the appropriate environment.

Usage:
    python deploy.py [--backend cpu|cuda|xpu] [--force-rebuild]
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command and optionally capture output."""
    print(f"  > {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def detect_cuda() -> bool:
    """Check if NVIDIA CUDA is available."""
    # Check for nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            result = run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                print(f"  CUDA GPU detected: {result.stdout.strip().split(chr(10))[0]}")
                return True
        except Exception:
            pass
    return False


def detect_xpu() -> bool:
    """Check if Intel XPU (Arc GPU) is available."""
    # Check for sycl-ls (Intel oneAPI tool)
    if shutil.which("sycl-ls"):
        try:
            result = run_command(["sycl-ls"], capture=True, check=False)
            if result.returncode == 0 and "level_zero:gpu" in result.stdout.lower():
                print("  Intel XPU detected via sycl-ls")
                return True
        except Exception:
            pass

    # Check for Intel GPU on Windows via PowerShell
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like '*Intel*Arc*' -or $_.Name -like '*Intel*Graphics*' } | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0]
                print(f"  Intel GPU detected: {gpu_name}")
                return True
        except Exception:
            pass

    # Check for Intel GPU on Linux
    if platform.system() == "Linux":
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, check=False
            )
            if result.returncode == 0 and "Intel" in result.stdout and ("Arc" in result.stdout or "Graphics" in result.stdout):
                print("  Intel GPU detected via lspci")
                return True
        except Exception:
            pass

    return False


def check_oneapi_installed() -> tuple[bool, str | None]:
    """Check if Intel oneAPI is installed and return the path."""
    possible_paths = [
        Path("C:/Program Files (x86)/Intel/oneAPI"),
        Path("C:/Program Files/Intel/oneAPI"),
        Path("/opt/intel/oneapi"),
        Path(os.path.expanduser("~/intel/oneapi")),
    ]

    for path in possible_paths:
        setvars = path / ("setvars.bat" if platform.system() == "Windows" else "setvars.sh")
        if setvars.exists():
            return True, str(path)

    # Check environment variable
    oneapi_root = os.environ.get("ONEAPI_ROOT")
    if oneapi_root and Path(oneapi_root).exists():
        return True, oneapi_root

    return False, None


def detect_backend() -> str:
    """Auto-detect the best available backend."""
    print("\nDetecting hardware...")

    # Priority: CUDA > XPU > CPU
    if detect_cuda():
        return "cuda"

    if detect_xpu():
        oneapi_ok, oneapi_path = check_oneapi_installed()
        if oneapi_ok:
            print(f"  oneAPI found at: {oneapi_path}")
            return "xpu"
        else:
            print("  WARNING: Intel GPU detected but oneAPI not found!")
            print("  Install Intel oneAPI Base Toolkit 2025.0.1 for XPU support.")
            print("  Falling back to CPU.")

    print("  No GPU detected, using CPU backend.")
    return "cpu"


def get_lock_file(backend: str) -> Path:
    """Get the lock file path for the given backend."""
    return Path(f"uv.{backend}.lock")


def get_config_file(backend: str) -> Path:
    """Get the UV config file for the given backend."""
    return Path(f"uv.{backend}.toml")


def setup_environment(backend: str, force_rebuild: bool = False) -> None:
    """Set up the virtual environment for the given backend."""
    project_dir = Path(__file__).parent
    os.chdir(project_dir)

    venv_path = project_dir / ".venv"
    lock_file = get_lock_file(backend)
    config_file = get_config_file(backend)
    backend_marker = venv_path / ".backend"

    # Check if we need to rebuild
    current_backend = None
    if backend_marker.exists():
        current_backend = backend_marker.read_text().strip()

    need_rebuild = (
        force_rebuild or
        not venv_path.exists() or
        current_backend != backend
    )

    if need_rebuild:
        print(f"\nSetting up environment for {backend.upper()} backend...")

        # Remove existing venv if backend changed
        if venv_path.exists() and current_backend != backend:
            print(f"  Removing old venv (was {current_backend or 'unknown'})...")
            shutil.rmtree(venv_path)

        # Create venv
        print("  Creating virtual environment...")
        run_command([sys.executable, "-m", "uv", "venv", str(venv_path)])

        # Sync dependencies
        print(f"  Installing dependencies for {backend}...")
        extra_arg = f"--extra={backend}"

        cmd = [sys.executable, "-m", "uv", "sync", extra_arg]

        # Add config file if it has content (for index overrides)
        if config_file.exists():
            config_content = config_file.read_text()
            if "index" in config_content or "sources" in config_content:
                # For uv, we need to add the index config to pyproject.toml temporarily
                # or use --index-url. Let's use a different approach.
                pass

        # For CUDA and XPU, we need special index handling
        if backend == "cuda":
            # Install base deps first
            run_command([sys.executable, "-m", "uv", "sync"])
            # Then install torch from CUDA index
            run_command([
                sys.executable, "-m", "uv", "pip", "install",
                "--index-url", "https://download.pytorch.org/whl/cu124",
                "torch", "torchaudio", "torchvision"
            ])
        elif backend == "xpu":
            # Install base deps first
            run_command([sys.executable, "-m", "uv", "sync"])
            # Then install torch from Intel index
            run_command([
                sys.executable, "-m", "uv", "pip", "install",
                "--index-url", "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/",
                "torch", "torchaudio", "torchvision", "intel-extension-for-pytorch"
            ])
        else:
            # CPU - just sync with extra
            run_command(cmd)

        # Save backend marker
        backend_marker.write_text(backend)

        # Export lock file for this backend
        print(f"  Exporting lock file to {lock_file}...")
        result = run_command([
            sys.executable, "-m", "uv", "pip", "freeze"
        ], check=False, capture=True)
        if result.returncode == 0:
            lock_file.write_text(result.stdout)

        print(f"\n[OK] Environment ready for {backend.upper()}")
    else:
        print(f"\n✓ Environment already configured for {backend.upper()}")

    # Print activation instructions
    if platform.system() == "Windows":
        activate_cmd = ".venv\\Scripts\\activate"
        if backend == "xpu":
            _, oneapi_path = check_oneapi_installed()
            if oneapi_path:
                print(f"\nFor XPU support, run in cmd.exe:")
                print(f'  call "{oneapi_path}\\setvars.bat"')
                print(f"  {activate_cmd}")
    else:
        activate_cmd = "source .venv/bin/activate"
        if backend == "xpu":
            _, oneapi_path = check_oneapi_installed()
            if oneapi_path:
                print(f"\nFor XPU support:")
                print(f"  source {oneapi_path}/setvars.sh")
                print(f"  {activate_cmd}")

    print(f"\nTo activate: {activate_cmd}")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy time-serie-explo with appropriate backend"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["cpu", "cuda", "xpu", "auto"],
        default="auto",
        help="Backend to use (default: auto-detect)"
    )
    parser.add_argument(
        "--force-rebuild", "-f",
        action="store_true",
        help="Force rebuild of the virtual environment"
    )
    parser.add_argument(
        "--detect-only", "-d",
        action="store_true",
        help="Only detect hardware, don't set up environment"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Time-Serie-Explo Deployment")
    print("=" * 50)

    # Detect or use specified backend
    if args.backend == "auto":
        backend = detect_backend()
    else:
        backend = args.backend
        print(f"\nUsing specified backend: {backend.upper()}")

    if args.detect_only:
        print(f"\nDetected backend: {backend.upper()}")
        return

    # Set up environment
    setup_environment(backend, args.force_rebuild)

    print("\n" + "=" * 50)
    print("Deployment complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
