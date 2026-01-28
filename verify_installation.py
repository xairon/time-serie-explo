#!/usr/bin/env python
"""
Script de vérification de l'installation du projet Time Series Explo.

Usage:
    python verify_installation.py [--venv venv_dir]
"""

import sys
import os
import platform
import subprocess
from pathlib import Path

def get_venv_python(venv_dir):
    """Returns the python command for the virtual environment."""
    if platform.system() == "Windows":
        python_cmd = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_cmd = os.path.join(venv_dir, "bin", "python")
    
    if not os.path.exists(python_cmd):
        return None
    return python_cmd

def check_import(python_cmd, package_name, version_attr=None):
    """Check if a package can be imported."""
    try:
        if version_attr:
            result = subprocess.run(
                [python_cmd, "-c", f"import {package_name}; print({package_name}.{version_attr})"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return True, version
            return False, None
        else:
            result = subprocess.run(
                [python_cmd, "-c", f"import {package_name}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0, None
    except Exception as e:
        return False, str(e)

def check_torch_device(python_cmd, device_type):
    """Check if PyTorch can detect the specified device."""
    if device_type == 'cuda':
        cmd = "import torch; print('CUDA available:', torch.cuda.is_available())"
    elif device_type == 'xpu':
        cmd = "import torch; print('XPU available:', hasattr(torch, 'xpu') and torch.xpu.is_available())"
    else:
        cmd = "import torch; print('CPU mode')"
    
    try:
        result = subprocess.run(
            [python_cmd, "-c", cmd],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr
    except Exception as e:
        return False, str(e)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Time Series Explo installation")
    parser.add_argument("--venv", default="venv", help="Virtual environment directory")
    parser.add_argument("--device", choices=["cpu", "cuda", "xpu"], 
                       help="Expected device type (for verification)")
    
    args = parser.parse_args()
    
    venv_dir = args.venv
    device_type = args.device or 'cpu'
    
    print("="*60)
    print("Time Series Explo - Installation Verification")
    print("="*60)
    print(f"\nChecking virtual environment: {venv_dir}")
    
    python_cmd = get_venv_python(venv_dir)
    if not python_cmd:
        print(f"[ERROR] Virtual environment not found at '{venv_dir}'")
        print(f"  Run: python setup_env.py --device {device_type}")
        sys.exit(1)
    
    print(f"[OK] Virtual environment found: {python_cmd}\n")
    
    # Critical packages to check
    packages = [
        ("torch", "__version__"),
        ("streamlit", "__version__"),
        ("darts", None),
        ("pandas", "__version__"),
        ("numpy", "__version__"),
        ("plotly", "__version__"),
        ("optuna", "__version__"),
        ("pytorch_lightning", "__version__"),
    ]
    
    print("Checking critical packages...")
    print("-" * 60)
    
    all_ok = True
    for package, version_attr in packages:
        success, version = check_import(python_cmd, package, version_attr)
        if success:
            if version:
                print(f"[OK] {package:20s} {version}")
            else:
                print(f"[OK] {package:20s} imported")
        else:
            print(f"[ERROR] {package:20s} NOT FOUND")
            all_ok = False
    
    print("\n" + "-" * 60)
    print("Checking PyTorch device support...")
    print("-" * 60)
    
    success, info = check_torch_device(python_cmd, device_type)
    if success:
        print(f"[OK] {info}")
    else:
        print(f"[WARNING] {info}")
        if device_type in ['cuda', 'xpu']:
            print(f"  Note: {device_type.upper()} may not be available on this system")
    
    # Test Streamlit can start
    print("\n" + "-" * 60)
    print("Testing Streamlit...")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [python_cmd, "-m", "streamlit", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"[OK] Streamlit CLI available")
            print(f"  {result.stdout.strip()}")
        else:
            print(f"[ERROR] Streamlit CLI error")
            all_ok = False
    except Exception as e:
        print(f"✗ Streamlit test failed: {e}")
        all_ok = False
    
    # Check project structure
    print("\n" + "-" * 60)
    print("Checking project structure...")
    print("-" * 60)
    
    required_paths = [
        "dashboard/training/Home.py",
        "dashboard/config.py",
        "run_app.py",
        "requirements/base.txt",
    ]
    
    for path in required_paths:
        if Path(path).exists():
            print(f"[OK] {path}")
        else:
            print(f"[ERROR] {path} NOT FOUND")
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("All checks passed! Installation is ready.")
        print("\nTo run the app:")
        print(f"  python run_app.py")
        sys.exit(0)
    else:
        print("Some checks failed. Please review the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
