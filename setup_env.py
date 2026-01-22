
import os
import subprocess
import sys
import platform
import shutil
import argparse

def create_venv(venv_dir):
    """Creates a virtual environment."""
    if os.path.exists(venv_dir):
        print(f"Virtual environment directory '{venv_dir}' already exists.")
        response = input("Do you want to recreate it? (y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree(venv_dir)
            print(f"Removed existing directory '{venv_dir}'.")
        else:
            print("Using existing virtual environment.")
            return

    print(f"Creating virtual environment in '{venv_dir}'...")
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    print("Virtual environment created.")

def get_venv_commands(venv_dir):
    """Returns the pip and python commands for the virtual environment."""
    if platform.system() == "Windows":
        pip_cmd = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_cmd = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        pip_cmd = os.path.join(venv_dir, "bin", "pip")
        python_cmd = os.path.join(venv_dir, "bin", "python")
    
    if not os.path.exists(pip_cmd):
        raise FileNotFoundError(f"pip executable not found at '{pip_cmd}'.")
    
    return pip_cmd, python_cmd

def install_pytorch(venv_dir, device):
    """Installs PyTorch with the appropriate index-url based on device."""
    pip_cmd, python_cmd = get_venv_commands(venv_dir)
    
    print(f"\n{'='*60}")
    print(f"Installing PyTorch for {device.upper()}...")
    print(f"{'='*60}")
    
    if device == 'cpu':
        # PyTorch CPU
        cmd = [
            pip_cmd, "install", 
            "torch>=2.0.0", "torchaudio>=2.0.0", "torchvision>=0.15.0",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
    elif device == 'cuda':
        # PyTorch CUDA 11.8
        cmd = [
            pip_cmd, "install",
            "torch>=2.0.0", "torchaudio>=2.0.0", "torchvision>=0.15.0",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
    elif device == 'xpu':
        # PyTorch XPU (Intel Arc)
        cmd = [
            pip_cmd, "install",
            "torch==2.10.0+xpu", "torchaudio==2.10.0+xpu", "torchvision==0.25.0+xpu",
            "--index-url", "https://download.pytorch.org/whl/test/xpu"
        ]
    else:
        raise ValueError(f"Unknown device: {device}")
    
    try:
        subprocess.check_call(cmd)
        print(f"✓ PyTorch installed successfully for {device.upper()}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing PyTorch: {e}")
        sys.exit(1)

def install_base_requirements(venv_dir):
    """Installs base requirements (without PyTorch)."""
    pip_cmd, python_cmd = get_venv_commands(venv_dir)
    
    base_requirements = os.path.join("requirements", "base.txt")
    if not os.path.exists(base_requirements):
        print(f"Error: Base requirements file '{base_requirements}' not found.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Installing base requirements...")
    print(f"{'='*60}")
    
    try:
        subprocess.check_call([pip_cmd, "install", "-r", base_requirements])
        print("✓ Base requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing base requirements: {e}")
        sys.exit(1)

def install_requirements(venv_dir, device):
    """Installs all requirements for the specified device."""
    pip_cmd, python_cmd = get_venv_commands(venv_dir)
    
    # Upgrade pip first
    print("Upgrading pip...")
    try:
        subprocess.check_call([python_cmd, "-m", "pip", "install", "--upgrade", "pip"])
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not upgrade pip: {e}")
    
    # Install PyTorch first with appropriate index-url
    install_pytorch(venv_dir, device)
    
    # Then install base requirements
    install_base_requirements(venv_dir)
    
    print(f"\n{'='*60}")
    print("✓ All dependencies installed successfully!")
    print(f"{'='*60}")

def verify_installation(venv_dir, device):
    """Verifies that the installation was successful."""
    pip_cmd, python_cmd = get_venv_commands(venv_dir)
    
    print(f"\n{'='*60}")
    print("Verifying installation...")
    print(f"{'='*60}")
    
    # Test Python import
    test_script = f"""
import sys
try:
    import torch
    print(f"✓ PyTorch {{torch.__version__}} imported successfully")
    
    # Check device availability
    device = '{device}'
    if device == 'cuda':
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {{torch.cuda.get_device_name(0)}}")
        else:
            print("⚠ CUDA not available (but PyTorch installed)")
    elif device == 'xpu':
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print(f"✓ XPU available")
        else:
            print("⚠ XPU not available (but PyTorch installed)")
    else:
        print("✓ CPU mode (PyTorch installed)")
    
    # Test other critical imports
    import streamlit
    print(f"✓ Streamlit {{streamlit.__version__}} imported")
    
    import darts
    print(f"✓ Darts imported")
    
    import pandas
    print(f"✓ Pandas {{pandas.__version__}} imported")
    
    print("\\n✓ All critical packages imported successfully!")
    sys.exit(0)
except ImportError as e:
    print(f"✗ Import error: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {{e}}")
    sys.exit(1)
"""
    
    # Write temporary test script
    test_file = os.path.join(venv_dir, "test_install.py")
    with open(test_file, "w") as f:
        f.write(test_script)
    
    try:
        result = subprocess.run(
            [python_cmd, test_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print("✗ Verification timed out")
        return False
    except Exception as e:
        print(f"✗ Verification error: {e}")
        return False
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)

def main():
    parser = argparse.ArgumentParser(
        description="Setup development environment for Time Series Explo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_env.py                    # Interactive mode
  python setup_env.py --device cpu       # CPU installation
  python setup_env.py --device cuda      # CUDA installation
  python setup_env.py --device xpu       # Intel XPU installation
  python setup_env.py --device cpu --venv venv_cpu  # Custom venv name
        """
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "xpu"], 
                       help="Target hardware accelerator (cpu, cuda, or xpu)")
    parser.add_argument("--venv", default="venv", 
                       help="Name of the virtual environment directory (default: venv)")
    parser.add_argument("--skip-verify", action="store_true",
                       help="Skip installation verification")
    
    args = parser.parse_args()
    
    # Interactive mode if no args provided
    device = args.device
    if not device:
        print("\n" + "="*60)
        print("Time Series Explo - Environment Setup")
        print("="*60)
        print("\nSelect target hardware:")
        print("  1. CPU (Default, works everywhere)")
        print("  2. NVIDIA CUDA (Requires NVIDIA GPU)")
        print("  3. Intel XPU (Requires Intel Arc GPU)")
        
        choice = input("\nEnter number (1-3) [default: 1]: ").strip()
        if choice == '2':
            device = 'cuda'
        elif choice == '3':
            device = 'xpu'
        else:
            device = 'cpu'
    
    venv_dir = args.venv
    if device == 'xpu' and venv_dir == 'venv':
        # Suggest venv_arc for consistency if user didn't specify
        print("\n💡 Tip: For Intel Arc, you might want to use 'venv_arc' as venv name.")
        print("   Run: python setup_env.py --device xpu --venv venv_arc\n")

    print(f"\n{'='*60}")
    print(f"Setting up environment for {device.upper()}...")
    print(f"Virtual environment: {venv_dir}")
    print(f"{'='*60}\n")
    
    # Create virtual environment
    create_venv(venv_dir)
    
    # Install requirements
    install_requirements(venv_dir, device)
    
    # Verify installation
    if not args.skip_verify:
        verify_installation(venv_dir, device)
    
    # Print activation instructions
    print(f"\n{'='*60}")
    print("✓ Setup successful!")
    print(f"{'='*60}")
    print(f"\nTo activate the environment:")
    if platform.system() == "Windows":
        print(f"    {venv_dir}\\Scripts\\activate")
    else:
        print(f"    source {venv_dir}/bin/activate")
    
    print(f"\nTo run the Streamlit app:")
    print(f"    python run_app.py")
    print(f"\nOr directly:")
    if platform.system() == "Windows":
        print(f"    {venv_dir}\\Scripts\\streamlit run dashboard\\training\\Home.py")
    else:
        print(f"    {venv_dir}/bin/streamlit run dashboard/training/Home.py")

if __name__ == "__main__":
    main()
