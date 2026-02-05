import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Check XPU (Intel Arc) - Native PyTorch XPU support
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f"XPU available: True")
    print(f"XPU device count: {torch.xpu.device_count()}")
    for i in range(torch.xpu.device_count()):
        try:
            name = torch.xpu.get_device_name(i)
            print(f"XPU device {i}: {name}")
        except Exception:
            print(f"XPU device {i}: Intel Arc GPU")
    # Test XPU computation
    try:
        x = torch.randn(100, 100, device='xpu')
        y = x @ x.T
        print(f"XPU compute test: OK")
    except Exception as e:
        print(f"XPU compute test failed: {e}")
else:
    print("XPU not available")

# Check CUDA
if torch.cuda.is_available():
    print(f"CUDA available: True")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available")

# Summary
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print("\n>> Using: Intel XPU (Arc GPU)")
elif torch.cuda.is_available():
    print("\n>> Using: NVIDIA CUDA")
else:
    print("\n>> Using: CPU")
