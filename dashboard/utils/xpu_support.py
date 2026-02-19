"""
Custom XPU support for PyTorch Lightning.

PyTorch Lightning 2.5 does not natively support Intel XPU accelerators.
This module provides custom Accelerator and Strategy classes to enable XPU training.

Usage:
    from dashboard.utils.xpu_support import get_xpu_trainer_kwargs
    trainer_kwargs = get_xpu_trainer_kwargs()
    # Pass to pl.Trainer or Darts model
"""

import torch
import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def is_xpu_available() -> bool:
    """Check if Intel XPU is available."""
    return hasattr(torch, 'xpu') and torch.xpu.is_available()


def get_xpu_device_name(device_id: int = 0) -> str:
    """Get the name of the XPU device."""
    if not is_xpu_available():
        return "XPU not available"
    try:
        return torch.xpu.get_device_name(device_id)
    except Exception:
        return "Intel XPU"


def get_xpu_device_count() -> int:
    """Get the number of XPU devices."""
    if not is_xpu_available():
        return 0
    return torch.xpu.device_count()


# Try to create custom XPU accelerator for PyTorch Lightning
try:
    from pytorch_lightning.accelerators import Accelerator
    from pytorch_lightning.strategies import SingleDeviceStrategy
    from pytorch_lightning.plugins.precision import Precision

    class XPUAccelerator(Accelerator):
        """Custom accelerator for Intel XPU devices."""

        @staticmethod
        def is_available() -> bool:
            return is_xpu_available()

        @staticmethod
        def parse_devices(devices: Any) -> Any:
            """Parse device specification."""
            if devices is None or devices == "auto":
                return [0] if is_xpu_available() else []
            if isinstance(devices, int):
                return list(range(devices))
            if isinstance(devices, (list, tuple)):
                return list(devices)
            return [0]

        @staticmethod
        def get_parallel_devices(devices: Any) -> list:
            """Get list of parallel devices."""
            parsed = XPUAccelerator.parse_devices(devices)
            return [torch.device(f"xpu:{d}") for d in parsed]

        def setup_device(self, device: torch.device) -> None:
            if device.type != "xpu":
                raise ValueError(f"XPUAccelerator only supports XPU devices, got {device.type}")

        def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
            """Get device statistics."""
            if not is_xpu_available():
                return {}
            try:
                return {
                    "xpu_memory_allocated": torch.xpu.memory_allocated(device),
                    "xpu_memory_reserved": torch.xpu.memory_reserved(device),
                }
            except Exception:
                return {}

        def teardown(self) -> None:
            if is_xpu_available():
                try:
                    torch.xpu.empty_cache()
                except Exception:
                    pass

        @staticmethod
        def auto_device_count() -> int:
            return get_xpu_device_count()

    class XPUSingleDeviceStrategy(SingleDeviceStrategy):
        """Strategy for training on a single XPU device."""

        strategy_name = "xpu_single"

        def __init__(
            self,
            device: Union[str, torch.device] = "xpu:0",
            accelerator: Optional[Accelerator] = None,
            precision_plugin: Optional[Precision] = None,
        ):
            if isinstance(device, str) and not device.startswith("xpu"):
                device = "xpu:0"
            super().__init__(
                device=device,
                accelerator=accelerator or XPUAccelerator(),
                precision_plugin=precision_plugin,
            )

        @property
        def is_distributed(self) -> bool:
            return False

        def setup(self, trainer) -> None:
            """Setup the strategy."""
            self.accelerator.setup_device(self.root_device)
            super().setup(trainer)

    XPU_LIGHTNING_SUPPORT = True
    logger.info("XPU Lightning support enabled (custom accelerator)")

except ImportError as e:
    XPU_LIGHTNING_SUPPORT = False
    XPUAccelerator = None
    XPUSingleDeviceStrategy = None
    logger.warning(f"Could not create XPU accelerator: {e}")


def get_xpu_trainer_kwargs(device_id: int = 0, force_cpu: bool = False) -> Dict[str, Any]:
    """
    Get PyTorch Lightning Trainer kwargs for XPU training.

    Args:
        device_id: XPU device ID (default: 0)
        force_cpu: Force CPU usage even if XPU is available

    Returns:
        Dict with strategy configuration for XPU.
        Falls back to CPU if XPU is not available or not supported.
    """
    if force_cpu:
        logger.info("CPU forced by user")
        return {"accelerator": "cpu"}

    if not is_xpu_available():
        logger.info("XPU not available, falling back to CPU")
        return {"accelerator": "cpu"}

    if not XPU_LIGHTNING_SUPPORT:
        logger.warning("XPU available but Lightning support not working, falling back to CPU")
        return {"accelerator": "cpu"}

    try:
        device = torch.device(f"xpu:{device_id}")
        # Strategy already includes the accelerator, so only pass strategy
        strategy = XPUSingleDeviceStrategy(device=device)

        return {
            "strategy": strategy,
            # Don't pass accelerator - it's included in strategy
        }
    except Exception as e:
        logger.error(f"Failed to configure XPU trainer: {e}")
        return {"accelerator": "cpu"}


def estimate_memory_usage(input_chunk: int, output_chunk: int, batch_size: int,
                          n_features: int = 1, model_type: str = "TSMixer") -> float:
    """
    Estimate GPU memory usage in MB for a given configuration.

    This is a rough estimate based on typical memory patterns.
    """
    # Base memory for model weights (varies by model)
    model_base_mb = {
        "TSMixer": 50,
        "TFT": 100,
        "NBEATS": 80,
        "NHiTS": 60,
        "Transformer": 80,
        "LSTM": 40,
        "GRU": 35,
        "DLinear": 10,
        "NLinear": 5,
        "TCN": 50,
        "TiDE": 60,
    }.get(model_type, 50)

    # Sequence memory: input + output chunks * batch * features * 4 bytes * 2 (forward + backward)
    seq_memory_mb = (input_chunk + output_chunk) * batch_size * n_features * 4 * 2 / (1024 * 1024)

    # Intermediate activations (rough multiplier)
    activation_multiplier = {
        "TSMixer": 8,
        "TFT": 12,
        "NBEATS": 6,
        "Transformer": 10,
    }.get(model_type, 6)

    total_mb = model_base_mb + seq_memory_mb * activation_multiplier

    return total_mb


def check_xpu_memory_available(required_mb: float, safety_margin: float = 0.8) -> bool:
    """
    Check if enough XPU memory is available.

    Args:
        required_mb: Required memory in MB
        safety_margin: Use only this fraction of total memory (default 80%)

    Returns:
        True if enough memory is available
    """
    if not is_xpu_available():
        return False

    try:
        # Get total memory (this is device-specific, may not work on all XPUs)
        # For Arc 140V, we know it's 8GB
        total_mb = 8 * 1024  # 8GB in MB (hardcoded for Arc 140V)

        available_mb = total_mb * safety_margin - torch.xpu.memory_allocated() / (1024 * 1024)

        return available_mb >= required_mb
    except Exception:
        # If we can't check, assume it's fine
        return True


def get_recommended_config(available_memory_mb: float = 6000) -> Dict[str, int]:
    """
    Get recommended configuration based on available memory.

    Args:
        available_memory_mb: Available GPU memory in MB

    Returns:
        Dict with recommended input_chunk, output_chunk, batch_size
    """
    if available_memory_mb >= 6000:  # 6GB+
        return {"input_chunk": 90, "output_chunk": 30, "batch_size": 32}
    elif available_memory_mb >= 4000:  # 4GB+
        return {"input_chunk": 60, "output_chunk": 14, "batch_size": 16}
    elif available_memory_mb >= 2000:  # 2GB+
        return {"input_chunk": 30, "output_chunk": 7, "batch_size": 8}
    else:
        return {"input_chunk": 14, "output_chunk": 7, "batch_size": 4}


def move_to_xpu(tensor_or_module, device_id: int = 0):
    """
    Move a tensor or module to XPU device.

    Args:
        tensor_or_module: PyTorch tensor or nn.Module
        device_id: XPU device ID (default: 0)

    Returns:
        The tensor/module on XPU, or unchanged if XPU not available.
    """
    if not is_xpu_available():
        return tensor_or_module

    try:
        device = torch.device(f"xpu:{device_id}")
        return tensor_or_module.to(device)
    except Exception as e:
        logger.warning(f"Failed to move to XPU: {e}")
        return tensor_or_module


def get_optimal_device() -> str:
    """
    Get the optimal available device string.

    Returns:
        'xpu', 'cuda', or 'cpu' based on availability.
    """
    if is_xpu_available():
        return 'xpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information."""
    info = {
        'xpu_available': is_xpu_available(),
        'cuda_available': torch.cuda.is_available(),
        'optimal_device': get_optimal_device(),
        'torch_version': torch.__version__,
    }

    if info['xpu_available']:
        info['xpu_count'] = get_xpu_device_count()
        info['xpu_name'] = get_xpu_device_name(0)
        info['xpu_lightning_support'] = XPU_LIGHTNING_SUPPORT

    if info['cuda_available']:
        info['cuda_count'] = torch.cuda.device_count()
        info['cuda_name'] = torch.cuda.get_device_name(0)

    return info


def cleanup_gpu_memory(model=None) -> None:
    """
    Clean up GPU memory after training.

    This function should be called after training is complete to release
    GPU memory back to the system. It handles both CUDA and XPU devices.

    Args:
        model: Optional Darts model to move to CPU before cleanup.
    """
    import gc

    # Try to move model to CPU first to release GPU tensors
    if model is not None:
        try:
            # Darts models have an internal PyTorch model
            if hasattr(model, 'model') and model.model is not None:
                model.model.cpu()
            # Also try the trainer
            # Removed model.trainer = None to avoid side effects (breaks predict())
            logger.info("Model moved to CPU")
        except Exception as e:
            logger.debug(f"Could not move model to CPU: {e}")

    # Force garbage collection first
    gc.collect()

    # Clean up CUDA memory
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("CUDA memory cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear CUDA cache: {e}")

    # Clean up XPU memory
    if is_xpu_available():
        try:
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
            logger.info("XPU memory cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear XPU cache: {e}")

    # Another round of garbage collection
    gc.collect()
