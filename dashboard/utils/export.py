"""Utilities for model and data export."""

import io
import zipfile
from pathlib import Path
from typing import Optional


def create_model_archive(model_dir: Path) -> Optional[bytes]:
    """
    Creates a ZIP archive containing the model, config, and data.

    Args:
        model_dir: Model directory (containing model_config.yaml, etc.)

    Returns:
        Bytes of the ZIP archive, or None if error
    """
    if not model_dir.exists():
        return None

    # Create in-memory buffer
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Walk through all files in directory
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                # Relative path in archive
                arcname = file_path.relative_to(model_dir.parent)
                zip_file.write(file_path, arcname)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
