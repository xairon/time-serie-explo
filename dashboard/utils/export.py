"""Utilities for model and data export."""

import io
import zipfile
from pathlib import Path
from typing import Optional
import streamlit as st


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


def add_download_button(model_dir: Path, key_suffix: str = ""):
    """
    Adds a download button for the archive in the Streamlit UI.
    
    Args:
        model_dir: Model directory
        key_suffix: Suffix for the button key (avoids duplicates)
    """
    if not model_dir.exists():
        st.warning("Model directory not found.")
        return
    
    # Create archive
    with st.spinner("Creating archive..."):
        archive_data = create_model_archive(model_dir)
    
    if archive_data:
        # Archive name
        archive_name = f"{model_dir.name}.zip"
        
        st.download_button(
            label="📦 Download full archive",
            data=archive_data,
            file_name=archive_name,
            mime="application/zip",
            key=f"download_archive_{model_dir.name}_{key_suffix}",
            help="Contains: model, YAML config, data (train/val/test), scalers"
        )
    else:
        st.error("Unable to create archive.")
