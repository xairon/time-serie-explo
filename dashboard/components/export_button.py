"""Download button component for model export."""

from pathlib import Path
import streamlit as st

from dashboard.utils.export import create_model_archive


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
