"""
Junon Time Series - Home Page.

Entry point for the forecasting platform. Handles page status display and navigation overview.
"""

import streamlit as st
from dashboard.utils.dataset_registry import get_dataset_registry

st.set_page_config(
    page_title="Junon Time Series",
    page_icon="J",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Junon Time Series")
st.markdown("**Time Series Forecasting Platform**")
st.markdown("---")

st.markdown("""
### Welcome

Use the **sidebar** to navigate between pages:

| Page | Description |
|------|-------------|
| **Dataset Preparation** | Load, configure and save datasets |
| **Train Models** | Train forecasting models |
| **Forecasting** | Make predictions and analyze results |

---

### Quick Start

1. **Dataset Preparation** - Upload CSV, configure target/covariates, save dataset
2. **Train Models** - Load saved dataset, select model, train
3. **Forecasting** - Load trained model, make predictions, analyze

""")

# Status Section
st.markdown("---")
st.subheader("Status")

col1, col2 = st.columns(2)

with col1:
    if st.session_state.get('training_data_configured'):
        st.success(f"Data loaded: **{st.session_state.get('training_filename', 'N/A')}**")
    else:
        st.info("No data loaded yet. Go to **Dataset Preparation**")

with col2:
    try:
        registry = get_dataset_registry()
        datasets = registry.scan_datasets()
        if datasets:
            st.success(f"**{len(datasets)}** prepared dataset(s) available")
        else:
            st.info("No saved datasets. Prepare and save one.")
    except Exception:
        st.info("Dataset registry not available")

st.markdown("---")
st.caption("Powered by Darts & PyTorch Lightning")
