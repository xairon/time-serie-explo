"""
Junon Time Series - Home Page.

Entry point for the forecasting platform with workflow overview and quick actions.
"""

import streamlit as st

st.set_page_config(
    page_title="Junon Time Series",
    page_icon="J",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("Junon Time Series")
st.markdown("**Time Series Forecasting Platform**")

st.markdown("---")

# Workflow Overview with visual cards
st.subheader("Workflow")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; height: 180px;">
        <h3 style="margin:0; font-size: 1.2rem;">1. Dataset Preparation</h3>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">Load data from CSV or PostgreSQL, explore, configure target & preprocessing</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white; height: 180px;">
        <h3 style="margin:0; font-size: 1.2rem;">2. Train Models</h3>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">Select model architecture, tune hyperparameters, train</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; height: 180px;">
        <h3 style="margin:0; font-size: 1.2rem;">3. Forecasting</h3>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">Generate predictions, analyze results, explain model</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Current Status
st.subheader("Current Session Status")

col_status1, col_status2, col_status3 = st.columns(3)

with col_status1:
    st.markdown("##### Data")
    if st.session_state.get('training_data_configured'):
        filename = st.session_state.get('training_filename', 'N/A')
        target = st.session_state.get('training_target_var', 'N/A')
        st.success(f"**{filename}**")
        st.caption(f"Target: {target}")
        
        if st.session_state.get('training_data') is not None:
            df = st.session_state['training_data']
            st.caption(f"{len(df):,} rows | {len(df.columns)} columns")
    else:
        st.info("No data loaded")
        st.caption("Go to Dataset Preparation")

with col_status2:
    st.markdown("##### Saved Datasets")
    try:
        from dashboard.utils.dataset_registry import get_dataset_registry
        registry = get_dataset_registry()
        datasets = registry.scan_datasets()
        if datasets:
            st.success(f"**{len(datasets)}** dataset(s)")
            for ds in list(datasets.keys())[:3]:
                st.caption(f"- {ds}")
            if len(datasets) > 3:
                st.caption(f"... +{len(datasets)-3} more")
        else:
            st.info("No saved datasets")
    except Exception:
        st.info("Registry unavailable")

with col_status3:
    st.markdown("##### Trained Models")
    try:
        from dashboard.utils.model_registry import get_registry
        from dashboard.config import CHECKPOINTS_DIR
        model_registry = get_registry(CHECKPOINTS_DIR.parent)
        models = model_registry.list_all_models()
        if models:
            st.success(f"**{len(models)}** model(s)")
            for m in models[:3]:
                st.caption(f"- {m.model_name} ({m.station})")
            if len(models) > 3:
                st.caption(f"... +{len(models)-3} more")
        else:
            st.info("No trained models")
    except Exception:
        st.info("No trained models")

st.markdown("---")

# Quick Actions
st.subheader("Quick Actions")

col_act1, col_act2, col_act3 = st.columns(3)

with col_act1:
    st.page_link("pages/1_Dataset_Preparation.py", label="📊 Prepare Dataset", icon="📊")
    
with col_act2:
    if st.session_state.get('training_data_configured'):
        st.page_link("pages/2_Train_Models.py", label="🚀 Train a Model", icon="🚀")
    else:
        st.info("Load data first to train")

with col_act3:
    if st.session_state.get('training_data_configured'):
        st.page_link("pages/3_Forecasting.py", label="📈 Make Predictions", icon="📈")
    else:
        st.info("Train a model first")

st.markdown("---")

# Footer
st.caption("Junon Time Series | Powered by Darts & PyTorch Lightning")
