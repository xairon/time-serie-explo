"""Dataset Preparation - Unified data loading, exploration, and preprocessing.

This page provides a unified workflow for:
1. Loading data from CSV upload, PostgreSQL database, or saved datasets
2. Exploring data with statistics, visualizations, and quality checks
3. Configuring columns (date, station, target, covariates)
4. Setting preprocessing options (fill, normalize, features)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Dataset Preparation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if 'prep_raw_df' not in st.session_state:
    st.session_state.prep_raw_df = None
if 'prep_source_name' not in st.session_state:
    st.session_state.prep_source_name = None


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("📊 Dataset Preparation")
    
    if st.session_state.get('training_data_configured'):
        st.success(f"✅ **{st.session_state.get('training_filename', 'Data')}**")
        st.caption(f"Target: {st.session_state.get('training_target_var', 'N/A')}")
        
        if st.button("🔄 Prepare New Dataset", use_container_width=True):
            keys_to_remove = [
                'training_data', 'training_data_raw', 'training_variables',
                'training_stations', 'training_date_col', 'training_station_col',
                'training_is_multistation', 'training_filename', 'training_data_configured',
                'training_target_var', 'training_covariate_vars', 'training_preprocessing',
                'prep_raw_df', 'prep_source_name'
            ]
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    elif st.session_state.prep_raw_df is not None:
        st.info(f"📝 Configuring: **{st.session_state.prep_source_name}**")
        st.caption(f"{len(st.session_state.prep_raw_df):,} rows loaded")
        
        if st.button("❌ Cancel & Start Over", use_container_width=True):
            st.session_state.prep_raw_df = None
            st.session_state.prep_source_name = None
            st.rerun()
    else:
        st.info("No data loaded yet")
    
    # Database connection status
    st.markdown("---")
    st.subheader("🔌 Database")
    if st.session_state.get('db_connected'):
        info = st.session_state.db_connection_info
        st.success(f"**{info['database']}**")
        st.caption(f"{info['host']}:{info['port']}")
    else:
        st.caption("Not connected")


# =============================================================================
# MAIN PAGE
# =============================================================================
st.title("📊 Dataset Preparation")
st.markdown("**Load, explore, and prepare your time series data**")


# =============================================================================
# IF DATA ALREADY CONFIGURED - SHOW SUMMARY
# =============================================================================
if st.session_state.get('training_data_configured'):
    st.success("✅ Dataset ready for training!")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📁 Source", st.session_state.get('training_filename', 'N/A'))
    with col2:
        st.metric("📊 Variables", len(st.session_state.get('training_variables', [])))
    with col3:
        st.metric("🎯 Target", st.session_state.get('training_target_var', 'N/A'))
    with col4:
        if st.session_state.get('training_is_multistation'):
            st.metric("📍 Stations", len(st.session_state.get('training_stations', [])))
        else:
            df = st.session_state.get('training_data')
            if df is not None and len(df) > 0:
                duration = (df.index[-1] - df.index[0]).days / 365.25
                st.metric("📅 Duration", f"{duration:.1f} years")
    
    with st.expander("⚙️ Preprocessing Configuration"):
        prep = st.session_state.get('training_preprocessing', {})
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.write(f"**Fill Method:** {prep.get('fill_method', 'N/A')}")
        with col_p2:
            st.write(f"**Normalization:** {prep.get('normalization', 'N/A')}")
        with col_p3:
            st.write("**Features:** None")
    
    st.markdown("---")
    st.subheader("💾 Save Prepared Dataset")
    
    col_save1, col_save2 = st.columns([2, 1])
    with col_save1:
        dataset_name = st.text_input(
            "Dataset name",
            value=f"{st.session_state.get('training_filename', 'dataset').replace('.csv', '').replace('db_', '')}_prepared"
        )
    
    with col_save2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Dataset", type="primary"):
            try:
                from dashboard.utils.dataset_registry import get_dataset_registry
                registry = get_dataset_registry()
                
                if st.session_state.get('training_is_multistation'):
                    df_to_save = st.session_state['training_data_raw']
                    stations = st.session_state['training_stations']
                    station_col = st.session_state.get('training_station_col')
                else:
                    df_to_save = st.session_state['training_data']
                    stations = []
                    station_col = None
                
                registry.save_dataset(
                    name=dataset_name,
                    df=df_to_save,
                    source_file=st.session_state['training_filename'],
                    station_column=station_col,
                    stations=stations,
                    target_column=st.session_state['training_target_var'],
                    covariate_columns=st.session_state.get('training_covariate_vars', []),
                    preprocessing_config=st.session_state.get('training_preprocessing', {})
                )
                st.success(f"✅ Dataset '{dataset_name}' saved!")
            except Exception as e:
                st.error(f"Error saving: {e}")
    
    st.markdown("---")
    st.info("➡️ Go to **Train Models** to train a forecasting model with this data")
    st.stop()


# =============================================================================
# EXPLORATION FUNCTIONS
# =============================================================================
def render_exploration_tabs(df, source_name):
    """Render exploration tabs for a loaded dataframe."""
    
    st.info(f"📊 Exploring: **{source_name}** ({len(df):,} rows, {len(df.columns)} columns)")
    
    tab_preview, tab_stats, tab_quality, tab_config = st.tabs([
        "📋 Preview", "📈 Statistics", "🔍 Data Quality", "⚙️ Configure"
    ])
    
    # TAB: Preview
    with tab_preview:
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(df):,} rows")
        
        # Schema info
        with st.expander("📝 Column Schema"):
            schema_data = []
            for col in df.columns:
                non_null = df[col].dropna()
                sample_val = str(non_null.iloc[0])[:50] if len(non_null) > 0 else "N/A"
                schema_data.append({
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Non-Null': f"{len(non_null):,}",
                    'Sample': sample_val
                })
            st.dataframe(pd.DataFrame(schema_data), use_container_width=True, hide_index=True)
    
    # TAB: Statistics
    with tab_stats:
        # For large dataframes, use sampling for visualizations
        MAX_ROWS_FOR_VIZ = 100000
        is_large_df = len(df) > MAX_ROWS_FOR_VIZ
        
        if is_large_df:
            st.info(f"📊 Dataset has {len(df):,} rows - using sampling for visualizations")
            df_sample = df.sample(n=min(MAX_ROWS_FOR_VIZ, len(df)), random_state=42)
        else:
            df_sample = df
        
        # Convert object columns that look numeric (on sample for speed)
        df_converted = df_sample.copy()
        for col in df_converted.columns:
            if df_converted[col].dtype == 'object':
                try:
                    converted = pd.to_numeric(df_converted[col], errors='coerce')
                    if converted.notna().mean() > 0.5:
                        df_converted[col] = converted
                except Exception:
                    pass
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            st.markdown("##### Select Columns to Analyze")
            selected_stats_cols = st.multiselect(
                "Columns",
                options=numeric_cols,
                default=numeric_cols[:5],
                label_visibility="collapsed",
                key="stats_cols"
            )
            
            if selected_stats_cols:
                # Descriptive stats - use FULL dataframe for accuracy
                st.markdown("##### Descriptive Statistics")
                with st.spinner("Computing statistics..."):
                    stats_df = df[selected_stats_cols].describe(percentiles=[.1, .25, .5, .75, .9]).T
                    stats_df['missing'] = df[selected_stats_cols].isna().sum()
                    stats_df['zeros'] = (df[selected_stats_cols] == 0).sum()
                st.dataframe(stats_df.round(3), use_container_width=True)
                
                # Visualizations
                st.markdown("##### Visualizations")
                viz_type = st.radio(
                    "Chart type",
                    ["Histograms", "Box Plots", "Scatter Matrix"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="viz_type"
                )
                
                # Use sample for visualizations (faster for large datasets)
                df_viz = df_sample[selected_stats_cols] if is_large_df else df[selected_stats_cols]
                
                if viz_type == "Histograms":
                    cols_to_plot = selected_stats_cols[:6]
                    n_cols = min(3, len(cols_to_plot))
                    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
                    
                    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols_to_plot)
                    
                    for i, col in enumerate(cols_to_plot):
                        row = i // n_cols + 1
                        col_idx = i % n_cols + 1
                        fig.add_trace(
                            go.Histogram(x=df_viz[col].dropna(), name=col, showlegend=False,
                                        marker_color='#667eea'),
                            row=row, col=col_idx
                        )
                    
                    fig.update_layout(height=300*n_rows, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Box Plots":
                    fig_box = go.Figure()
                    for col in selected_stats_cols[:8]:
                        fig_box.add_trace(go.Box(y=df_viz[col].dropna(), name=col))
                    fig_box.update_layout(height=400, title="Distribution Comparison")
                    st.plotly_chart(fig_box, use_container_width=True)
                
                elif viz_type == "Scatter Matrix":
                    if len(selected_stats_cols) >= 2:
                        cols_for_scatter = selected_stats_cols[:4]
                        fig_scatter = px.scatter_matrix(
                            df_viz[cols_for_scatter].dropna(),
                            dimensions=cols_for_scatter,
                            height=600
                        )
                        fig_scatter.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.5))
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info("Select at least 2 columns for scatter matrix")
                
                # Correlation - use FULL data for accuracy
                if len(selected_stats_cols) > 1:
                    st.markdown("##### Correlation Analysis")
                    with st.spinner("Computing correlations..."):
                        corr_matrix = df[selected_stats_cols].corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        labels=dict(color="Correlation"),
                        color_continuous_scale="RdBu_r",
                        aspect="auto",
                        zmin=-1, zmax=1,
                        text_auto=".2f"
                    )
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Top correlations
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_pairs.append({
                                'Pair': f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}",
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                    
                    if corr_pairs:
                        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False).head(5)
                        st.markdown("**Top 5 Correlations:**")
                        st.dataframe(corr_df, hide_index=True, use_container_width=True)
        else:
            st.warning("No numeric columns found in the data.")
    
    # TAB: Data Quality
    with tab_quality:
        st.markdown("##### Missing Values")
        
        missing_data = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isna().sum().values,
            'missing_pct': (df.isna().sum().values / len(df) * 100).round(2),
            'dtype': df.dtypes.astype(str).values
        })
        missing_data = missing_data.sort_values('missing_pct', ascending=False)
        
        # Missing values chart
        missing_with_values = missing_data[missing_data['missing_pct'] > 0]
        if len(missing_with_values) > 0:
            fig_missing = px.bar(
                missing_with_values,
                x='column', y='missing_pct',
                title="Columns with Missing Values",
                labels={'missing_pct': 'Missing %', 'column': 'Column'}
            )
            fig_missing.update_layout(height=300)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing values in the dataset!")
        
        # Quality metrics
        completeness = 100 - missing_data['missing_pct'].mean()
        col_q1, col_q2, col_q3 = st.columns(3)
        
        with col_q1:
            st.metric("Completeness", f"{completeness:.1f}%")
        with col_q2:
            cols_complete = (missing_data['missing_pct'] == 0).sum()
            st.metric("Complete Columns", f"{cols_complete}/{len(missing_data)}")
        with col_q3:
            duplicate_pct = (1 - len(df.drop_duplicates()) / len(df)) * 100
            st.metric("Duplicate Rows", f"{duplicate_pct:.1f}%")
        
        # Date detection
        st.markdown("##### Temporal Analysis")
        date_cols = []
        for col in df.columns:
            if df[col].dtype in ['datetime64[ns]', 'object']:
                try:
                    pd.to_datetime(df[col].head(10))
                    date_cols.append(col)
                except:
                    pass
        
        if date_cols:
            st.success(f"Date columns detected: **{', '.join(date_cols)}**")
            
            for date_col in date_cols[:1]:
                try:
                    dates = pd.to_datetime(df[date_col])
                    min_date = dates.min()
                    max_date = dates.max()
                    
                    col_t1, col_t2, col_t3 = st.columns(3)
                    with col_t1:
                        st.metric("Start Date", min_date.strftime('%Y-%m-%d'))
                    with col_t2:
                        st.metric("End Date", max_date.strftime('%Y-%m-%d'))
                    with col_t3:
                        duration_days = (max_date - min_date).days
                        if duration_days > 365:
                            st.metric("Duration", f"{duration_days/365:.1f} years")
                        else:
                            st.metric("Duration", f"{duration_days:,} days")
                except Exception as e:
                    st.warning(f"Could not parse dates: {e}")
        else:
            st.warning("No date/timestamp columns detected")
    
    # TAB: Configure (this leads to the next step)
    with tab_config:
        render_configuration_ui(df, source_name)


def render_configuration_ui(df, source_name):
    """Render the configuration UI for preparing the dataset."""
    
    st.subheader("⚙️ Configure Dataset")
    
    # Step 1: Date Column
    st.markdown("##### 1️⃣ Time Column")
    
    potential_date_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['date', 'time', 'timestamp', 'datetime', 'day', 'jour']):
            potential_date_cols.append(col)
        elif df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(5))
                potential_date_cols.append(col)
            except:
                pass
    
    if not potential_date_cols:
        potential_date_cols = list(df.columns)
    
    date_col = st.selectbox(
        "Select the column containing dates/timestamps",
        options=potential_date_cols,
        help="This column will be used as the time index",
        key="config_date_col"
    )
    
    # Step 2: Station Column (optional)
    st.markdown("##### 2️⃣ Station Column (optional)")
    
    has_station_col = st.checkbox(
        "Data contains multiple stations/locations identified by a column",
        value=False,
        key="config_has_station"
    )
    
    station_col = None
    if has_station_col:
        # Find potential station columns: string columns or any column with many unique values
        potential_station_cols = [
            col for col in df.columns
            if col != date_col and (
                df[col].dtype == 'object' or  # String columns
                str(df[col].dtype) == 'category' or  # Category columns
                (df[col].nunique() > 1 and df[col].nunique() < len(df) * 0.5)  # Has some grouping
            )
        ]
        
        if potential_station_cols:
            station_col = st.selectbox(
                "Select the station identifier column",
                options=potential_station_cols,
                key="config_station_col"
            )
            stations_found = df[station_col].unique()
            st.success(f"**{len(stations_found):,} stations** detected: {', '.join(map(str, stations_found[:5]))}" +
                      (f" (+{len(stations_found)-5:,} more)" if len(stations_found) > 5 else ""))
        else:
            st.warning("No suitable categorical column found. Make sure your data includes a text column for station identifiers.")
    
    # Step 3: Filters (optional)
    st.markdown("##### 3️⃣ Filters (optional)")
    
    df_filtered = df.copy()
    df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')
    invalid_dates = df_filtered[date_col].isna().sum()
    if invalid_dates > 0:
        st.warning(f"⚠️ {invalid_dates:,} lignes avec dates invalides ont été ignorées pour les filtres.")
        df_filtered = df_filtered.dropna(subset=[date_col])
    
    min_date = df_filtered[date_col].min()
    max_date = df_filtered[date_col].max()
    
    date_filter_mode = "Aucun"
    start_date = end_date = None
    years_back = None
    
    if min_date is not None and max_date is not None:
        available_years = max(1, int((max_date - min_date).days // 365))
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            date_filter_mode = st.radio(
                "Filtre temporel",
                options=["Aucun", "Dernières N années", "Plage personnalisée"],
                horizontal=True,
                key="filter_date_mode"
            )
        with col_f2:
            if date_filter_mode == "Dernières N années":
                st.caption(f"Années disponibles : {available_years}")
                years_back = st.number_input(
                    "N années",
                    min_value=1,
                    max_value=available_years,
                    value=min(10, available_years),
                    key="filter_years"
                )
                end_date = max_date.date()
                start_date = (max_date - pd.DateOffset(years=years_back)).date()
            elif date_filter_mode == "Plage personnalisée":
                start_date = st.date_input(
                    "Du",
                    value=min_date.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="filter_start"
                )
                end_date = st.date_input(
                    "Au",
                    value=max_date.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="filter_end"
                )
    else:
        st.info("Aucune date exploitable pour filtrer.")
    
    if start_date and end_date:
        if start_date > end_date:
            st.error("La date de début est après la date de fin.")
            st.stop()
        df_filtered = df_filtered[
            (df_filtered[date_col] >= pd.Timestamp(start_date)) &
            (df_filtered[date_col] <= pd.Timestamp(end_date))
        ]
    
    selected_stations_filter = None
    if station_col:
        filter_stations = st.checkbox(
            "Filtrer par stations",
            value=False,
            key="filter_stations_enabled"
        )
        if filter_stations:
            station_options = sorted(df_filtered[station_col].dropna().unique().tolist())
            selected_stations_filter = st.multiselect(
                "Stations à conserver",
                options=station_options,
                default=station_options[:1] if station_options else [],
                key="filter_stations_list"
            )
            if selected_stations_filter:
                df_filtered = df_filtered[df_filtered[station_col].isin(selected_stations_filter)]
    
    st.caption(f"Après filtres: {len(df_filtered):,} lignes")
    
    # Step 4: Variables
    st.markdown("##### 4️⃣ Variables Selection")
    
    exclude_cols = [date_col]
    if station_col:
        exclude_cols.append(station_col)
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    available_vars = [col for col in numeric_cols if col not in exclude_cols]
    
    if not available_vars:
        st.error("❌ No numeric columns found in the data!")
        st.stop()
    
    col_var1, col_var2 = st.columns(2)
    
    with col_var1:
        target_var = st.selectbox(
            "🎯 Target Variable (to predict)",
            options=available_vars,
            help="The variable you want to forecast",
            key="config_target"
        )
    
    with col_var2:
        covariate_vars = st.multiselect(
            "📈 Covariates (features)",
            options=[v for v in available_vars if v != target_var],
            default=[v for v in available_vars if v != target_var][:5],
            help="Variables that can help predict the target",
            key="config_covariates"
        )
    
    all_selected_vars = [target_var] + covariate_vars
    
    # Step 5: Preprocessing
    st.markdown("##### 5️⃣ Preprocessing Configuration")
    
    col_prep1, col_prep2 = st.columns(2)
    
    with col_prep1:
        fill_method = st.selectbox(
            "Missing Values",
            options=["Linear Interpolation", "Forward fill", "Backward fill", "Drop rows"],
            help="How to handle missing values",
            key="config_fill"
        )
    
    with col_prep2:
        normalization = st.selectbox(
            "Normalization",
            options=["MinMax (0-1)", "StandardScaler (z-score)", "None"],
            help="MinMax recommended for neural networks",
            key="config_norm"
        )
    
    # Preview & Validate
    st.markdown("---")
    col_action1, col_action2 = st.columns(2)
    
    with col_action1:
        if st.button("👁️ Preview Preprocessed Data", use_container_width=True, key="btn_preview"):
            try:
                df_preview = df_filtered.copy()
                
                if station_col:
                    first_station = df_preview[station_col].iloc[0]
                    df_preview = df_preview[df_preview[station_col] == first_station]
                    st.info(f"Preview for station: {first_station}")
                
                df_preview = df_preview[[date_col] + all_selected_vars].set_index(date_col).sort_index()
                
                missing_before = df_preview.isnull().sum().sum()
                
                if fill_method == "Linear Interpolation":
                    df_preview = df_preview.interpolate(method='linear')
                elif fill_method == "Forward fill":
                    df_preview = df_preview.ffill()
                elif fill_method == "Backward fill":
                    df_preview = df_preview.bfill()
                else:
                    df_preview = df_preview.dropna()
                
                missing_after = df_preview.isnull().sum().sum()
                
                st.success(f"Missing values: {missing_before} → {missing_after}")
                st.metric("Samples after preprocessing", len(df_preview))
                st.dataframe(df_preview.head(50), use_container_width=True)
                
            except Exception as e:
                st.error(f"Preview error: {e}")
    
    with col_action2:
        if st.button("✅ Validate & Load Dataset", type="primary", use_container_width=True, key="btn_validate"):
            try:
                if df_filtered.empty:
                    st.error("❌ Aucun enregistrement après filtrage.")
                    st.stop()
                
                df_processed = df_filtered.copy()
                
                # Check for duplicate dates if no station column selected
                if not station_col:
                    date_counts = df_processed[date_col].value_counts()
                    duplicates = date_counts[date_counts > 1]
                    
                    if len(duplicates) > 0:
                        n_duplicates = len(duplicates)
                        total_duplicate_rows = duplicates.sum() - len(duplicates)  # Extra rows
                        sample_dates = duplicates.head(5).index.tolist()
                        sample_str = ", ".join([str(d)[:10] for d in sample_dates])
                        
                        st.error(f"""
                        ❌ **Duplicate dates detected!**
                        
                        Found **{n_duplicates:,} dates** with multiple rows ({total_duplicate_rows:,} extra rows).
                        
                        Examples: {sample_str}{'...' if n_duplicates > 5 else ''}
                        
                        **This usually means your data contains multiple stations/locations.**
                        
                        👉 Go back and check **"Data contains multiple stations"** and select the station identifier column.
                        """)
                        st.stop()
                
                preprocessing_config = {
                    'fill_method': fill_method,
                    'normalization': normalization,
                    'date_filter': {
                        'mode': date_filter_mode,
                        'start': str(start_date) if start_date else None,
                        'end': str(end_date) if end_date else None,
                        'years': years_back
                    },
                    'station_filter': selected_stations_filter
                }
                
                if station_col:
                    df_processed = df_processed[[date_col, station_col] + all_selected_vars]
                    
                    st.session_state['training_data_raw'] = df_processed
                    st.session_state['training_stations'] = df_processed[station_col].unique().tolist()
                    st.session_state['training_date_col'] = date_col
                    st.session_state['training_station_col'] = station_col
                    st.session_state['training_variables'] = all_selected_vars
                    st.session_state['training_target_var'] = target_var
                    st.session_state['training_covariate_vars'] = covariate_vars
                    st.session_state['training_is_multistation'] = True
                else:
                    df_processed = df_processed[[date_col] + all_selected_vars]
                    df_processed = df_processed.set_index(date_col).sort_index()
                    
                    st.session_state['training_data'] = df_processed
                    st.session_state['training_variables'] = all_selected_vars
                    st.session_state['training_target_var'] = target_var
                    st.session_state['training_covariate_vars'] = covariate_vars
                    st.session_state['training_is_multistation'] = False
                
                st.session_state['training_filename'] = source_name
                st.session_state['training_dataset_name'] = source_name
                st.session_state['training_preprocessing'] = preprocessing_config
                st.session_state['training_data_configured'] = True
                
                st.session_state.prep_raw_df = None
                st.session_state.prep_source_name = None
                
                st.success("✅ Dataset loaded successfully!")
                st.balloons()
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())


# =============================================================================
# IF RAW DATA LOADED - SHOW EXPLORATION + CONFIGURATION
# =============================================================================
if st.session_state.prep_raw_df is not None:
    render_exploration_tabs(st.session_state.prep_raw_df, st.session_state.prep_source_name)
    st.stop()


# =============================================================================
# DATA SOURCE SELECTION (when no raw data loaded yet)
# =============================================================================
st.markdown("---")

tab_csv, tab_database, tab_saved = st.tabs([
    "📁 Upload CSV",
    "🔌 From Database", 
    "📦 Saved Datasets"
])


# -----------------------------------------------------------------------------
# TAB: Upload CSV
# -----------------------------------------------------------------------------
with tab_csv:
    st.subheader("Upload CSV File")
    
    st.markdown("""
    **Expected format:**
    - A date/time column
    - Numeric target variable
    - Optional: covariates, station identifier
    """)
    
    with st.expander("📥 Download example CSV"):
        example_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=365, freq='D'),
            'level': [10.5 + i*0.01 + (i%30)*0.2 for i in range(365)],
            'precipitation': [2.3 if i%5==0 else 0.5 for i in range(365)],
            'temperature': [15 + (i%365)*0.05 for i in range(365)]
        })
        st.download_button(
            "📥 Download example.csv",
            data=example_df.to_csv(index=False),
            file_name="example_training_data.csv",
            mime="text/csv"
        )
        st.dataframe(example_df.head(10), use_container_width=True)
    
    uploaded_file = st.file_uploader(
        "Select your CSV file",
        type=['csv'],
        help="CSV format required"
    )
    
    if uploaded_file is not None:
        try:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin1')
            
            st.success(f"✅ **{uploaded_file.name}** loaded ({len(df):,} rows, {len(df.columns)} columns)")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("➡️ Continue to Exploration & Configuration", type="primary", key="csv_continue"):
                st.session_state.prep_raw_df = df
                st.session_state.prep_source_name = uploaded_file.name
                st.rerun()
                
        except Exception as e:
            st.error(f"Error reading file: {e}")


# -----------------------------------------------------------------------------
# TAB: From Database
# -----------------------------------------------------------------------------
with tab_database:
    st.subheader("Load from PostgreSQL")
    
    # Connection form if not connected
    if not st.session_state.get('db_connected'):
        st.markdown("##### Connect to Database")
        
        col1, col2 = st.columns(2)
        with col1:
            db_host = st.text_input("Host", value="dib-2019006065", key="db_host")
            db_port = st.number_input("Port", value=49502, min_value=1, max_value=65535, key="db_port")
            db_name = st.text_input("Database", value="postgres", key="db_name")

        with col2:
            db_user = st.text_input("Username", value="postgres", key="db_user")
            db_password = st.text_input("Password", type="password", key="db_password")
        
        if st.button("🔌 Connect", type="primary", use_container_width=True, key="db_connect"):
            if not all([db_host, db_name, db_user, db_password]):
                st.error("Please fill in all required fields")
            else:
                try:
                    from dashboard.utils.postgres_connector import create_connection, test_connection
                    
                    with st.spinner("Connecting..."):
                        engine = create_connection(
                            host=db_host,
                            port=int(db_port),
                            database=db_name,
                            user=db_user,
                            password=db_password
                        )
                        success, message = test_connection(engine)
                    
                    if success:
                        st.session_state.db_engine = engine
                        st.session_state.db_connected = True
                        st.session_state.db_connection_info = {
                            'host': db_host, 'port': int(db_port), 'database': db_name,
                            'user': db_user, 'schema': 'gold'
                        }
                        st.success(f"Connected! {message}")
                        st.rerun()
                    else:
                        st.error(f"Connection failed: {message}")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    else:
        # Already connected - show table selection
        info = st.session_state.db_connection_info
        st.success(f"✅ Connected to **{info['database']}** ({info['host']}:{info['port']})")
        
        col_schema, col_disconnect = st.columns([3, 1])
        with col_disconnect:
            if st.button("Disconnect", key="db_disconnect"):
                st.session_state.db_engine = None
                st.session_state.db_connected = False
                st.rerun()
        
        engine = st.session_state.db_engine
        
        try:
            from dashboard.utils.postgres_connector import (
                list_schemas, list_tables_and_views, get_table_schema, get_row_count,
                detect_date_columns, fetch_data, get_date_range, get_distinct_values
            )
            
            # Schema selection
            with col_schema:
                available_schemas = list_schemas(engine)
                current_schema = info.get('schema', 'public')
                schema = st.selectbox(
                    "Schema",
                    options=available_schemas,
                    index=available_schemas.index(current_schema) if current_schema in available_schemas else 0,
                    key="db_schema"
                )
                st.session_state.db_connection_info['schema'] = schema
            
            # Table selection
            tables_views = list_tables_and_views(engine, schema)
            all_items = (
                [f"[VIEW] {v}" for v in tables_views['views']] +
                [f"[TABLE] {t}" for t in tables_views['tables']]
            )
            
            if not all_items:
                st.warning(f"No tables/views found in schema '{schema}'")
            else:
                selected_item = st.selectbox(
                    "Select Table/View",
                    options=all_items,
                    key="db_table_select"
                )
                table_name = selected_item.replace("[TABLE] ", "").replace("[VIEW] ", "")
                
                # Quick info
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    row_count = get_row_count(engine, table_name, schema)
                    st.metric("Rows", f"{row_count:,}")
                with col_info2:
                    columns_info = get_table_schema(engine, table_name, schema)
                    st.metric("Columns", len(columns_info))
                
                # Column selection
                st.markdown("---")
                st.markdown("##### Select Columns to Load")
                
                all_columns = [col['name'] for col in columns_info]
                date_columns = detect_date_columns(engine, table_name, schema)
                
                col_db1, col_db2 = st.columns(2)
                
                with col_db1:
                    date_column = st.selectbox(
                        "Date Column",
                        options=[None] + all_columns,
                        index=(all_columns.index(date_columns[0]) + 1) if date_columns and date_columns[0] in all_columns else 0,
                        key="db_date_col"
                    )
                
                with col_db2:
                    # Station column (optional) - for multi-station data
                    text_types = ['varchar', 'text', 'char', 'character']
                    potential_station_cols = [
                        col['name'] for col in columns_info
                        if any(t in col['type'].lower() for t in text_types)
                        and col['name'] != date_column
                    ]

                    station_column = st.selectbox(
                        "Station Column (optional)",
                        options=[None] + potential_station_cols + [c for c in all_columns if c not in potential_station_cols and c != date_column],
                        help="Select if data contains multiple stations/locations (e.g., code_bss)",
                        key="db_station_col"
                    )

                # Station filter (only if station column is selected)
                selected_stations_db = None
                if station_column:
                    st.markdown("##### Filter by Station (recommended for large tables)")

                    # Get ALL distinct station values (no limit)
                    with st.spinner("Loading station list..."):
                        station_values = get_distinct_values(engine, table_name, station_column, schema, limit=None)

                    if station_values:
                        st.info(f"**{len(station_values):,}** stations found in table")

                        filter_mode = st.radio(
                            "Station filter mode",
                            options=["Load all stations", "Select specific stations"],
                            horizontal=True,
                            key="db_station_filter_mode"
                        )

                        if filter_mode == "Select specific stations":
                            # Two methods: search/select OR paste codes directly
                            input_method = st.radio(
                                "Input method",
                                options=["Search and select", "Paste station codes"],
                                horizontal=True,
                                key="db_station_input_method"
                            )

                            if input_method == "Paste station codes":
                                # Direct text input for pasting codes
                                codes_text = st.text_area(
                                    "Paste station codes (one per line or comma-separated)",
                                    height=100,
                                    placeholder="BSS001ABCD\nBSS002EFGH\nor: BSS001ABCD, BSS002EFGH",
                                    key="db_station_codes_paste"
                                )

                                if codes_text.strip():
                                    # Parse codes (handle both newlines and commas)
                                    raw_codes = codes_text.replace(',', '\n').split('\n')
                                    parsed_codes = [c.strip() for c in raw_codes if c.strip()]

                                    # Validate against available stations
                                    valid_codes = [c for c in parsed_codes if c in station_values]
                                    invalid_codes = [c for c in parsed_codes if c not in station_values]

                                    if valid_codes:
                                        st.success(f"**{len(valid_codes)}** valid station(s) found")
                                        selected_stations_db = valid_codes

                                    if invalid_codes:
                                        st.warning(f"**{len(invalid_codes)}** code(s) not found in table: {', '.join(invalid_codes[:5])}" +
                                                  (f" (+{len(invalid_codes)-5} more)" if len(invalid_codes) > 5 else ""))
                                else:
                                    st.caption("Paste station codes above to filter")

                            else:
                                # Search and select method
                                search_term = st.text_input(
                                    "Search stations",
                                    key="db_station_search",
                                    placeholder="Type to filter (e.g., BSS001 or partial code)..."
                                )

                                # Filter station values based on search
                                if search_term and len(search_term) >= 2:
                                    filtered_stations = [s for s in station_values if search_term.lower() in str(s).lower()]
                                    if filtered_stations:
                                        st.caption(f"Found {len(filtered_stations)} matching station(s)")
                                    else:
                                        st.warning(f"No stations matching '{search_term}'. Try a different search term.")
                                        # Show some examples
                                        st.caption(f"Examples of available codes: {', '.join(str(s) for s in station_values[:5])}")
                                        filtered_stations = []
                                elif search_term:
                                    st.caption("Type at least 2 characters to search...")
                                    filtered_stations = station_values[:50]
                                else:
                                    # No search - show first N stations
                                    filtered_stations = station_values[:100]
                                    if len(station_values) > 100:
                                        st.caption(f"Showing first 100 of {len(station_values)} stations. Type to search.")

                                if filtered_stations:
                                    selected_stations_db = st.multiselect(
                                        "Select stations to load",
                                        options=filtered_stations,
                                        default=[],
                                        help="Select one or more stations. Only data for these stations will be loaded.",
                                        key="db_selected_stations"
                                    )

                            # Summary
                            if selected_stations_db:
                                stations_str = ", ".join([f"'{s}'" for s in selected_stations_db[:5]])
                                if len(selected_stations_db) > 5:
                                    stations_str += f" (+{len(selected_stations_db) - 5} more)"
                                st.success(f"**{len(selected_stations_db)}** station(s) selected: {stations_str}")
                            elif filter_mode == "Select specific stations":
                                st.warning("No stations selected. Please select at least one station to load data.")
                    else:
                        st.warning("Could not retrieve station values from the table.")

                # Data columns (numeric)
                st.markdown("##### Data Columns")
                numeric_types = ['int', 'float', 'numeric', 'decimal', 'double', 'real', 'bigint', 'smallint']
                numeric_cols = [
                    col['name'] for col in columns_info
                    if any(t in col['type'].lower() for t in numeric_types)
                    and col['name'] != date_column
                    and col['name'] != station_column
                ]
                
                selected_columns = st.multiselect(
                    "Select numeric columns to load",
                    options=numeric_cols if numeric_cols else [c for c in all_columns if c != date_column and c != station_column],
                    default=numeric_cols[:10] if numeric_cols else [],
                    key="db_data_cols"
                )
                
                # Date range filter
                start_date = end_date = None
                if date_column:
                    st.markdown("##### Date Range Filter")
                    min_date_str, max_date_str = get_date_range(engine, table_name, date_column, schema)
                    
                    if min_date_str and max_date_str:
                        min_date = pd.to_datetime(min_date_str).date()
                        max_date = pd.to_datetime(max_date_str).date()
                        
                        col_date1, col_date2 = st.columns(2)
                        with col_date1:
                            start_date = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="db_start")
                        with col_date2:
                            end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date, key="db_end")
                
                # Info for large tables
                if row_count and row_count > 1000000:
                    if station_column and selected_stations_db:
                        # Filtered by station - estimate is much lower
                        estimated_rows = row_count // len(station_values) * len(selected_stations_db) if station_values else row_count
                        st.info(f"Estimated **~{estimated_rows:,} rows** for {len(selected_stations_db)} station(s)")
                    elif station_column and st.session_state.get('db_station_filter_mode') == "Select specific stations":
                        st.error(f"Table has **{row_count:,} rows** - select specific stations to avoid loading all data!")
                    else:
                        st.warning(f"Table has **{row_count:,} rows** - consider filtering by station to reduce load time")

                # Load button - disable if station column selected but no stations chosen
                load_disabled = not selected_columns
                if station_column and st.session_state.get('db_station_filter_mode') == "Select specific stations" and not selected_stations_db:
                    load_disabled = True

                # Load button
                st.markdown("---")

                if st.button("Load Data from Database", type="primary", use_container_width=True, disabled=load_disabled, key="db_load"):
                    # Build filters
                    db_filters = {}
                    if station_column and selected_stations_db:
                        db_filters[station_column] = selected_stations_db

                    with st.spinner(f"Loading data from database{f' ({len(selected_stations_db)} stations)' if selected_stations_db else ''}..."):
                        # Build query columns
                        query_columns = []
                        if date_column:
                            query_columns.append(date_column)
                        if station_column:
                            query_columns.append(station_column)
                        query_columns.extend(selected_columns)

                        df = fetch_data(
                            engine, table_name, query_columns, schema,
                            date_column,
                            str(start_date) if start_date else None,
                            str(end_date) if end_date else None,
                            filters=db_filters,
                            limit=None
                        )
                        
                        if len(df) == 0:
                            st.error("No data returned. Check your filters.")
                        else:
                            st.success(f"✅ Loaded {len(df):,} rows")
                            st.session_state.prep_raw_df = df
                            st.session_state.prep_source_name = f"db_{table_name}"
                            st.rerun()
                                
        except Exception as e:
            st.error(f"Database error: {e}")
            import traceback
            st.code(traceback.format_exc())


# -----------------------------------------------------------------------------
# TAB: Saved Datasets
# -----------------------------------------------------------------------------
with tab_saved:
    st.subheader("Load Saved Dataset")
    
    try:
        from dashboard.utils.dataset_registry import get_dataset_registry
        registry = get_dataset_registry()
        datasets = registry.scan_datasets()
        
        if not datasets:
            st.info("No saved datasets found. Prepare and save a dataset first.")
        else:
            st.success(f"**{len(datasets)}** saved dataset(s) available")
            
            for dataset in datasets:
                dataset_key = f"{dataset.name}_{dataset.creation_date}"
                with st.expander(f"📦 **{dataset.name}**"):
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.write(f"**Source:** {dataset.source_file or 'N/A'}")
                        st.write(f"**Target:** {dataset.target_column or 'N/A'}")
                        st.write(f"**Rows:** {dataset.n_rows:,}")
                    with col_m2:
                        st.write(f"**Stations:** {len(dataset.stations or [])}")
                        st.write(f"**Covariates:** {len(dataset.covariate_columns or [])}")
                        st.write(f"**Created:** {dataset.creation_date[:10] if dataset.creation_date else 'N/A'}")

                    col_load, col_delete = st.columns([3, 1])
                    with col_load:
                        if st.button(f"📥 Load", key=f"load_{dataset_key}", use_container_width=True):
                            try:
                                loaded_df, loaded_config = registry.load_dataset(dataset)

                                st.session_state['training_data'] = loaded_df
                                st.session_state['training_variables'] = [loaded_config['target_column']] + loaded_config.get('covariate_columns', [])
                                st.session_state['training_target_var'] = loaded_config['target_column']
                                st.session_state['training_covariate_vars'] = loaded_config.get('covariate_columns', [])
                                st.session_state['training_is_multistation'] = len(loaded_config.get('stations', [])) > 0
                                source_file = loaded_config.get('source_file') or dataset.source_file or dataset.name
                                st.session_state['training_filename'] = source_file
                                st.session_state['training_dataset_name'] = dataset.name
                                st.session_state['training_preprocessing'] = loaded_config.get('preprocessing', {})
                                st.session_state['training_data_configured'] = True

                                if loaded_config.get('stations'):
                                    st.session_state['training_stations'] = loaded_config['stations']
                                    st.session_state['training_station_col'] = loaded_config.get('station_column')

                                st.success(f"✅ Loaded '{dataset.name}'!")
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error loading: {e}")

                    with col_delete:
                        delete_key = f"delete_{dataset_key}"
                        confirm_key = f"confirm_delete_{dataset_key}"

                        if st.session_state.get(confirm_key, False):
                            # Show confirmation buttons
                            st.warning("Confirm?")
                            col_yes, col_no = st.columns(2)
                            with col_yes:
                                if st.button("Yes", key=f"yes_{dataset_key}", type="primary", use_container_width=True):
                                    try:
                                        registry.delete_dataset(dataset)
                                        st.session_state[confirm_key] = False
                                        st.success("Deleted!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                            with col_no:
                                if st.button("No", key=f"no_{dataset_key}", use_container_width=True):
                                    st.session_state[confirm_key] = False
                                    st.rerun()
                        else:
                            if st.button("🗑️", key=delete_key, use_container_width=True, help="Delete dataset"):
                                st.session_state[confirm_key] = True
                                st.rerun()
                            
    except Exception as e:
        st.warning(f"Could not access dataset registry: {e}")


# Footer
st.markdown("---")
st.caption("📊 Junon Time Series - Dataset Preparation")
