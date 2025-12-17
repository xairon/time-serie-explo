"""Database Import Page - Connect to PostgreSQL and explore/prepare datasets.

This page allows users to:
1. Connect to a PostgreSQL database
2. Browse tables and views
3. Explore data with statistics and visualizations
4. Select columns and filter rows
5. Load data as a dataset for training
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Database Import",
    page_icon="",
    layout="wide"
)

# Custom CSS for better UX
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Database Import & Explorer")
st.markdown("**Connect to PostgreSQL, explore your data, and prepare datasets**")

# Initialize session state
if 'db_engine' not in st.session_state:
    st.session_state.db_engine = None
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'db_connection_info' not in st.session_state:
    st.session_state.db_connection_info = {}
if 'db_selected_table' not in st.session_state:
    st.session_state.db_selected_table = None
if 'db_cached_data' not in st.session_state:
    st.session_state.db_cached_data = None


# =============================================================================
# SIDEBAR: CONNECTION & NAVIGATION
# =============================================================================
with st.sidebar:
    st.header("Database Connection")
    
    if st.session_state.db_connected:
        info = st.session_state.db_connection_info
        st.success(f"**{info['database']}**")
        st.caption(f"{info['host']}:{info['port']}")
        
        # Prominent disconnect button
        if st.button("Disconnect", use_container_width=True, type="secondary"):
            st.session_state.db_engine = None
            st.session_state.db_connected = False
            st.session_state.db_cached_data = None
            st.session_state.db_sample_data = None
            st.session_state.db_selected_table = None
            st.session_state.db_current_schema = None
            st.rerun()
        
        st.markdown("---")
        
        # Schema browser
        st.subheader("Schema")
        try:
            from dashboard.utils.postgres_connector import list_schemas
            available_schemas = list_schemas(st.session_state.db_engine)
            
            current_schema = st.session_state.db_connection_info.get('schema', 'public')
            if current_schema not in available_schemas:
                current_schema = available_schemas[0] if available_schemas else 'public'
            
            selected_schema = st.selectbox(
                "Schema",
                options=available_schemas,
                index=available_schemas.index(current_schema) if current_schema in available_schemas else 0,
                label_visibility="collapsed"
            )
            
            # Update schema if changed
            if selected_schema != st.session_state.db_connection_info.get('schema'):
                st.session_state.db_connection_info['schema'] = selected_schema
                st.session_state.db_sample_data = None  # Clear cache
                st.session_state.db_selected_table = None
                st.rerun()
                
        except Exception as e:
            st.warning(f"Could not list schemas: {e}")
            selected_schema = st.text_input("Schema", value="public")
            st.session_state.db_connection_info['schema'] = selected_schema
        
        st.markdown("---")
        st.subheader("Navigation")
        nav_section = st.radio(
            "Go to",
            ["Connect", "Explore", "Filter & Load"],
            index=1 if st.session_state.db_connected else 0,
            label_visibility="collapsed"
        )
    else:
        nav_section = "Connect"
        st.info("Not connected")


# =============================================================================
# SECTION: CONNECT
# =============================================================================
if nav_section == "Connect" or not st.session_state.db_connected:
    st.markdown("---")
    st.subheader("Connect to PostgreSQL")
    
    # Presets file path
    import json
    from pathlib import Path
    presets_file = Path(__file__).parent.parent.parent / "utils" / "db_presets.json"
    
    # Load saved presets
    def load_presets():
        if presets_file.exists():
            try:
                with open(presets_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save_presets(presets_dict):
        presets_file.parent.mkdir(parents=True, exist_ok=True)
        with open(presets_file, 'w') as f:
            json.dump(presets_dict, f, indent=2)
    
    saved_presets = load_presets()
    
    # Built-in presets
    builtin_presets = {
        "Local Development": {'host': 'localhost', 'port': 5432, 'database': 'postgres', 'user': 'postgres'},
        "Docker Local": {'host': 'localhost', 'port': 5433, 'database': 'postgres', 'user': 'postgres'}
    }
    
    # Combine all presets
    all_presets = {**builtin_presets, **saved_presets}
    preset_options = ["Custom"] + list(all_presets.keys())
    
    # Preset selection
    col_preset, col_manage = st.columns([3, 1])
    
    with col_preset:
        preset = st.selectbox(
            "Load Preset",
            options=preset_options,
            help="Select a saved preset or Custom"
        )
    
    with col_manage:
        st.markdown("<br>", unsafe_allow_html=True)
        if preset in saved_presets:
            if st.button("Delete", use_container_width=True):
                del saved_presets[preset]
                save_presets(saved_presets)
                st.success(f"Deleted '{preset}'")
                st.rerun()
    
    # Get preset values
    if preset != "Custom" and preset in all_presets:
        preset_values = all_presets[preset]
    else:
        preset_values = st.session_state.db_connection_info or {}
    
    # Connection form
    col1, col2 = st.columns(2)
    
    with col1:
        db_host = st.text_input("Host", value=preset_values.get('host', 'localhost'))
        db_port = st.number_input("Port", value=preset_values.get('port', 5432), min_value=1, max_value=65535)
        db_name = st.text_input("Database", value=preset_values.get('database', ''))
    
    with col2:
        db_user = st.text_input("Username", value=preset_values.get('user', ''))
        db_password = st.text_input("Password", type="password")
        
        # Save preset
        with st.expander("Save as Preset"):
            new_preset_name = st.text_input("Preset name", key="save_preset_name")
            if st.button("Save Preset"):
                if new_preset_name:
                    saved_presets[new_preset_name] = {
                        'host': db_host,
                        'port': int(db_port),
                        'database': db_name,
                        'user': db_user
                    }
                    save_presets(saved_presets)
                    st.success(f"Saved preset '{new_preset_name}'!")
                    st.rerun()
                else:
                    st.warning("Enter a name")
    
    if st.button("Connect", type="primary", use_container_width=True):
        if not all([db_host, db_name, db_user, db_password]):
            st.error("Please fill in all required fields")
        else:
            try:
                from dashboard.utils.postgres_connector import create_connection, test_connection
                
                with st.spinner("Connecting to database..."):
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
                        'user': db_user, 'schema': 'public'
                    }
                    st.success(f"Connected! {message}")
                    st.rerun()
                else:
                    st.error(f"Connection failed: {message}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Stop here if not connected
    if not st.session_state.db_connected:
        st.stop()


# =============================================================================
# IMPORTS (only if connected)
# =============================================================================
from dashboard.utils.postgres_connector import (
    list_tables_and_views, get_table_schema, get_row_count,
    detect_date_columns, detect_dimension_columns, get_distinct_values,
    fetch_data, build_query_preview
)

engine = st.session_state.db_engine
schema = st.session_state.db_connection_info.get('schema', 'public')


# =============================================================================
# SECTION: EXPLORE
# =============================================================================
if nav_section == "Explore":
    st.markdown("---")
    
    # Table selection
    col_table, col_info = st.columns([2, 1])
    
    with col_table:
        st.subheader("Select Data Source")
        
        try:
            tables_views = list_tables_and_views(engine, schema)
            all_items = (
                [f"[VIEW] {v}" for v in tables_views['views']] +  # Views first (more likely to be useful)
                [f"[TABLE] {t}" for t in tables_views['tables']]
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
        
        if not all_items:
            st.warning(f"No tables or views found in schema '{schema}'")
            st.stop()
        
        selected_item = st.selectbox(
            "Table/View",
            options=all_items,
            label_visibility="collapsed",
            help="Select the table or view to explore"
        )
        table_name = selected_item.replace("[TABLE] ", "").replace("[VIEW] ", "")
        st.session_state.db_selected_table = table_name
    
    with col_info:
        st.subheader("Quick Stats")
        try:
            row_count = get_row_count(engine, table_name, schema)
            columns_info = get_table_schema(engine, table_name, schema)
            st.metric("Rows", f"{row_count:,}")
            st.metric("Columns", len(columns_info))
        except Exception as e:
            st.warning(f"Stats unavailable: {e}")
            columns_info = []
    
    st.markdown("---")
    
    # Data Preview & Schema tabs
    tab_preview, tab_schema, tab_stats, tab_quality = st.tabs([
        "Data Preview", "Schema", "Statistics", "Data Quality"
    ])
    
    # Fetch sample data for analysis
    if 'db_sample_data' not in st.session_state or st.session_state.get('db_sample_table') != table_name:
        with st.spinner("Loading sample data..."):
            try:
                all_columns = [col['name'] for col in columns_info]
                sample_df = fetch_data(engine, table_name, all_columns, schema, limit=5000)
                st.session_state.db_sample_data = sample_df
                st.session_state.db_sample_table = table_name
            except Exception as e:
                st.error(f"Error loading data: {e}")
                sample_df = pd.DataFrame()
    else:
        sample_df = st.session_state.db_sample_data
    
    # TAB: Data Preview
    with tab_preview:
        if len(sample_df) > 0:
            st.dataframe(sample_df.head(100), use_container_width=True)
            st.caption(f"Showing 100 of {len(sample_df):,} sampled rows")
        else:
            st.warning("No data to display")
    
    # TAB: Schema
    with tab_schema:
        if columns_info:
            # Build schema dataframe with proper alignment
            schema_data = []
            for col in columns_info:
                col_name = col['name']
                row = {
                    'name': col_name,
                    'type': col['type'],
                    'nullable': col.get('nullable', True)
                }
                
                # Add sample value from sample data
                if col_name in sample_df.columns:
                    non_null = sample_df[col_name].dropna()
                    row['sample_value'] = str(non_null.iloc[0])[:50] if len(non_null) > 0 else "N/A"
                else:
                    row['sample_value'] = "N/A"
                
                schema_data.append(row)
            
            schema_df = pd.DataFrame(schema_data)
            st.dataframe(schema_df, use_container_width=True, hide_index=True)
    
    # TAB: Statistics
    with tab_stats:
        st.info(f"**Note:** Statistics based on sample of {len(sample_df):,} rows for performance. Full table has {row_count:,} rows.")
        
        if len(sample_df) > 0:
            # Try to convert object columns that look numeric
            sample_df_converted = sample_df.copy()
            for col in sample_df_converted.columns:
                if sample_df_converted[col].dtype == 'object':
                    try:
                        converted = pd.to_numeric(sample_df_converted[col], errors='coerce')
                        if converted.notna().mean() > 0.5:
                            sample_df_converted[col] = converted
                    except Exception:
                        pass
            
            numeric_cols = sample_df_converted.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                # Column selector
                st.markdown("##### Select Columns to Analyze")
                selected_stats_cols = st.multiselect(
                    "Columns",
                    options=numeric_cols,
                    default=numeric_cols[:5],
                    label_visibility="collapsed"
                )
                
                if selected_stats_cols:
                    # Stats table with more info
                    st.markdown("##### Descriptive Statistics")
                    stats_df = sample_df_converted[selected_stats_cols].describe(percentiles=[.1, .25, .5, .75, .9]).T
                    stats_df['missing'] = sample_df_converted[selected_stats_cols].isna().sum()
                    stats_df['zeros'] = (sample_df_converted[selected_stats_cols] == 0).sum()
                    stats_df['skew'] = sample_df_converted[selected_stats_cols].skew()
                    st.dataframe(stats_df.round(3), use_container_width=True)
                    
                    # Visualization options
                    st.markdown("##### Visualizations")
                    viz_type = st.radio(
                        "Chart type",
                        ["Histograms", "Box Plots", "Scatter Matrix"],
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                    
                    if viz_type == "Histograms":
                        cols_to_plot = selected_stats_cols[:6]
                        n_cols = min(3, len(cols_to_plot))
                        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
                        
                        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols_to_plot)
                        
                        for i, col in enumerate(cols_to_plot):
                            row = i // n_cols + 1
                            col_idx = i % n_cols + 1
                            fig.add_trace(
                                go.Histogram(x=sample_df_converted[col].dropna(), name=col, showlegend=False, 
                                           marker_color='#667eea'),
                                row=row, col=col_idx
                            )
                        
                        fig.update_layout(height=300*n_rows, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Box Plots":
                        fig_box = go.Figure()
                        for col in selected_stats_cols[:8]:
                            fig_box.add_trace(go.Box(y=sample_df_converted[col].dropna(), name=col))
                        fig_box.update_layout(height=400, title="Distribution Comparison")
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    elif viz_type == "Scatter Matrix":
                        if len(selected_stats_cols) >= 2:
                            cols_for_scatter = selected_stats_cols[:4]  # Max 4 for readability
                            fig_scatter = px.scatter_matrix(
                                sample_df_converted[cols_for_scatter].dropna(),
                                dimensions=cols_for_scatter,
                                height=600
                            )
                            fig_scatter.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.5))
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                            st.info("Select at least 2 columns for scatter matrix")
                    
                    # Correlation analysis
                    if len(selected_stats_cols) > 1:
                        st.markdown("##### Correlation Analysis")
                        corr_matrix = sample_df_converted[selected_stats_cols].corr()
                        
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
                st.warning("No numeric columns found. Data may be stored as text (VARCHAR).")
    
    # TAB: Data Quality
    with tab_quality:
        if len(sample_df) > 0:
            st.markdown("##### Missing Values")
            
            missing_data = pd.DataFrame({
                'column': sample_df.columns,
                'missing_count': sample_df.isna().sum().values,
                'missing_pct': (sample_df.isna().sum().values / len(sample_df) * 100).round(2),
                'dtype': sample_df.dtypes.astype(str).values
            })
            missing_data = missing_data.sort_values('missing_pct', ascending=False)
            
            # Visualize missing
            fig_missing = px.bar(
                missing_data[missing_data['missing_pct'] > 0],
                x='column', y='missing_pct',
                title="Columns with Missing Values",
                labels={'missing_pct': 'Missing %', 'column': 'Column'}
            )
            fig_missing.update_layout(height=300)
            st.plotly_chart(fig_missing, use_container_width=True)
            
            # Data quality score
            completeness = 100 - missing_data['missing_pct'].mean()
            col_q1, col_q2, col_q3 = st.columns(3)
            
            with col_q1:
                st.metric("Completeness", f"{completeness:.1f}%")
            with col_q2:
                cols_complete = (missing_data['missing_pct'] == 0).sum()
                st.metric("Complete Columns", f"{cols_complete}/{len(missing_data)}")
            with col_q3:
                duplicate_pct = (1 - len(sample_df.drop_duplicates()) / len(sample_df)) * 100
                st.metric("Duplicate Rows", f"{duplicate_pct:.1f}%")
            
            st.caption("*Stats based on sample of 5,000 rows*")
            
            # Date detection - query FULL TABLE for real min/max
            st.markdown("##### Temporal Analysis (Full Table)")
            date_cols = detect_date_columns(engine, table_name, schema)
            
            if date_cols:
                st.success(f"Date columns detected: **{', '.join(date_cols)}**")
                
                # Query actual min/max from database (NOT sample!)
                from dashboard.utils.postgres_connector import get_date_range
                
                for date_col in date_cols[:1]:  # First date column
                    min_date_str, max_date_str = get_date_range(engine, table_name, date_col, schema)
                    
                    if min_date_str and max_date_str:
                        try:
                            min_date = pd.to_datetime(min_date_str)
                            max_date = pd.to_datetime(max_date_str)
                            
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
                            
                            # Time series preview (still uses sample for performance)
                            if date_col in sample_df.columns:
                                sample_df[date_col] = pd.to_datetime(sample_df[date_col])
                                numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()[:3]
                                if numeric_cols:
                                    fig_ts = px.line(
                                        sample_df.sort_values(date_col),
                                        x=date_col, y=numeric_cols[0],
                                        title=f"Time Series Preview (sample): {numeric_cols[0]}"
                                    )
                                    fig_ts.update_layout(height=300)
                                    st.plotly_chart(fig_ts, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not parse dates: {e}")
                    else:
                        st.warning(f"Could not query date range for {date_col}")
            else:
                st.warning("No date/timestamp columns detected")


# =============================================================================
# SECTION: FILTER & LOAD
# =============================================================================
if nav_section == "Filter & Load":
    st.markdown("---")
    st.subheader("Configure Dataset")
    
    # Allow table selection directly here (not dependent on Explore)
    try:
        tables_views = list_tables_and_views(engine, schema)
        all_items = (
            [f"[VIEW] {v}" for v in tables_views['views']] +
            [f"[TABLE] {t}" for t in tables_views['tables']]
        )
    except Exception as e:
        st.error(f"Error listing tables: {e}")
        st.stop()
    
    if not all_items:
        st.warning(f"No tables or views found in schema '{schema}'")
        st.stop()
    
    # Pre-select from Explore if available
    preselected = st.session_state.get('db_selected_table')
    if preselected:
        preselected_item = f"[TABLE] {preselected}" if f"[TABLE] {preselected}" in all_items else f"[VIEW] {preselected}"
        default_idx = all_items.index(preselected_item) if preselected_item in all_items else 0
    else:
        default_idx = 0
    
    selected_item = st.selectbox(
        "Select Table/View",
        options=all_items,
        index=default_idx
    )
    table_name = selected_item.replace("[TABLE] ", "").replace("[VIEW] ", "")
    st.session_state.db_selected_table = table_name
    
    # Get schema
    columns_info = get_table_schema(engine, table_name, schema)
    all_columns = [col['name'] for col in columns_info]
    
    if not all_columns:
        st.error("Could not retrieve columns from this table")
        st.stop()
    
    # --- COLUMN SELECTION ---
    st.markdown("---")
    st.markdown("##### Select Columns")
    
    col_sel1, col_sel2 = st.columns(2)
    
    with col_sel1:
        date_columns = detect_date_columns(engine, table_name, schema)
        st.caption(f"Detected date columns: {date_columns if date_columns else 'None (will try name-based detection)'}")
        
        date_column = st.selectbox(
            "Date Column (required for time series)",
            options=[None] + all_columns,
            index=(all_columns.index(date_columns[0]) + 1) if date_columns and date_columns[0] in all_columns else 0
        )
    
    with col_sel2:
        # Accept both numeric and other potentially useful columns
        numeric_types = ['int', 'float', 'numeric', 'decimal', 'double', 'real', 'bigint', 'smallint']
        target_candidates = [
            col['name'] for col in columns_info
            if any(t in col['type'].lower() for t in numeric_types)
            and col['name'] != date_column
        ]
        
        if not target_candidates:
            st.warning("No numeric columns found. Showing all columns.")
            target_candidates = [c for c in all_columns if c != date_column]
        
        target_column = st.selectbox(
            "Target Variable (to predict)",
            options=target_candidates if target_candidates else all_columns,
            help="The variable you want to forecast"
        )
    
    # Covariate selection with smart suggestions
    st.markdown("##### Covariate Columns (features)")
    
    available_for_cov = [c for c in target_candidates if c != target_column]
    default_covs = available_for_cov[:5] if len(available_for_cov) > 5 else available_for_cov
    
    selected_covariates = st.multiselect(
        "Select covariates",
        options=available_for_cov,
        default=default_covs,
        help="Variables that can help predict the target"
    )
    
    # --- ROW FILTERS ---
    st.markdown("---")
    st.markdown("##### Filter Rows")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        if date_column:
            st.markdown("**Date Range**")
            start_date = st.date_input("From", value=datetime.now() - timedelta(days=365*10))
            end_date = st.date_input("To", value=datetime.now())
        else:
            start_date = end_date = None
            st.info("No date column selected - date filtering disabled")
    
    with col_f2:
        st.markdown("**Dimension Filters**")
        dimension_cols = detect_dimension_columns(engine, table_name, schema, max_cardinality=200)
        
        filters = {}
        if dimension_cols:
            for dim in dimension_cols[:3]:  # Max 3 dimension filters
                try:
                    values = get_distinct_values(engine, table_name, dim['name'], schema)
                    selected = st.multiselect(
                        f"{dim['name']} ({dim['cardinality']})",
                        options=values,
                        key=f"filter_{dim['name']}"
                    )
                    if selected:
                        filters[dim['name']] = selected
                except Exception:
                    pass
        else:
            st.caption("No categorical columns detected")
    
    # --- QUERY PREVIEW ---
    st.markdown("---")
    query_columns = ([date_column] if date_column else []) + [target_column] + selected_covariates
    
    with st.expander("SQL Query Preview"):
        query_preview = build_query_preview(
            table_name=table_name,
            columns=query_columns,
            schema=schema,
            date_column=date_column,
            start_date=str(start_date) if start_date else None,
            end_date=str(end_date) if end_date else None,
            filters=filters
        )
        st.code(query_preview, language="sql")
    
    # --- LOAD DATA ---
    st.markdown("---")
    col_action1, col_action2 = st.columns(2)
    
    with col_action1:
        if st.button("Preview (1000 rows)", use_container_width=True):
            with st.spinner("Loading preview..."):
                try:
                    df = fetch_data(
                        engine, table_name, query_columns, schema,
                        date_column, str(start_date) if start_date else None,
                        str(end_date) if end_date else None, filters, limit=1000
                    )
                    st.session_state.db_preview_df = df
                    st.success(f"Loaded {len(df):,} rows")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if 'db_preview_df' in st.session_state:
        st.dataframe(st.session_state.db_preview_df.head(100), use_container_width=True)
    
    with col_action2:
        if st.button("Load Full Dataset", type="primary", use_container_width=True):
            with st.spinner("Loading full dataset..."):
                try:
                    df_full = fetch_data(
                        engine, table_name, query_columns, schema,
                        date_column, str(start_date) if start_date else None,
                        str(end_date) if end_date else None, filters, limit=None
                    )
                    
                    if len(df_full) == 0:
                        st.warning("No data returned. Check your filters.")
                    else:
                        # Prepare for training
                        if date_column:
                            df_full[date_column] = pd.to_datetime(df_full[date_column])
                            df_full = df_full.set_index(date_column).sort_index()
                        
                        st.session_state['training_data'] = df_full
                        st.session_state['training_variables'] = [target_column] + selected_covariates
                        st.session_state['training_target_var'] = target_column
                        st.session_state['training_covariate_vars'] = selected_covariates
                        st.session_state['training_is_multistation'] = False
                        st.session_state['training_filename'] = f"db_{table_name}"
                        st.session_state['training_preprocessing'] = {
                            'fill_method': 'Linear Interpolation',
                            'normalization': 'MinMax (0-1)',
                            'datetime_features': False,
                            'lags': []
                        }
                        st.session_state['training_data_configured'] = True
                        
                        st.success(f"Loaded {len(df_full):,} rows!")
                        st.balloons()
                        st.info("**Next step:** Go to **Dataset Preparation** to review, then **Train Models**")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("Junon Model Training - Database Import & Explorer")
