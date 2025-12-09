"""Streamlit application for forecasting model training."""

import streamlit as st
import pandas as pd
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Junon Model Training",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("""
### Available Pages
- 🎯 **Train Models**: Train forecasting models
- 🔮 **Forecasting**: Predict on new data
- 📉 **Model Comparison**: Compare performance
""")
st.sidebar.markdown("---")

# Show loaded data
if 'training_data_configured' in st.session_state and st.session_state['training_data_configured']:
    st.sidebar.success(f"✅ Data loaded: **{st.session_state['training_filename']}**")
    st.sidebar.info(f"📊 {len(st.session_state['training_variables'])} variables")

    if st.sidebar.button("🔄 Load another file"):
        # Reset session state
        keys_to_remove = ['training_data', 'training_data_raw', 'training_variables',
                         'training_stations', 'training_date_col', 'training_station_col',
                         'training_is_multistation', 'training_filename', 'training_data_configured',
                         'training_target_var', 'training_preprocessing']
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
else:
    st.sidebar.warning("⚠️ No data loaded")

# Main Title
st.title("📊 Dataset Preparation")
st.markdown("**Prepare and configure your time series data for training**")
st.markdown("---")

# If data already loaded, show summary
if 'training_data_configured' in st.session_state and st.session_state['training_data_configured']:
    st.success("🎉 Data ready for training!")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📁 File", st.session_state['training_filename'])

    with col2:
        st.metric("📊 Variables", len(st.session_state['training_variables']))

    with col3:
        st.metric("🎯 Target", st.session_state['training_target_var'])

    with col4:
        if st.session_state.get('training_is_multistation', False):
            st.metric("🏢 Stations", len(st.session_state['training_stations']))
        else:
            df = st.session_state['training_data']
            duration_years = (df.index[-1] - df.index[0]).days / 365.25
            st.metric("📅 Duration", f"{duration_years:.1f} years")

    # Save Dataset Button
    st.markdown("---")
    st.subheader("💾 Save Prepared Dataset")
    st.caption("Save this dataset configuration to quickly load it for training later.")
    
    col_save1, col_save2 = st.columns([2, 1])
    with col_save1:
        dataset_name = st.text_input(
            "Dataset name",
            value=f"{st.session_state['training_filename'].replace('.csv', '')}_prepared",
            help="Name for the saved dataset"
        )
    
    with col_save2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer
        if st.button("💾 Save Dataset", type="primary"):
            try:
                from dashboard.utils.dataset_registry import get_dataset_registry
                
                registry = get_dataset_registry()
                
                # Get the data to save
                if st.session_state.get('training_is_multistation', False):
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
                
                st.success(f"✅ Dataset '{dataset_name}' saved successfully!")
                
            except Exception as e:
                st.error(f"❌ Error saving dataset: {e}")

    # Action Buttons
    st.markdown("---")
    col_act1, col_act2 = st.columns(2)
    
    with col_act1:
        if st.button("🔄 Prepare Another Dataset (Reset)", use_container_width=True):
             # Reset session state
            keys_to_remove = ['training_data', 'training_data_raw', 'training_variables',
                             'training_stations', 'training_date_col', 'training_station_col',
                             'training_is_multistation', 'training_filename', 'training_data_configured',
                             'training_target_var', 'training_preprocessing']
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    with col_act2:
        st.info("👉 Use the sidebar to go to **Train Models** page with this data")
    
    # Do NOT stop script execution here entirely, so sidebar remains active
    # but we obscure the upload form below
    st.divider()
    st.caption("Detailed view below is hidden because data is configured. Click Reset to change.")
    st.stop()

# If data is not configured, show the upload section
def render_upload_section():
    # Upload CSV Section
    st.subheader("📤 Upload your training data")

    st.markdown("""
    ### 📝 Expected Format

    Your CSV file must contain:
    - **A time column** (date, time, timestamp, etc.)
    - **A target variable** to predict (e.g., water level)
    - **Optional covariates** (rain, temperature, etc.)
    - **Optional**: a station code column (if multiple stations)

    **Important**: The more historical data you have, the better the model (minimum 1 year recommended).
    """)

    # Downloadable example
    with st.expander("📄 Download CSV example"):
        example_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=365, freq='D'),
            'level': [10.5 + i*0.01 + (i%30)*0.2 for i in range(365)],
            'precipitation': [2.3 if i%5==0 else 0.5 for i in range(365)],
            'temperature': [15 + (i%365)*0.05 for i in range(365)],
            'etp': [3.0 + (i%365)*0.01 for i in range(365)]
        })

        st.download_button(
            label="📥 Download example_training.csv",
            data=example_df.to_csv(index=False),
            file_name="example_training_data.csv",
            mime="text/csv"
        )

        st.dataframe(example_df.head(10), use_container_width=True)

    st.markdown("---")

    # Upload
    uploaded_file = st.file_uploader(
        "Select your CSV file",
        type=['csv'],
        help="The file must be in CSV format"
    )

    if uploaded_file is not None:
        try:
            # Lire le CSV avec gestion de l'encodage
            # Essayer UTF-8 d'abord, puis Latin-1 si échec
            try:
                df_raw = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)  # Retour au début du fichier
                df_raw = pd.read_csv(uploaded_file, encoding='latin1')

            st.success(f"✅ File **{uploaded_file.name}** read successfully ({len(df_raw):,} rows)")

            st.markdown("### 🔧 Data Configuration")

            # Step 1: Time Column
            st.markdown("#### 1️⃣ Time Column")

            potential_date_cols = []
            for col in df_raw.columns:
                if col.lower() in ['date', 'time', 'timestamp', 'datetime', 'day', 'jour']:
                    potential_date_cols.append(col)
                elif df_raw[col].dtype == 'object':
                    try:
                        pd.to_datetime(df_raw[col].head(5))
                        potential_date_cols.append(col)
                    except:
                        pass

            if not potential_date_cols:
                potential_date_cols = list(df_raw.columns)

            date_col = st.selectbox(
                "Select the column containing dates",
                options=potential_date_cols,
                help="The column with dates/timestamps"
            )

            # Step 2: Station Column (optional)
            st.markdown("#### 2️⃣ Station Column (optional)")

            has_station_col = st.checkbox(
                "CSV contains multiple stations identified by a column",
                value=False,
                help="Check if a column identifies different stations"
            )

            station_col = None
            if has_station_col:
                potential_station_cols = [col for col in df_raw.columns
                                         if col != date_col and
                                         (df_raw[col].dtype == 'object' or df_raw[col].nunique() < 50)]

                if potential_station_cols:
                    station_col = st.selectbox(
                        "Select the column containing station codes",
                        options=potential_station_cols
                    )

                    stations_found = df_raw[station_col].unique()
                    st.info(f"📍 **{len(stations_found)} stations** detected: {', '.join(map(str, stations_found[:5]))}" +
                           (f" (+ {len(stations_found)-5} others)" if len(stations_found) > 5 else ""))
                else:
                    st.warning("No categorical column found")

            # Step 3: Variables
            st.markdown("#### 3️⃣ Variables")

            exclude_cols = [date_col]
            if station_col:
                exclude_cols.append(station_col)

            numeric_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
            available_vars = [col for col in numeric_cols if col not in exclude_cols]

            if not available_vars:
                st.error("❌ No numeric columns found!")
                st.stop()

            col1, col2 = st.columns(2)

            with col1:
                target_var = st.selectbox(
                    "Target variable (to predict)",
                    options=available_vars,
                    help="The variable you want to predict (e.g., water level)"
                )

            with col2:
                covariate_vars = st.multiselect(
                    "Covariates (optional)",
                    options=[v for v in available_vars if v != target_var],
                    help="Variables that can help predict the target (rain, temperature, etc.)"
                )

            all_selected_vars = [target_var] + covariate_vars

            # Step 4: Preprocessing
            st.markdown("#### 4️⃣ Preprocessing Configuration")

            with st.expander("⚙️ Preprocessing Options", expanded=True):
                col_prep1, col_prep2 = st.columns(2)
                
                with col_prep1:
                    st.markdown("##### Missing Values")
                    fill_method = st.selectbox(
                        "Method",
                        options=["Linear Interpolation", "Forward fill", "Drop rows"],
                        help="Interpolation recommended for piezo levels"
                    )

                with col_prep2:
                    st.markdown("##### Normalization")
                    normalization = st.selectbox(
                        "Type",
                        options=["MinMax (0-1)", "StandardScaler (z-score)", "None"],
                        help="MinMax recommended for neural networks"
                    )
                
                st.markdown("##### Additional Features")
                col_feat1, col_feat2 = st.columns(2)
                
                with col_feat1:
                    use_datetime_features = st.checkbox(
                        "📅 Time Features",
                        value=False,
                        help="Adds: day, month, season (cyclic)"
                    )

                with col_feat2:
                    use_lags = st.checkbox(
                        "📊 Target Lags",
                        value=False,
                        help="Adds past values as covariates"
                    )

                if use_lags:
                    lag_values = st.text_input(
                        "Lags (comma separated)",
                        value="1,7,30",
                        help="Ex: 1,7,30 = values from 1, 7, and 30 days ago"
                    )
                    lags_list = [int(x.strip()) for x in lag_values.split(',') if x.strip()]
                else:
                    lags_list = []

            # Validation and preview
            st.markdown("---")
            st.markdown("#### 5️⃣ Validation")

            if st.button("🔍 Preview preprocessed data", use_container_width=False):
                try:
                    # Preview process
                    df_preview = df_raw.copy()
                    df_preview[date_col] = pd.to_datetime(df_preview[date_col])

                    if station_col:
                        # Preview first station only
                        first_station = df_preview[station_col].iloc[0]
                        df_preview = df_preview[df_preview[station_col] == first_station]
                        st.info(f"Preview for station: {first_station}")

                    df_preview = df_preview[[date_col] + all_selected_vars].set_index(date_col).sort_index()

                    # Gestion valeurs manquantes
                    missing_before = df_preview.isnull().sum().sum()

                    if fill_method == "Linear Interpolation":
                        df_preview = df_preview.interpolate(method='linear')
                    elif fill_method == "Forward fill":
                        df_preview = df_preview.fillna(method='ffill')
                    elif fill_method == "Backward fill":
                        df_preview = df_preview.fillna(method='bfill')
                    else:
                        df_preview = df_preview.dropna()

                    missing_after = df_preview.isnull().sum().sum()

                    st.success(f"✅ Missing values: {missing_before} → {missing_after}")
                    st.metric("Samples after preprocessing", len(df_preview))

                    # Show data
                    st.dataframe(df_preview.head(50), use_container_width=True)

                    # Stats
                    st.markdown("**Statistics**")
                    st.dataframe(df_preview.describe(), use_container_width=True)

                except Exception as e:
                    st.error(f"Preview error: {e}")

            st.markdown("---")

            if st.button("✅ Validate and load data", type="primary", use_container_width=True):
                try:
                    df_processed = df_raw.copy()
                    df_processed[date_col] = pd.to_datetime(df_processed[date_col])

                    # Stocker la configuration preprocessing
                    preprocessing_config = {
                        'fill_method': fill_method,
                        'normalization': normalization,
                        'datetime_features': use_datetime_features,
                        'lags': lags_list if use_lags else []
                    }

                    if station_col:
                        # Multi-stations
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
                        # Mono-station
                        df_processed = df_processed[[date_col] + all_selected_vars]
                        df_processed = df_processed.set_index(date_col).sort_index()

                        st.session_state['training_data'] = df_processed
                        st.session_state['training_variables'] = all_selected_vars
                        st.session_state['training_target_var'] = target_var
                        st.session_state['training_covariate_vars'] = covariate_vars
                        st.session_state['training_is_multistation'] = False

                    st.session_state['training_filename'] = uploaded_file.name
                    st.session_state['training_preprocessing'] = preprocessing_config
                    st.session_state['training_data_configured'] = True

                    st.success("🎉 Data loaded successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.info("Check your CSV file format")

# Call the function to render the upload section
render_upload_section()

# Footer
st.markdown("---")
st.caption("⚡ Junon Model Training - Powered by Darts & PyTorch Lightning")
