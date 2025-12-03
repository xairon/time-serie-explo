"""Application Streamlit pour l'entraînement de modèles de prévision."""

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
### Pages disponibles
- 🎯 **Train Models** : Entraînement de modèles
- 🔮 **Forecasting** : Prédictions sur nouvelles données
- 📉 **Model Comparison** : Comparer les performances
""")
st.sidebar.markdown("---")

# Afficher les données chargées
if 'training_data_configured' in st.session_state and st.session_state['training_data_configured']:
    st.sidebar.success(f"✅ Données chargées : **{st.session_state['training_filename']}**")
    st.sidebar.info(f"📊 {len(st.session_state['training_variables'])} variables")

    if st.sidebar.button("🔄 Charger un autre fichier"):
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
    st.sidebar.warning("⚠️ Aucune donnée chargée")

# Titre principal
st.title("🎯 Junon Model Training")
st.markdown("**Entraînement de modèles de prévision de séries temporelles**")
st.markdown("---")

# Si données déjà chargées, afficher le résumé
if 'training_data_configured' in st.session_state and st.session_state['training_data_configured']:
    st.success("🎉 Données prêtes pour l'entraînement !")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📁 Fichier", st.session_state['training_filename'])

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
            st.metric("📅 Durée", f"{duration_years:.1f} ans")

    st.info("👉 Utilisez les pages dans la sidebar pour entraîner vos modèles")

    # Aperçu des données
    with st.expander("🔍 Voir un aperçu des données"):
        if st.session_state.get('training_is_multistation', False):
            st.dataframe(st.session_state['training_data_raw'].head(20), use_container_width=True)
        else:
            st.dataframe(st.session_state['training_data'].head(20), use_container_width=True)

    # Configuration preprocessing
    with st.expander("⚙️ Configuration du preprocessing"):
        preprocessing = st.session_state.get('training_preprocessing', {})
        st.json(preprocessing)

    st.stop()

# Section Upload CSV
st.subheader("📤 Charger vos données d'entraînement")

st.markdown("""
### 📝 Format attendu

Votre fichier CSV doit contenir :
- **Une colonne temporelle** (date, time, timestamp, etc.)
- **Une variable cible** à prédire (ex: niveau d'eau)
- **Des covariables** optionnelles (pluie, température, etc.)
- **Optionnel** : une colonne avec les codes stations (si plusieurs stations)

**Important** : Plus vous avez de données historiques, meilleur sera le modèle (minimum 1 an recommandé).
""")

# Exemple téléchargeable
with st.expander("📄 Télécharger un exemple de CSV"):
    example_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=365, freq='D'),
        'level': [10.5 + i*0.01 + (i%30)*0.2 for i in range(365)],
        'precipitation': [2.3 if i%5==0 else 0.5 for i in range(365)],
        'temperature': [15 + (i%365)*0.05 for i in range(365)],
        'etp': [3.0 + (i%365)*0.01 for i in range(365)]
    })

    st.download_button(
        label="📥 Télécharger example_training.csv",
        data=example_df.to_csv(index=False),
        file_name="example_training_data.csv",
        mime="text/csv"
    )

    st.dataframe(example_df.head(10), use_container_width=True)

st.markdown("---")

# Upload
uploaded_file = st.file_uploader(
    "Sélectionnez votre fichier CSV",
    type=['csv'],
    help="Le fichier doit être au format CSV"
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

        st.success(f"✅ Fichier **{uploaded_file.name}** lu avec succès ({len(df_raw):,} lignes)")

        st.markdown("### 🔧 Configuration des données")

        # Étape 1: Colonne temporelle
        st.markdown("#### 1️⃣ Colonne temporelle")

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
            "Sélectionnez la colonne contenant les dates",
            options=potential_date_cols,
            help="La colonne avec les dates/timestamps"
        )

        # Étape 2: Colonne station (optionnel)
        st.markdown("#### 2️⃣ Colonne stations (optionnel)")

        has_station_col = st.checkbox(
            "Le CSV contient plusieurs stations identifiées par une colonne",
            value=False,
            help="Cochez si une colonne identifie différentes stations"
        )

        station_col = None
        if has_station_col:
            potential_station_cols = [col for col in df_raw.columns
                                     if col != date_col and
                                     (df_raw[col].dtype == 'object' or df_raw[col].nunique() < 50)]

            if potential_station_cols:
                station_col = st.selectbox(
                    "Sélectionnez la colonne contenant les codes stations",
                    options=potential_station_cols
                )

                stations_found = df_raw[station_col].unique()
                st.info(f"📍 **{len(stations_found)} stations** détectées : {', '.join(map(str, stations_found[:5]))}" +
                       (f" (+ {len(stations_found)-5} autres)" if len(stations_found) > 5 else ""))
            else:
                st.warning("Aucune colonne catégorielle trouvée")

        # Étape 3: Variables
        st.markdown("#### 3️⃣ Variables")

        exclude_cols = [date_col]
        if station_col:
            exclude_cols.append(station_col)

        numeric_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
        available_vars = [col for col in numeric_cols if col not in exclude_cols]

        if not available_vars:
            st.error("❌ Aucune colonne numérique trouvée !")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            target_var = st.selectbox(
                "Variable cible (à prédire)",
                options=available_vars,
                help="La variable que vous voulez prédire (ex: niveau d'eau)"
            )

        with col2:
            covariate_vars = st.multiselect(
                "Covariables (optionnel)",
                options=[v for v in available_vars if v != target_var],
                help="Variables qui peuvent aider à prédire la cible (pluie, température, etc.)"
            )

        all_selected_vars = [target_var] + covariate_vars

        # Étape 4: Preprocessing
        st.markdown("#### 4️⃣ Configuration du preprocessing")

        with st.expander("⚙️ Options de preprocessing"):
            st.markdown("##### Gestion des valeurs manquantes")
            fill_method = st.selectbox(
                "Méthode",
                options=["Supprimer les lignes", "Interpolation linéaire", "Forward fill", "Backward fill"],
                help="Comment traiter les valeurs manquantes"
            )

            st.markdown("##### Normalisation")
            normalization = st.selectbox(
                "Type de normalisation",
                options=["MinMax (0-1)", "StandardScaler (z-score)", "RobustScaler (médiane+IQR)", "Aucune"],
                help="Normaliser les données améliore l'entraînement des réseaux de neurones"
            )

            st.markdown("##### Transformation")
            transformation = st.selectbox(
                "Transformation des données",
                options=["Aucune", "Log", "BoxCox", "Différenciation (order 1)"],
                help="Transformations pour rendre les données plus stationnaires"
            )

            st.markdown("##### Meta-features (features Darts)")
            use_datetime_features = st.checkbox(
                "Extraire features temporelles (jour, mois, etc.)",
                value=False,
                help="Ajoute: jour du mois, mois, jour de la semaine, etc."
            )

            use_lags = st.checkbox(
                "Ajouter des lags de la cible",
                value=False,
                help="Ajoute les valeurs passées comme features (utile pour certains modèles)"
            )

            if use_lags:
                lag_values = st.text_input(
                    "Lags à ajouter (séparés par des virgules)",
                    value="1,7,30",
                    help="Ex: 1,7,30 pour ajouter les valeurs d'il y a 1, 7 et 30 jours"
                )
                lags_list = [int(x.strip()) for x in lag_values.split(',') if x.strip()]
            else:
                lags_list = []

        # Validation et preview
        st.markdown("---")
        st.markdown("#### 5️⃣ Validation")

        if st.button("🔍 Prévisualiser les données préprocessées", use_container_width=False):
            try:
                # Processus de preview
                df_preview = df_raw.copy()
                df_preview[date_col] = pd.to_datetime(df_preview[date_col])

                if station_col:
                    # Preview première station seulement
                    first_station = df_preview[station_col].iloc[0]
                    df_preview = df_preview[df_preview[station_col] == first_station]
                    st.info(f"Preview pour la station: {first_station}")

                df_preview = df_preview[[date_col] + all_selected_vars].set_index(date_col).sort_index()

                # Gestion valeurs manquantes
                missing_before = df_preview.isnull().sum().sum()

                if fill_method == "Interpolation linéaire":
                    df_preview = df_preview.interpolate(method='linear')
                elif fill_method == "Forward fill":
                    df_preview = df_preview.fillna(method='ffill')
                elif fill_method == "Backward fill":
                    df_preview = df_preview.fillna(method='bfill')
                else:
                    df_preview = df_preview.dropna()

                missing_after = df_preview.isnull().sum().sum()

                st.success(f"✅ Valeurs manquantes : {missing_before} → {missing_after}")
                st.metric("Échantillons après preprocessing", len(df_preview))

                # Afficher les données
                st.dataframe(df_preview.head(50), use_container_width=True)

                # Stats
                st.markdown("**Statistiques**")
                st.dataframe(df_preview.describe(), use_container_width=True)

            except Exception as e:
                st.error(f"Erreur lors du preview : {e}")

        st.markdown("---")

        if st.button("✅ Valider et charger les données", type="primary", use_container_width=True):
            try:
                df_processed = df_raw.copy()
                df_processed[date_col] = pd.to_datetime(df_processed[date_col])

                # Stocker la configuration preprocessing
                preprocessing_config = {
                    'fill_method': fill_method,
                    'normalization': normalization,
                    'transformation': transformation,
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

                st.success("🎉 Données chargées avec succès !")
                st.balloons()
                st.rerun()

            except Exception as e:
                st.error(f"❌ Erreur : {e}")
                import traceback
                st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture : {e}")
        st.info("Vérifiez le format de votre fichier CSV")

# Footer
st.markdown("---")
st.caption("⚡ Junon Model Training - Powered by Darts & PyTorch Lightning")
