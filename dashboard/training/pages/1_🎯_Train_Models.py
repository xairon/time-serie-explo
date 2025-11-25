"""Page d'entraînement de modèles avec support étendu des modèles Darts."""

import streamlit as st
import pandas as pd
import torch
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dashboard.models_config import ALL_MODELS, MODEL_CATEGORIES, RECOMMENDED_MODELS, get_model_info, get_hyperparam_description
from dashboard.utils.preprocessing import (
    TimeSeriesPreprocessor,
    prepare_dataframe_for_darts,
    add_datetime_features,
    add_lag_features,
    split_train_val_test,
    compute_data_statistics,
    get_preprocessing_summary
)

st.set_page_config(page_title="Train Models", page_icon="🎯", layout="wide")

st.title("🎯 Train Models")
st.markdown("Entraînement de modèles de prévision avec preprocessing configurable.")
st.markdown("---")

# Vérifier si des données sont chargées
if 'training_data_configured' not in st.session_state or not st.session_state['training_data_configured']:
    st.warning("⚠️ Aucune donnée chargée")
    st.info("👉 Retournez à la page d'accueil pour charger un fichier CSV")
    st.stop()

# Récupérer les données
is_multistation = st.session_state.get('training_is_multistation', False)
target_var = st.session_state['training_target_var']
covariate_vars = st.session_state['training_covariate_vars']
preprocessing_config = st.session_state['training_preprocessing']

st.success(f"✅ Données chargées : **{st.session_state['training_filename']}**")

# Afficher résumé preprocessing
with st.expander("📋 Résumé du preprocessing configuré"):
    st.markdown(get_preprocessing_summary(preprocessing_config))

st.markdown("---")

# ============================================================================
# SÉLECTION DES DONNÉES
# ============================================================================
st.subheader("1️⃣ Sélection des données")

if is_multistation:
    df_raw = st.session_state['training_data_raw']
    date_col = st.session_state['training_date_col']
    station_col = st.session_state['training_station_col']
    stations = st.session_state['training_stations']

    col1, col2 = st.columns(2)

    with col1:
        train_mode = st.radio(
            "Mode d'entraînement",
            options=["Une station spécifique", "Toutes les stations (boucle)"],
            help="Entraîner sur une seule station ou toutes successivement"
        )

    with col2:
        if train_mode == "Une station spécifique":
            selected_station = st.selectbox(
                "Station",
                options=stations,
                help="Station à utiliser pour l'entraînement"
            )
            selected_stations = [selected_station]
        else:
            selected_stations = stations
            st.info(f"📍 **{len(selected_stations)} stations** seront entraînées successivement")

else:
    df_raw = st.session_state['training_data']
    # Utiliser le nom du fichier sans extension comme nom de station
    filename = st.session_state.get('training_filename', 'station_data')
    station_name = Path(filename).stem
    selected_stations = [station_name]
    st.info(f"📊 Données mono-station détectées : {station_name}")

st.markdown("---")

# ============================================================================
# SÉLECTION DU MODÈLE
# ============================================================================
st.subheader("2️⃣ Sélection du modèle")

col1, col2 = st.columns([1, 2])

with col1:
    # Catégorie
    selected_category = st.selectbox(
        "Catégorie",
        options=list(MODEL_CATEGORIES.keys()),
        help="Type de modèle"
    )

    # Modèles de la catégorie
    models_in_category = MODEL_CATEGORIES[selected_category]

    # Mettre en évidence les recommandés
    display_names = []
    for model_name in models_in_category:
        if model_name in RECOMMENDED_MODELS:
            display_names.append(f"⭐ {model_name}")
        else:
            display_names.append(model_name)

    selected_model_display = st.selectbox(
        "Modèle",
        options=display_names,
        help="⭐ = Recommandé pour démarrer"
    )

    # Extraire le nom réel (enlever ⭐)
    selected_model = selected_model_display.replace("⭐ ", "")

with col2:
    # Info modèle
    model_info = get_model_info(selected_model)

    if model_info:
        st.markdown(f"### {model_info['name']}")
        st.info(model_info['description'])

        # Capacités
        capabilities = []
        if model_info['multivariate']:
            capabilities.append("✅ Multivari\u00e9")
        if model_info['supports_past_covariates']:
            capabilities.append("✅ Past covariates")
        if model_info['supports_future_covariates']:
            capabilities.append("✅ Future covariates")

        if capabilities:
            st.markdown("**Capacités** : " + " | ".join(capabilities))
        else:
            st.markdown("**Capacités** : Univarié seulement")

st.markdown("---")

# ============================================================================
# CONFIGURATION DE L'ENTRAÎNEMENT
# ============================================================================
st.subheader("3️⃣ Configuration de l'entraînement")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### Horizon de prévision")
    input_chunk = st.number_input(
        "Input chunk (jours)",
        min_value=7,
        max_value=365,
        value=30,
        help=get_hyperparam_description('input_chunk_length')
    )
    output_chunk = st.number_input(
        "Output chunk (jours)",
        min_value=1,
        max_value=90,
        value=7,
        help=get_hyperparam_description('output_chunk_length')
    )

with col2:
    st.markdown("##### Splits des données")
    train_ratio = st.slider("Train (%)", 50, 90, 70) / 100
    val_ratio = st.slider("Validation (%)", 5, 30, 15) / 100
    test_ratio = 1.0 - train_ratio - val_ratio
    st.metric("Test (%)", f"{test_ratio*100:.0f}")

with col3:
    st.markdown("##### Hyperparamètres communs")
    batch_size = st.number_input(
        "Batch size",
        8, 128, 32,
        help=get_hyperparam_description('batch_size')
    )
    n_epochs = st.number_input(
        "Epochs",
        10, 200, 50,
        help=get_hyperparam_description('n_epochs')
    )
    learning_rate = st.number_input(
        "Learning rate",
        min_value=1e-5,
        max_value=1e-2,
        value=1e-3,
        format="%.5f",
        help=get_hyperparam_description('learning_rate')
    )

# Hyperparamètres spécifiques au modèle
st.markdown("##### Hyperparamètres spécifiques au modèle")

with st.expander(f"⚙️ Configurer les hyperparamètres de {selected_model}"):
    hyperparams = {
        'input_chunk_length': input_chunk,
        'output_chunk_length': output_chunk,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate
    }

    # Récupérer l'espace de recherche du modèle
    model_hyperparams_space = model_info['hyperparams']

    # Créer les inputs selon le type
    for param_name, param_space in model_hyperparams_space.items():
        # Récupérer la description
        description = get_hyperparam_description(param_name)

        if isinstance(param_space, tuple) and len(param_space) == 3:
            # (min, max, default)
            min_val, max_val, default_val = param_space

            if isinstance(min_val, int):
                hyperparams[param_name] = st.slider(
                    param_name,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    help=description
                )
            else:
                hyperparams[param_name] = st.slider(
                    param_name,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=(max_val - min_val) / 100,
                    help=description
                )

        elif isinstance(param_space, list):
            # Choix parmi une liste
            hyperparams[param_name] = st.selectbox(
                param_name,
                options=param_space,
                help=description
            )

# Sélection de la fonction de loss (pour modèles PyTorch)
st.markdown("##### Fonction de loss")
from dashboard.utils.model_factory import ModelFactory

if ModelFactory.is_torch_model(selected_model):
    loss_function = st.selectbox(
        "Loss function",
        options=['MAE', 'MSE', 'Huber', 'Quantile', 'RMSE'],
        index=0,  # MAE par défaut
        help="""
        **MAE (Mean Absolute Error)** : Erreur absolue moyenne. Robuste aux outliers, pénalise uniformément toutes les erreurs.

        **MSE (Mean Squared Error)** : Erreur quadratique moyenne. Pénalise fortement les grandes erreurs.

        **Huber Loss** : Compromis entre MAE et MSE. Quadratique pour petites erreurs, linéaire pour grandes erreurs.

        **Quantile Loss** : Pour prédictions probabilistes avec intervalles de confiance.

        **RMSE (Root Mean Squared Error)** : Racine de MSE. Même échelle que les données d'origine.
        """
    )

    # Configuration spéciale pour Quantile Loss
    if loss_function == 'Quantile':
        quantile_value = st.slider(
            "Quantile",
            min_value=0.01,
            max_value=0.99,
            value=0.5,
            step=0.01,
            help="Quantile à prédire. 0.5 = médiane, 0.1 = 10ème percentile, 0.9 = 90ème percentile"
        )
        hyperparams['loss_quantile'] = quantile_value

    # Ajouter la loss aux hyperparams
    hyperparams['loss_fn'] = loss_function
else:
    st.info("ℹ️ La sélection de fonction de loss n'est disponible que pour les modèles Deep Learning")

# Option covariables
use_covariates = False
if covariate_vars and (model_info['supports_past_covariates'] or model_info['supports_future_covariates']):
    use_covariates = st.checkbox(
        f"Utiliser les covariables ({len(covariate_vars)} disponibles)",
        value=True,
        help=f"Covariables: {', '.join(covariate_vars)}"
    )

st.markdown("---")

# ============================================================================
# ENTRAÎNEMENT
# ============================================================================
st.subheader("4️⃣ Lancer l'entraînement")

## ⚠️ Vérifications avant entraînement
# Vérifier si normalisation est activée pour les modèles deep learning
if ModelFactory.is_torch_model(selected_model):
    normalization = preprocessing_config.get('normalization', 'Aucune')
    if normalization == 'Aucune':
        st.warning("""
        ⚠️ **Attention** : Vous n'avez pas sélectionné de normalisation, mais vous utilisez un modèle deep learning.
        
        Les modèles deep learning fonctionnent **beaucoup mieux** avec des données normalisées.
        
        **Risques sans normalisation** :
        - Pertes `nan` (explosion des gradients)
        - Non-convergence
        - Mauvaises performances
        
        👉 **Recommandation** : Retournez à l'accueil et sélectionnez **StandardScaler (z-score)** ou **MinMax (0-1)**.
        """)
        
        force_normalization = st.checkbox(
            "Je comprends les risques, forcer l'entraînement sans normalisation",
            value=False
        )
        
        if not force_normalization:
            st.stop()

if st.button("🚀 Démarrer l'entraînement", type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Boucle sur les stations
        results_all_stations = {}

        for idx, station_name in enumerate(selected_stations):
            status_text.text(f"📊 Station {idx+1}/{len(selected_stations)}: {station_name}")

            # 1. Préparer les données pour cette station
            if is_multistation:
                df_station = df_raw[df_raw[station_col] == station_name].copy()
                df_station = df_station.set_index(date_col).sort_index()
                df_station = df_station[[target_var] + covariate_vars]
            else:
                df_station = df_raw.copy()

            # 1.5 Gérer les valeurs manquantes selon la config AVANT conversion Darts
            fill_method = preprocessing_config.get('fill_method', 'Supprimer les lignes')
            if fill_method == 'Supprimer les lignes':
                df_station = df_station.dropna()
            elif fill_method == 'Interpolation linéaire':
                df_station = df_station.interpolate(method='linear')
            elif fill_method == 'Forward fill':
                df_station = df_station.ffill()
            elif fill_method == 'Backward fill':
                df_station = df_station.bfill()

            # 1.8 Gérer les doublons d'index (CRITIQUE pour Darts)
            if df_station.index.duplicated().any():
                n_dupes = df_station.index.duplicated().sum()
                status_text.text(f"⚠️ {n_dupes} doublons temporels détectés : agrégation par moyenne...")
                # Faire la moyenne des valeurs pour les dates en double
                df_station = df_station.groupby(df_station.index).mean()
            
            # 2. Convertir en TimeSeries Darts
            from darts import TimeSeries
            
            # Détecter la fréquence automatiquement si possible, sinon 'D'
            inferred_freq = pd.infer_freq(df_station.index)
            freq = inferred_freq if inferred_freq else 'D'

            target_series = TimeSeries.from_dataframe(
                df_station,
                value_cols=target_var,
                freq=freq
            )

            covariates_series = None
            if use_covariates and covariate_vars:
                covariates_series = TimeSeries.from_dataframe(
                    df_station,
                    value_cols=covariate_vars,
                    freq='D'
                )

            progress_bar.progress(10 + idx * 10)

            # 3. Split train/val/test sur DONNÉES BRUTES (avant normalisation!)
            # ⚠️ IMPORTANT: Split AVANT normalisation pour éviter data leakage
            status_text.text(f"✂️ Split des données (avant normalisation)...")

            train_raw, val_raw, test_raw = split_train_val_test(
                target_series,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )

            if covariates_series is not None:
                train_cov_raw, val_cov_raw, test_cov_raw = split_train_val_test(
                    covariates_series,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio
                )
            else:
                train_cov_raw = val_cov_raw = test_cov_raw = None

            progress_bar.progress(20 + idx * 10)

            # 4. Appliquer preprocessing (FIT sur TRAIN seulement!)
            # ⚠️ CRITIQUE: On fit le scaler UNIQUEMENT sur train pour éviter data leakage
            status_text.text(f"⚙️ Normalisation (fit sur train seulement)...")

            # Target preprocessing
            target_preprocessor = TimeSeriesPreprocessor(preprocessing_config)

            # FIT + TRANSFORM sur train
            train = target_preprocessor.fit_transform(train_raw)

            # TRANSFORM SEULEMENT sur val et test (sans refitter!)
            val = target_preprocessor.transform(val_raw)
            test = target_preprocessor.transform(test_raw)

            # Covariates preprocessing (scaler séparé)
            if covariates_series is not None:
                cov_preprocessor = TimeSeriesPreprocessor(preprocessing_config)

                # FIT + TRANSFORM sur train
                train_cov = cov_preprocessor.fit_transform(train_cov_raw)

                # TRANSFORM SEULEMENT sur val et test (sans refitter!)
                val_cov = cov_preprocessor.transform(val_cov_raw)
                test_cov = cov_preprocessor.transform(test_cov_raw)

                # Créer les covariates complètes (pour prédiction) en transformant toutes les données
                # avec le scaler fitté sur train
                covariates_scaled = cov_preprocessor.transform(covariates_series)
            else:
                train_cov = val_cov = test_cov = None
                covariates_scaled = None

            st.success(f"✅ Données préparées : Train={len(train)}, Val={len(val)}, Test={len(test)}")
            st.info("ℹ️ Scaler fitté sur train uniquement (pas de data leakage)")
            
            # ⚠️ Vérification critique : NaN dans les données après preprocessing
            # Utiliser .values() pour accéder au tableau numpy sous-jacent (plus robuste)
            import numpy as np
            train_has_nan = np.isnan(train.values()).any()
            val_has_nan = np.isnan(val.values()).any()
            test_has_nan = np.isnan(test.values()).any()
            
            if train_has_nan or val_has_nan or test_has_nan:
                st.error("""
                ❌ **ERREUR** : Des valeurs NaN ont été détectées dans les données après preprocessing !
                
                Cela va causer des pertes `nan` pendant l'entraînement.
                
                **Solution** : Retournez à l'accueil et sélectionnez une méthode de gestion des valeurs manquantes
                (ex: **Interpolation linéaire** au lieu de **Supprimer les lignes**).
                """)
                st.stop()
            
            # Vérification des valeurs extrêmes (non normalisées)
            train_values = train.values()
            if train_values.max() > 1000 or train_values.min() < -1000:
                st.warning(f"""
                ⚠️ **Valeurs extrêmes détectées** : min={train_values.min():.2f}, max={train_values.max():.2f}
                
                Cela peut causer des problèmes d'entraînement. Assurez-vous que la normalisation est bien appliquée.
                """)

            progress_bar.progress(30 + idx * 10)

            # 5. Créer et entraîner le modèle
            status_text.text(f"🧠 Entraînement du modèle {selected_model} pour {station_name}...")

            from dashboard.utils.training import run_training_pipeline
            from dashboard.utils.callbacks import StreamlitProgressCallback
            from dashboard.config import CHECKPOINTS_DIR
            from dashboard.utils.model_factory import ModelFactory

            # Créer les placeholders pour le suivi en temps réel
            st.markdown("---")
            st.markdown("### 📈 Progression de l'entraînement")

            epoch_progress = st.progress(0)
            epoch_status = st.empty()
            epoch_metrics = st.empty()
            loss_chart = st.empty()

            # Créer le callback pour PyTorch Lightning
            callback = None
            pl_trainer_kwargs = None

            if ModelFactory.is_torch_model(selected_model):
                callback = StreamlitProgressCallback(
                    total_epochs=n_epochs,
                    progress_bar=epoch_progress,
                    status_text=epoch_status,
                    metrics_placeholder=epoch_metrics,
                    chart_placeholder=loss_chart
                )

                pl_trainer_kwargs = {
                    'callbacks': [callback]
                }

            # Préparer le mapping des colonnes pour sauvegarder avec noms standardisés
            col_mapping = {
                'target_var': target_var,
                'covariate_vars': covariate_vars
            }

            # Lancer l'entraînement
            training_results = run_training_pipeline(
                model_name=selected_model,
                hyperparams=hyperparams,
                train=train,
                val=val,
                test=test,
                train_cov=train_cov,
                val_cov=val_cov,
                test_cov=test_cov,
                full_cov=covariates_scaled,  # Full covariates for prediction
                use_covariates=use_covariates and (train_cov is not None),
                save_dir=CHECKPOINTS_DIR,
                station_name=station_name,
                verbose=False,
                pl_trainer_kwargs=pl_trainer_kwargs,
                station_data_df=df_station,  # Toujours sauvegarder les données
                column_mapping=col_mapping,  # Mapping pour renommer aux noms standardisés
                target_preprocessor=target_preprocessor,  # Scaler fitté sur train
                cov_preprocessor=cov_preprocessor if covariates_series is not None else None,  # Scaler fitté sur train
                original_filename=st.session_state.get('training_filename', 'unknown'),  # Fichier source
                preprocessing_config=preprocessing_config  # Config preprocessing
            )

            progress_bar.progress(90 + idx * 5)

            # Stocker les résultats
            results_all_stations[station_name] = training_results

        progress_bar.progress(100)
        status_text.text("✅ Entraînement terminé !")

        st.success(f"🎉 Entraînement terminé pour {len(selected_stations)} station(s) !")

        # Afficher résultats
        st.markdown("### 📊 Résultats")

        # Tableau récapitulatif
        summary_data = []
        for station_name, result in results_all_stations.items():
            if result['status'] == 'success':
                metrics = result.get('metrics', {})
                summary_data.append({
                    'Station': station_name,
                    'Status': '✅ Success',
                    'MAE': f"{metrics.get('MAE', 0):.4f}" if metrics.get('MAE') else 'N/A',
                    'RMSE': f"{metrics.get('RMSE', 0):.4f}" if metrics.get('RMSE') else 'N/A',
                    'R²': f"{metrics.get('R2', 0):.4f}" if metrics.get('R2') else 'N/A',
                    'Saved': '✅' if 'saved_path' in result else '❌'
                })
            else:
                summary_data.append({
                    'Station': station_name,
                    'Status': '❌ Error',
                    'MAE': 'N/A',
                    'RMSE': 'N/A',
                    'R²': 'N/A',
                    'Saved': '❌'
                })

        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

        # Détails par station
        for station_name, result in results_all_stations.items():
            with st.expander(f"📍 Détails - {station_name}"):
                if result['status'] == 'success':
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Métriques")
                        metrics = result.get('metrics', {})
                        for metric_name, metric_value in metrics.items():
                            if not pd.isna(metric_value):
                                st.metric(metric_name, f"{metric_value:.4f}")

                    with col2:
                        st.markdown("#### Info modèle")
                        st.info(f"""
                        - Modèle : {result['model_name']}
                        - Device : {'CUDA' if torch.cuda.is_available() else 'CPU'}
                        - Checkpoint : {result.get('saved_path', 'Non sauvegardé')}
                        """)

                    # Afficher les hyperparamètres
                    with st.expander("⚙️ Hyperparamètres utilisés"):
                        st.json(hyperparams)

                else:
                    st.error(f"Erreur : {result.get('error', 'Inconnue')}")
                    if 'traceback' in result:
                        st.code(result['traceback'])

        st.markdown("---")
        st.success("""
        ✅ **Entraînement terminé avec succès !**

        Les modèles ont été entraînés et sauvegardés dans `checkpoints/darts/`.

        **Prochaines étapes** :
        - Utilisez la page **Forecasting** pour faire des prédictions
        - Utilisez la page **Model Comparison** pour comparer les modèles
        - Ajoutez optimisation Optuna pour améliorer les performances
        """)

    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        import traceback
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("💡 Astuce : Commencez avec un modèle recommandé ⭐ pour des résultats rapides")
