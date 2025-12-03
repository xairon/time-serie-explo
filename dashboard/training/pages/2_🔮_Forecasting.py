"""Page de forecasting sur l'intégralité du test set."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import pandas as pd
import plotly.graph_objects as go

from dashboard.config import CHECKPOINTS_DIR
from dashboard.utils.model_registry import get_registry, ModelInfo
from dashboard.utils.preprocessing import prepare_dataframe_for_darts
from darts import TimeSeries
from darts.metrics import mae, rmse, mape, r2_score, smape
import numpy as np

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")

st.title("🔮 Forecasting - Prédictions sur Test Set")
st.markdown("Générez les prédictions sur la **totalité du test set** et inspectez l'erreur au fil du temps.")

# Bouton de rafraîchissement
col_title, col_refresh = st.columns([4, 1])
with col_refresh:
    if st.button("🔄 Rafraîchir", help="Re-scanner les modèles disponibles"):
        # Vider le cache des modèles dans session_state
        keys_to_delete = [k for k in st.session_state.keys() if k.startswith("model_")]
        for key in keys_to_delete:
            del st.session_state[key]
        st.rerun()

st.markdown("---")

# Initialiser le registre de modèles
registry = get_registry()

# Scanner les modèles disponibles
all_models = registry.scan_models()

# Filtrer les modèles dont les fichiers existent encore
models = [m for m in all_models if m.model_path.exists()]

# Afficher un warning si des modèles ont été supprimés
if len(all_models) != len(models):
    removed_count = len(all_models) - len(models)
    st.info(f"ℹ️ {removed_count} modèle(s) supprimé(s) du disque ont été retirés de la liste.")

if not models:
    st.warning("""
    ⚠️ Aucun modèle entraîné trouvé.

    Veuillez d'abord entraîner un modèle dans la page **Train Models**.

    Les modèles sont recherchés dans: `{}`
    """.format(CHECKPOINTS_DIR))
    st.stop()

st.success(f"✅ **{len(models)} modèle(s)** disponible(s)")

# Grouper les modèles par station ET créer mapping vers nom complet
models_by_station = {}
station_full_names = {}  # Mapping: station courte -> nom complet

for model in models:
    if model.station not in models_by_station:
        models_by_station[model.station] = []
    models_by_station[model.station].append(model)

    # Récupérer l'original_station_id depuis la config
    if model.station not in station_full_names:
        try:
            from dashboard.utils.model_config import ModelConfig
            config_path = model.model_path.parent / "model_config.yaml"
            if config_path.exists():
                cfg = ModelConfig.load(config_path)
                station_full_names[model.station] = cfg.original_station_id
            else:
                station_full_names[model.station] = model.station
        except:
            station_full_names[model.station] = model.station

# Section de sélection
st.subheader("1️⃣ Sélection du modèle")

col1, col2 = st.columns(2)

with col1:
    # Sélection de la station avec nom complet affiché
    available_stations = sorted(models_by_station.keys())
    selected_station = st.selectbox(
        "Station",
        options=available_stations,
        format_func=lambda s: station_full_names.get(s, s),  # Affiche "02267X0030/S1"
        help="Station pour laquelle générer des prédictions"
    )

with col2:
    # Sélection du modèle pour cette station
    station_models = models_by_station[selected_station]

    # Créer des labels lisibles pour chaque modèle
    model_labels = []
    for model in station_models:
        metrics_str = ""
        if model.metrics:
            mae_val = model.metrics.get('MAE')
            rmse_val = model.metrics.get('RMSE')
            
            if mae_val is not None and isinstance(mae_val, (int, float)):
                metrics_str = f" (MAE: {mae_val:.3f}"
                if rmse_val is not None and isinstance(rmse_val, (int, float)):
                    metrics_str += f", RMSE: {rmse_val:.3f})"
                else:
                    metrics_str += ")"
            else:
                metrics_str = ""

        label = f"{model.model_type} - {model.model_name}{metrics_str}"
        model_labels.append(label)

    selected_model_idx = st.selectbox(
        "Modèle",
        options=range(len(station_models)),
        format_func=lambda i: model_labels[i],
        help="Modèle à utiliser pour les prédictions"
    )

selected_model = station_models[selected_model_idx]

# Charger le modèle et ses données embarquées
model_dir = selected_model.model_path.parent

# Utiliser session_state pour cacher manuellement le modèle
# Cela évite les conflits entre Streamlit et pickle lors de torch.load()
cache_key = f"model_{model_dir}"
results_key = f"forecast_results_{model_dir}"

if cache_key not in st.session_state:
    with st.spinner("⏳ Chargement du modèle..."):
        from dashboard.utils.model_config import load_model_with_config, load_scalers

        # Charger le modèle, sa config et ses données
        loaded_model, config, data_dict = load_model_with_config(model_dir)

        # Charger les scalers
        scalers = load_scalers(model_dir)

        # Stocker dans session_state pour éviter de recharger
        st.session_state[cache_key] = {
            'model': loaded_model,
            'config': config,
            'data_dict': data_dict,
            'scalers': scalers
        }

# Récupérer depuis session_state
cached_data = st.session_state[cache_key]
loaded_model = cached_data['model']
config = cached_data['config']
data_dict = cached_data['data_dict']
scalers = cached_data['scalers']

try:

    # Utiliser les données complètes
    if 'full' in data_dict:
        df = data_dict['full']
    else:
        # Fallback: reconstruire depuis les splits
        df = pd.concat([data_dict['train'], data_dict['val'], data_dict['test']])

    df = df.sort_index()

    st.success("✅ Modèle et données chargés avec succès depuis le dossier du modèle")

except FileNotFoundError as e:
    st.error(f"""
    ❌ **Erreur de chargement**: {e}

    Le modèle ou ses données n'ont pas été trouvés.

    **Solutions possibles:**
    - Ré-entraînez le modèle avec la nouvelle version de l'application
    - Vérifiez que le dossier `{model_dir}` contient bien les fichiers:
        - `model_config.yaml`
        - `{selected_model.station}.pkl`
        - `train.csv`, `val.csv`, `test.csv` ou `full_data.csv`
    """)
    st.stop()
except Exception as e:
    error_str = str(e)

    # Détecter l'erreur spécifique Streamlit/pickle
    if "__setstate__" in error_str or "StreamlitAPIException" in str(type(e)):
        st.error(f"""
        ❌ **Conflit Streamlit/PyTorch détecté**

        Cette erreur se produit lorsque Streamlit interfère avec le chargement de modèles PyTorch.

        **🔧 Solutions:**

        1. **Redémarrer l'application Streamlit** (Ctrl+C puis relancer)
        2. **Ré-entraîner le modèle** avec la version actuelle de l'application
        3. **Vider le cache Streamlit**: `streamlit cache clear`

        **📝 Détail technique:**
        Streamlit patch le module pickle de Python, ce qui interfère avec `torch.load()`.
        Le redémarrage de l'application résout souvent ce problème temporaire.
        """)

        with st.expander("🔍 Détails techniques de l'erreur"):
            st.code(f"""
Type d'erreur: {type(e).__name__}
Message: {error_str}

Cette erreur survient quand Streamlit intercepte les appels à __setstate__()
utilisés par pickle lors de la désérialisation de modèles PyTorch.

Chemin du modèle: {model_dir}
            """)
    else:
        st.error(f"❌ Erreur lors du chargement: {e}")
        import traceback
        with st.expander("📋 Détails de l'erreur"):
            st.code(traceback.format_exc())
    st.stop()

# Récupérer le nom de la colonne target
target_col = config.columns.get('target', 'level')
covariate_cols = config.columns.get('covariates', []) or []
use_covariates = bool(covariate_cols) and getattr(config, 'use_covariates', True)

# Récupérer les tailles des splits depuis la config
train_size = config.splits.get('train_size', 0)
val_size = config.splits.get('val_size', 0)
test_size = config.splits.get('test_size', 0)

# Calculer les indices
train_end_idx = train_size
val_end_idx = train_size + val_size

# Extraire les DataFrames
train_df = df.iloc[:train_end_idx]
val_df = df.iloc[train_end_idx:val_end_idx]
test_df = df.iloc[val_end_idx:]

# Dates
train_start = train_df.index[0]
train_end = train_df.index[-1]
val_start = val_df.index[0]
val_end = val_df.index[-1]
test_start = test_df.index[0]
test_end = test_df.index[-1]

# Afficher les informations du modèle sélectionné
with st.expander("ℹ️ Informations du modèle", expanded=False):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Type", selected_model.model_type)

    with col2:
        # Afficher le nom COMPLET de la station (original_station_id)
        full_station_name = config.original_station_id
        st.metric("Station complète", full_station_name)

    with col3:
        creation_date = pd.to_datetime(selected_model.creation_date).strftime("%Y-%m-%d %H:%M")
        st.metric("Créé le", creation_date)

with col4:
    # Afficher output_chunk_length si disponible
        output_chunk_length = int(config.hyperparams.get('output_chunk_length', 30))
        st.metric("Output chunk", output_chunk_length)

st.markdown("**Fenêtres temporelles :**")
col_dates1, col_dates2, col_dates3 = st.columns(3)
with col_dates1:
    st.metric("Train", f"{train_start.date()} → {train_end.date()}")
with col_dates2:
    st.metric("Validation", f"{val_start.date()} → {val_end.date()}")
with col_dates3:
    st.metric("Test", f"{test_start.date()} → {test_end.date()}")

    if selected_model.metrics:
        st.markdown("**Métriques d'entraînement (sur test set):**")
        metric_cols = st.columns(len(selected_model.metrics))
        for i, (metric_name, metric_value) in enumerate(selected_model.metrics.items()):
            if metric_value is not None:
                metric_cols[i].metric(metric_name, f"{metric_value:.4f}")

    if config.hyperparams:
        st.markdown("**Hyperparamètres :**")
        st.json(config.hyperparams, expanded=False)

    # Afficher les infos sur les données
    st.markdown("**Source des données:**")
    st.info(f"""
    - **Fichier original**: {config.data_source.get('original_file', 'N/A')}
    - **Données embarquées**: ✅ (train.csv, val.csv, test.csv)
    - **Preprocessing**: {config.preprocessing.get('fill_method', 'N/A')} / {config.preprocessing.get('scaler_type', 'N/A')}
    """)

st.markdown("---")

# Section de configuration temporelle
st.subheader("2️⃣ Jeu de test et historique utilisé")

# Afficher les splits
col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"""
    **📊 Train Set**
    - Du: {train_start.date()}
    - Au: {train_end.date()}
    - Taille: {len(train_df)} points
    """)

with col2:
    st.info(f"""
    **📈 Validation Set**
    - Du: {val_start.date()}
    - Au: {val_end.date()}
    - Taille: {len(val_df)} points
    """)

with col3:
    st.success(f"""
    **🎯 Test Set (prédiction)**
    - Du: {test_start.date()}
    - Au: {test_end.date()}
    - Taille: {len(test_df)} points
    """)

st.markdown("---")

st.info("""
ℹ️ **Principe** : on utilise tout l'historique _train+validation_ comme contexte et l'on prédit
en une seule fois l'intégralité du test set. La courbe ci-dessous permet ensuite de visualiser
à quels instants l'erreur explose.
""")

forecast_df = test_df

col1, col2, col3 = st.columns(3)
with col1:
    start_date = forecast_df.index[0]
    start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date)
    st.markdown(f"**Début test :** {start_str}")
with col2:
    end_date = forecast_df.index[-1]
    end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)
    st.markdown(f"**Fin test :** {end_str}")
with col3:
    st.metric("Points test", len(forecast_df))

st.markdown("---")

# Bouton de prédiction
if st.button("🔮 Générer les prédictions", type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 1. Préparer l'historique (train + val)
        status_text.text("📊 Préparation des données...")

        # Historique = train + validation (tout ce qui précède le test)
        history_df = df.iloc[:val_end_idx]
        if len(history_df) == 0:
            st.error("⚠️ Impossible de prédire sans historique (train+val).")
            st.stop()

        # Cible = l'intégralité du test set
        target_df = forecast_df

        history_series, _ = prepare_dataframe_for_darts(
            history_df,
            target_col=target_col,
            covariate_cols=covariate_cols if use_covariates else None,
            freq='D',
            fill_method='Interpolation linéaire'
        )

        target_series, _ = prepare_dataframe_for_darts(
            target_df,
            target_col=target_col,
            covariate_cols=covariate_cols if use_covariates else None,
            freq='D',
            fill_method='Interpolation linéaire'
        )

        covariates_for_prediction = None
        if use_covariates:
            cov_source_end = val_end_idx + len(forecast_df)
            _, covariates_for_prediction = prepare_dataframe_for_darts(
                df.iloc[:cov_source_end],
                target_col=target_col,
                covariate_cols=covariate_cols,
                freq='D',
                fill_method='Interpolation linéaire'
            )

        progress_bar.progress(30)

        # 2. Appliquer le preprocessing si scalers disponibles
        target_preprocessor = scalers.get('target_preprocessor')

        if target_preprocessor:
            # Appliquer le preprocessing
            history_series = target_preprocessor.transform(history_series)
            target_series = target_preprocessor.transform(target_series)

        cov_preprocessor = scalers.get('cov_preprocessor')
        if cov_preprocessor and covariates_for_prediction is not None:
            covariates_for_prediction = cov_preprocessor.transform(covariates_for_prediction)

        progress_bar.progress(50)

        # 3. Générer les prédictions
        status_text.text("🔮 Génération des prédictions...")

        n_pred = len(target_series)

        try:
            predict_kwargs = {
                'n': n_pred,
                'series': history_series
            }
            if covariates_for_prediction is not None:
                predict_kwargs['past_covariates'] = covariates_for_prediction

            pred_series = loaded_model.predict(**predict_kwargs)
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {e}")
            st.info("""
            💡 **Astuce**: Certains modèles nécessitent des covariables ou un historique spécifique.
            """)
            st.stop()

        # Dénormaliser si scaler disponible
        if target_preprocessor:
            pred_series = target_preprocessor.inverse_transform(pred_series)
            target_series_original = target_preprocessor.inverse_transform(target_series)
        else:
            target_series_original = target_series

        progress_bar.progress(70)

        # 4. Calculer les métriques
        status_text.text("📊 Calcul des métriques...")

        # S'assurer que les séries ont la même longueur
        min_len = min(len(pred_series), len(target_series_original))
        pred_series = pred_series[:min_len]
        target_series_original = target_series_original[:min_len]

        mae_val = mae(target_series_original, pred_series)
        rmse_val = rmse(target_series_original, pred_series)
        mape_val = mape(target_series_original, pred_series)
        r2_val = r2_score(target_series_original, pred_series)
        smape_val = smape(target_series_original, pred_series)

        # NRMSE
        y_range = target_series_original.values().max() - target_series_original.values().min()
        nrmse_val = rmse_val / y_range if y_range > 0 else 0

        # Directional Accuracy
        y_true_diff = np.diff(target_series_original.values().flatten())
        y_pred_diff = np.diff(pred_series.values().flatten())
        dir_acc = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100

        global_metrics = {
            'MAE': float(mae_val),
            'RMSE': float(rmse_val),
            'MAPE': float(mape_val),
            'R²': float(r2_val),
            'sMAPE': float(smape_val),
            'NRMSE': float(nrmse_val),
            'Dir_Acc': float(dir_acc)
        }

        # 5. Prédictions locales (teacher forcing)
        teacher_series = None
        teacher_windows = []
        if len(target_series_original) > 0:
            full_series = history_series.append(target_series_original)
            try:
                teacher_forecasts = loaded_model.historical_forecasts(
                    series=full_series,
                    past_covariates=covariates_for_prediction if use_covariates else None,
                    forecast_horizon=output_chunk_length,
                    stride=output_chunk_length,
                    start=history_series.end_time(),
                    retrain=False,
                    last_points_only=False,
                    verbose=False
                )

                for ts in teacher_forecasts:
                    if len(ts) == 0:
                        continue
                    teacher_windows.append(ts)

                if teacher_windows:
                    teacher_series = teacher_windows[0]
                    for ts in teacher_windows[1:]:
                        teacher_series = teacher_series.concatenate(ts)
                    teacher_series = teacher_series.slice_intersect(target_series_original)
            except Exception:
                teacher_series = None
                teacher_windows = []

        progress_bar.progress(100)
        status_text.text("✅ Prédictions générées !")

        pred_df = pred_series.to_dataframe()
        target_df_plot = target_series_original.to_dataframe()

        context_days = 120
        context_start_idx = max(0, len(df) - (len(target_df) + context_days))

        st.session_state[results_key] = {
            'pred_series': pred_series,
            'target_series': target_series_original,
            'pred_df': pred_df,
            'target_df': target_df_plot,
            'full_df': df,
            'global_metrics': global_metrics,
            'teacher_series': teacher_series,
            'teacher_windows': teacher_windows,
            'full_station_name': config.original_station_id,
            'target_col': target_col,
            'context_start_idx': context_start_idx,
            'test_start': test_start,
            'min_date': pred_df.index[0].to_pydatetime(),
            'max_date': pred_df.index[-1].to_pydatetime(),
            'window_size': output_chunk_length
        }
        st.session_state['forecast_period_range'] = (
            pred_df.index[0].to_pydatetime(),
            pred_df.index[-1].to_pydatetime()
        )

        st.success("🎉 Prédictions mises en cache. Ajustez la fenêtre mobile pour analyser les performances.")

    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        import traceback
        with st.expander("📋 Détails de l'erreur"):
            st.code(traceback.format_exc())

results = st.session_state.get(results_key)

if results:
    pred_series = results['pred_series']
    target_series_original = results['target_series']
    pred_df = results['pred_df']
    target_df_plot = results['target_df']
    full_df_plot = results.get('full_df', df)
    global_metrics = results['global_metrics']
    teacher_series = results.get('teacher_series')
    teacher_windows = results.get('teacher_windows', [])
    window_size = results.get('window_size', output_chunk_length)
    full_station_name = results['full_station_name']
    target_col = results['target_col']
    context_start_idx = results['context_start_idx']
    test_start = results['test_start']

    st.markdown("### 🪟 Fenêtre mobile (déplacement par blocs)")
    import math
    if teacher_windows:
        num_windows = len(teacher_windows)
    else:
        num_windows = max(1, math.ceil(len(target_df_plot) / max(1, window_size)))
    window_idx = st.slider(
        "Position de la fenêtre (taille = output_chunk_length)",
        min_value=0,
        max_value=max(0, num_windows - 1),
        value=0,
        step=1
    )

    if teacher_windows and window_idx < len(teacher_windows):
        current_teacher_window = teacher_windows[window_idx]
        window_start_time = current_teacher_window.start_time()
        window_end_time = current_teacher_window.end_time()
    else:
        start_idx = min(window_idx * max(1, window_size), max(0, len(target_df_plot) - max(1, window_size)))
        end_idx = min(start_idx + max(1, window_size), len(target_df_plot))
        current_teacher_window = None
        window_start_time = target_df_plot.index[start_idx]
        window_end_time = target_df_plot.index[end_idx - 1]

    teacher_df_full = teacher_series.to_dataframe() if (teacher_series is not None and len(teacher_series) > 0) else None
    window_pred_df = pred_df.loc[window_start_time:window_end_time]
    window_teacher_df = None
    if current_teacher_window is not None:
        window_teacher_df = current_teacher_window.to_dataframe()
    elif teacher_df_full is not None:
        slice_teacher = teacher_series.slice(window_start_time, window_end_time)
        if len(slice_teacher) > 0:
            window_teacher_df = slice_teacher.to_dataframe()

    period_start = window_start_time
    period_end = window_end_time

    def compute_metrics(true_ts: TimeSeries, pred_ts: TimeSeries):
        if pred_ts is None or len(pred_ts) == 0:
            return None
        start = max(true_ts.start_time(), pred_ts.start_time())
        end = min(true_ts.end_time(), pred_ts.end_time())
        if start > end:
            return None
        y_true = true_ts.slice(start, end)
        y_pred = pred_ts.slice(start, end)
        if len(y_true) == 0 or len(y_pred) == 0:
            return None
        mae_val = mae(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)
        mape_val = mape(y_true, y_pred)
        r2_val = r2_score(y_true, y_pred)
        smape_val = smape(y_true, y_pred)
        y_range = y_true.values().max() - y_true.values().min()
        nrmse_val = rmse_val / y_range if y_range > 0 else 0
        y_true_diff = np.diff(y_true.values().flatten())
        y_pred_diff = np.diff(y_pred.values().flatten())
        dir_acc = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100 if len(y_true_diff) > 0 else 0
        return {
            'MAE': float(mae_val),
            'RMSE': float(rmse_val),
            'MAPE': float(mape_val),
            'R²': float(r2_val),
            'sMAPE': float(smape_val),
            'NRMSE': float(nrmse_val),
            'Dir_Acc': float(dir_acc)
        }

    selected_target_slice = target_series_original.slice(period_start, period_end)
    selected_pred_slice = pred_series.slice(period_start, period_end)
    if len(selected_target_slice) == 0 or len(selected_pred_slice) == 0:
        selected_target_slice = target_series_original
        selected_pred_slice = pred_series
        period_start = target_df_plot.index[0]
        period_end = target_df_plot.index[-1]

    auto_selected = compute_metrics(selected_target_slice, selected_pred_slice)
    teacher_selected = None
    selected_teacher_slice = None
    if current_teacher_window is not None:
        selected_teacher_slice = current_teacher_window
        teacher_selected = compute_metrics(selected_target_slice, selected_teacher_slice)
    elif teacher_series is not None and len(teacher_series) > 0:
        selected_teacher_slice = teacher_series.slice(period_start, period_end)
        if len(selected_teacher_slice) > 0:
            teacher_selected = compute_metrics(selected_target_slice, selected_teacher_slice)

    st.caption("💡 Les métriques se mettent à jour lorsque vous déplacez la fenêtre (pas = taille des prédictions).")

    def render_metric_block(title, metrics_dict):
        st.markdown(f"_{title}_")
        cols_metrics = st.columns(len(metrics_dict) if 0 < len(metrics_dict) <= 4 else 4)
        for i, (metric_name, metric_value) in enumerate(metrics_dict.items()):
            cols_metrics[i % len(cols_metrics)].metric(metric_name, f"{metric_value:.4f}")

    # Global metrics
    st.markdown("**📈 Métriques globales**")
    col_auto_global, col_teacher_global = st.columns(2)
    with col_auto_global:
        render_metric_block("Auto-régressif", global_metrics)
    with col_teacher_global:
        teacher_global = compute_metrics(target_series_original, teacher_series) if (teacher_series is not None and len(teacher_series) > 0) else None
        if teacher_global:
            render_metric_block("Fenêtre locale (teacher)", teacher_global)
        else:
            st.info("Fenêtre locale non disponible")

    # Window metrics
    st.markdown("**🎯 Métriques sur la fenêtre sélectionnée**")
    col_auto_sel, col_teacher_sel = st.columns(2)
    with col_auto_sel:
        if auto_selected:
            render_metric_block("Auto-régressif", auto_selected)
        else:
            st.info("Auto-régressif non disponible")
    with col_teacher_sel:
        if teacher_selected:
            render_metric_block("Fenêtre locale (teacher)", teacher_selected)
        else:
            st.info("Fenêtre locale non disponible")

    st.markdown("---")

    context_df = df.iloc[context_start_idx:]

    st.markdown("### 📊 Réel vs Prédiction")

    fig = go.Figure()

    context_before_test = context_df.loc[:test_start]
    if len(context_before_test) > 0:
        fig.add_trace(go.Scatter(
            x=context_before_test.index,
            y=context_before_test[target_col],
            mode='lines',
            name='Historique',
            line=dict(color='gray', width=1),
            opacity=0.3,
            hovertemplate='<b>Historique</b><br>Date: %{x}<br>Valeur: %{y:.3f}<extra></extra>'
        ))

    fig.add_trace(go.Scatter(
        x=target_df_plot.index,
        y=target_df_plot[target_col] if target_col in target_df_plot.columns else target_df_plot[target_df_plot.columns[0]],
        mode='lines',
        name='Réel (test)',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Réel</b><br>Date: %{x}<br>Valeur: %{y:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=pred_df.index,
        y=pred_df[pred_df.columns[0]],
        mode='lines',
        name='Prédiction auto-régressive',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='<b>Auto-reg</b><br>Date: %{x}<br>Valeur: %{y:.3f}<extra></extra>'
    ))

    if not window_pred_df.empty:
        fig.add_trace(go.Scatter(
            x=window_pred_df.index,
            y=window_pred_df[window_pred_df.columns[0]],
            mode='lines+markers',
            name='Auto-reg (fenêtre)',
            line=dict(color='red', width=3),
            marker=dict(color='red', size=6),
            hovertemplate='<b>Auto-reg (fenêtre)</b><br>Date: %{x}<br>Valeur: %{y:.3f}<extra></extra>'
        ))

    if window_teacher_df is not None and not window_teacher_df.empty:
        fig.add_trace(go.Scatter(
            x=window_teacher_df.index,
            y=window_teacher_df[window_teacher_df.columns[0]],
            mode='lines+markers',
            name='Fenêtre locale (teacher)',
            line=dict(color='orange', width=3),
            marker=dict(color='orange', size=6, symbol='diamond'),
            hovertemplate='<b>Teacher (fenêtre)</b><br>Date: %{x}<br>Valeur: %{y:.3f}<extra></extra>'
        ))

    fig.add_vrect(
        x0=target_df_plot.index[0],
        x1=target_df_plot.index[-1],
        fillcolor="lightgreen",
        opacity=0.05,
        layer="below",
        line_width=0,
        annotation_text="Test set",
        annotation_position="top left"
    )
    fig.add_vrect(
        x0=period_start,
        x1=period_end,
        fillcolor="rgba(255,165,0,0.15)",
        layer="below",
        line_width=0
    )

    fig.update_layout(
        title=f"Réel vs Prédiction – {selected_model.model_type} ({full_station_name})",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    fig.update_xaxes(
        title_text="Date",
        rangeslider=dict(visible=True)
    )
    fig.update_yaxes(title_text=f"{target_col}")

    st.plotly_chart(fig, use_container_width=True)

    # Analyse des erreurs (signées)
    st.markdown("### 📉 Analyse détaillée des erreurs")

    selected_pred_df = selected_pred_slice.to_dataframe()
    selected_target_df = selected_target_slice.to_dataframe()
    target_vals = selected_target_df[target_col] if target_col in selected_target_df.columns else selected_target_df[selected_target_df.columns[0]]
    pred_vals = selected_pred_df[selected_pred_df.columns[0]]
    errors = pred_vals - target_vals

    fig_errors = go.Figure()
    fig_errors.add_trace(go.Scatter(
        x=errors.index,
        y=errors.values,
        mode='lines+markers',
        name='Erreur (prédit - réel)',
        line=dict(color='purple', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(128, 0, 128, 0.15)',
        hovertemplate='<b>Erreur</b><br>Date: %{x}<br>Erreur: %{y:.3f}<extra></extra>'
    ))
    fig_errors.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig_errors.update_layout(
        height=320,
        xaxis_title="Date",
        yaxis_title="Erreur signée",
        hovermode='x',
        xaxis=dict(rangeslider=dict(visible=True))
    )
    st.plotly_chart(fig_errors, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Erreur moyenne", f"{errors.mean():.4f}")
    col2.metric("Erreur absolue moyenne", f"{errors.abs().mean():.4f}")
    col3.metric("Erreur max", f"{errors.max():.4f}")
    col4.metric("Erreur min", f"{errors.min():.4f}")

    st.markdown("---")
    st.markdown("### 💾 Export des résultats")

    export_df = pd.DataFrame({
        'date': target_df_plot.index,
        'vraie_valeur': target_df_plot[target_df_plot.columns[0]].values,
        'prediction_auto': pred_df[pred_df.columns[0]].values
    })
    export_df['erreur_auto'] = export_df['prediction_auto'] - export_df['vraie_valeur']
    if teacher_df_full is not None and not teacher_df_full.empty:
        export_df = export_df.join(teacher_df_full.rename(columns={teacher_df_full.columns[0]: 'prediction_teacher'}), how='left')
        export_df['erreur_teacher'] = export_df['prediction_teacher'] - export_df['vraie_valeur']

    col1, col2 = st.columns(2)
    with col1:
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les prédictions (CSV)",
            data=csv,
            file_name=f"predictions_{full_station_name}_{selected_model.model_type}_{target_df_plot.index[0].date()}.csv",
            mime="text/csv"
        )

    with col2:
        st.info(f"✅ {len(pred_df)} prédictions générées sur le test set")
else:
    st.info("👉 Cliquez sur « Générer les prédictions » pour afficher les métriques et graphiques.")

