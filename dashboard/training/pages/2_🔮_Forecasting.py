"""Page de forecasting avec sliding window sur test set."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.config import CHECKPOINTS_DIR
from dashboard.utils.model_registry import get_registry, ModelInfo
from dashboard.utils.preprocessing import prepare_dataframe_for_darts
from darts import TimeSeries
from darts.metrics import mae, rmse, mape, r2_score, smape
import numpy as np

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")

st.title("🔮 Forecasting - Prédictions sur Test Set")
st.markdown("Générez des prédictions sur le **test set** avec une fenêtre glissante.")

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
        output_chunk = config.hyperparams.get('output_chunk_length', 'N/A')
        st.metric("Output chunk", output_chunk)

    if selected_model.metrics:
        st.markdown("**Métriques d'entraînement (sur test set):**")
        metric_cols = st.columns(len(selected_model.metrics))
        for i, (metric_name, metric_value) in enumerate(selected_model.metrics.items()):
            if metric_value is not None:
                metric_cols[i].metric(metric_name, f"{metric_value:.4f}")

    # Afficher les infos sur les données
    st.markdown("**Source des données:**")
    st.info(f"""
    - **Fichier original**: {config.data_source.get('original_file', 'N/A')}
    - **Données embarquées**: ✅ (train.csv, val.csv, test.csv)
    - **Preprocessing**: {config.preprocessing.get('fill_method', 'N/A')} / {config.preprocessing.get('scaler_type', 'N/A')}
    """)

st.markdown("---")

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

# Section de configuration temporelle
st.subheader("2️⃣ Configuration de la fenêtre de prédiction")

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

# Obtenir output_chunk_length du modèle
output_chunk_length = int(config.hyperparams.get('output_chunk_length', 30))

st.info(f"""
ℹ️ **Fenêtre de prédiction**: Le modèle prédit **{output_chunk_length} jours** à la fois.

Utilisez le slider ci-dessous pour déplacer cette fenêtre sur le **test set**.
""")

# Slider pour choisir la position de la fenêtre dans le test set
max_window_position = len(test_df) - output_chunk_length

if max_window_position < 0:
    st.error(f"⚠️ Le test set ({len(test_df)} points) est plus petit que la fenêtre de prédiction ({output_chunk_length} points)!")
    st.stop()

window_position = st.slider(
    "Position de la fenêtre dans le test set",
    min_value=0,
    max_value=max_window_position,
    value=0,
    step=1,
    help=f"Déplacez cette fenêtre pour choisir quelle partie du test set prédire (fenêtre de {output_chunk_length} jours)"
)

# Calculer la fenêtre de prédiction
forecast_start_idx = val_end_idx + window_position
forecast_end_idx = forecast_start_idx + output_chunk_length

forecast_df = df.iloc[forecast_start_idx:forecast_end_idx]

# Afficher les dates de la fenêtre
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Date de début", forecast_df.index[0].date())
with col2:
    st.metric("Date de fin", forecast_df.index[-1].date())
with col3:
    st.metric("Nombre de jours", len(forecast_df))

st.markdown("---")

# Bouton de prédiction
if st.button("🔮 Générer les prédictions", type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 1. Préparer l'historique (train + val)
        status_text.text("📊 Préparation des données...")

        # Historique = train + val (tout avant le test set)
        history_df = df.iloc[:val_end_idx]

        # Cible = la fenêtre dans le test set
        target_df = forecast_df

        history_series, _ = prepare_dataframe_for_darts(
            history_df,
            target_col=target_col,
            freq='D',
            fill_method='Interpolation linéaire'
        )

        target_series, _ = prepare_dataframe_for_darts(
            target_df,
            target_col=target_col,
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

        progress_bar.progress(50)

        # 3. Générer les prédictions
        status_text.text("🔮 Génération des prédictions...")

        n_pred = len(target_series)

        try:
            pred_series = loaded_model.predict(n=n_pred, series=history_series)
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

        metrics = {
            'MAE': float(mae_val),
            'RMSE': float(rmse_val),
            'MAPE': float(mape_val),
            'R²': float(r2_val),
            'sMAPE': float(smape_val),
            'NRMSE': float(nrmse_val),
            'Dir_Acc': float(dir_acc)
        }

        progress_bar.progress(100)
        status_text.text("✅ Prédictions générées !")

        st.success("🎉 Prédictions générées avec succès sur le test set!")

        # Afficher les métriques
        st.markdown("### 📈 Métriques de Performance")

        cols = st.columns(len(metrics))
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            cols[i].metric(metric_name, f"{metric_value:.4f}")

        st.markdown("---")

        # Graphique: Contexte + Prédictions
        st.markdown("### 📊 Visualisation avec Contexte")

        # Préparer les données pour le plot
        # Afficher les 90 derniers jours avant la fenêtre + la fenêtre
        context_days = 90
        context_start_idx = max(0, forecast_start_idx - context_days)
        context_df = df.iloc[context_start_idx:forecast_end_idx]

        pred_df = pred_series.pd_dataframe()
        target_df_plot = target_series_original.pd_dataframe()

        # Créer le graphique avec Plotly
        fig = go.Figure()

        # Contexte (historique)
        context_before_forecast = context_df.iloc[:-(len(target_df_plot))]
        if len(context_before_forecast) > 0:
            fig.add_trace(go.Scatter(
                x=context_before_forecast.index,
                y=context_before_forecast[target_col],
                mode='lines',
                name='Historique (contexte)',
                line=dict(color='gray', width=1),
                opacity=0.5,
                hovertemplate='<b>Historique</b><br>Date: %{x}<br>Valeur: %{y:.3f}<extra></extra>'
            ))

        # Vraies valeurs (test set)
        fig.add_trace(go.Scatter(
            x=target_df_plot.index,
            y=target_df_plot[target_df_plot.columns[0]],
            mode='lines+markers',
            name='Vraies valeurs (test)',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Réel</b><br>Date: %{x}<br>Valeur: %{y:.3f}<extra></extra>'
        ))

        # Prédictions
        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df[pred_df.columns[0]],
            mode='lines+markers',
            name='Prédictions',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6, symbol='x'),
            hovertemplate='<b>Prédit</b><br>Date: %{x}<br>Valeur: %{y:.3f}<extra></extra>'
        ))

        # Ajouter une zone pour délimiter le test set
        fig.add_vrect(
            x0=target_df_plot.index[0],
            x1=target_df_plot.index[-1],
            fillcolor="lightgreen",
            opacity=0.1,
            layer="below",
            line_width=0,
            annotation_text="Test Set",
            annotation_position="top left"
        )

        # Mise en forme
        full_station_name = config.original_station_id
        fig.update_layout(
            title=f"Prédictions sur Test Set - {selected_model.model_type} sur {full_station_name}",
            xaxis_title="Date",
            yaxis_title=f"{target_col}",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Graphique des erreurs
        st.markdown("### 📉 Analyse des erreurs")

        errors = pred_df[pred_df.columns[0]] - target_df_plot[target_df_plot.columns[0]]

        fig_errors = go.Figure()

        # Erreurs au fil du temps
        fig_errors.add_trace(go.Scatter(
            x=errors.index,
            y=errors.values,
            mode='lines+markers',
            name='Erreur (Prédit - Réel)',
            line=dict(color='purple', width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)',
            hovertemplate='<b>Erreur</b><br>Date: %{x}<br>Erreur: %{y:.3f}<extra></extra>'
        ))

        # Ligne zéro
        fig_errors.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig_errors.update_layout(
            title="Erreurs de prédiction au fil du temps",
            xaxis_title="Date",
            yaxis_title="Erreur",
            height=300,
            hovermode='x'
        )

        st.plotly_chart(fig_errors, use_container_width=True)

        # Statistiques des erreurs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Erreur moyenne", f"{errors.mean():.4f}")
        col2.metric("Erreur absolue moyenne", f"{errors.abs().mean():.4f}")
        col3.metric("Erreur max", f"{errors.max():.4f}")
        col4.metric("Erreur min", f"{errors.min():.4f}")

        # Export
        st.markdown("---")
        st.markdown("### 💾 Export des résultats")

        col1, col2 = st.columns(2)

        with col1:
            # Export CSV
            export_df = pd.DataFrame({
                'date': target_df_plot.index,
                'vraie_valeur': target_df_plot[target_df_plot.columns[0]].values,
                'prediction': pred_df[pred_df.columns[0]].values,
                'erreur': errors.values
            })

            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les prédictions (CSV)",
                data=csv,
                file_name=f"predictions_{full_station_name}_{selected_model.model_type}_{forecast_df.index[0].date()}.csv",
                mime="text/csv"
            )

        with col2:
            st.info(f"✅ {len(pred_df)} prédictions générées sur le test set")

    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        import traceback
        with st.expander("📋 Détails de l'erreur"):
            st.code(traceback.format_exc())
