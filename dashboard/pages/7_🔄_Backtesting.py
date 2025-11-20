"""Page de Backtesting (Validation Historique)."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.config import STATIONS, CHECKPOINTS_DIR, MODEL_COLORS
from dashboard.utils.data_loader import prepare_data_for_training
from dashboard.utils.forecasting import load_model_from_checkpoint, compute_metrics
from dashboard.utils.state import init_session_state, load_context
from dashboard.utils.plots import plot_predictions

st.set_page_config(page_title="Backtesting", page_icon="🔄", layout="wide")

st.title("🔄 Backtesting - Validation Historique")
st.markdown("""
Le backtesting permet de tester la robustesse du modèle en simulant des prédictions passées 
sur une fenêtre glissante. C'est une méthode plus fiable qu'un simple split Train/Test.
""")
st.markdown("---")

# Gestion de l'état
init_session_state()
ctx = load_context()

# Lister les checkpoints
checkpoints = list(CHECKPOINTS_DIR.glob("**/*.pth.tar"))
if not checkpoints:
    st.warning("⚠️ Aucun modèle entraîné trouvé.")
    st.stop()

checkpoint_names = [f"{c.parent.name}/{c.name}" for c in checkpoints]

# Trouver index par défaut
default_model_ix = 0
if ctx['model_path']:
    try:
        ctx_path = Path(ctx['model_path'])
        rel_name = f"{ctx_path.parent.name}/{ctx_path.name}"
        if rel_name in checkpoint_names:
            default_model_ix = checkpoint_names.index(rel_name)
    except:
        pass

default_station_ix = 0
if ctx['station'] and ctx['station'] in STATIONS:
    default_station_ix = STATIONS.index(ctx['station'])

# Configuration
col1, col2 = st.columns(2)

with col1:
    selected_checkpoint = st.selectbox(
        "Modèle à tester",
        options=checkpoint_names,
        index=default_model_ix
    )

with col2:
    station = st.selectbox(
        "Station",
        options=STATIONS,
        index=default_station_ix
    )

# Paramètres du backtest
with st.expander("⚙️ Paramètres du Backtest", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        start_ratio = st.slider("Début du test (% des données)", 0.5, 0.9, 0.7, 
                              help="Point de départ du backtest dans l'historique")
    with c2:
        forecast_horizon = st.number_input("Horizon de prévision (jours)", 1, 90, 14)
    with c3:
        stride = st.number_input("Pas (Stride) (jours)", 1, 30, 7, 
                               help="Décalage entre chaque prévision")

if st.button("🚀 Lancer le Backtest", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Charger modèle
        status_text.text("📥 Chargement du modèle...")
        checkpoint_path = CHECKPOINTS_DIR / selected_checkpoint
        model = load_model_from_checkpoint(str(checkpoint_path))
        progress_bar.progress(10)
        
        # 2. Charger données (complètes)
        status_text.text("📊 Chargement des données...")
        # On charge tout, pas de split train/test classique ici
        # Mais on a besoin des scalers, donc on utilise prepare_data quand même
        # On mettra tout dans 'train' virtuellement ou on utilisera raw
        data = prepare_data_for_training(station, fill_missing=True) # Force fill missing for backtest stability
        
        # Reconstruire la série complète normalisée
        # Darts historical_forecasts a besoin de la série complète
        # On va concaténer train/val/test scaled
        from darts import TimeSeries
        
        # Note: C'est une approx, idéalement on refit le scaler à chaque pas, 
        # mais historical_forecasts de Darts gère le retrain si retrain=True.
        # Ici on suppose un modèle pré-entraîné fixe (retrain=False) pour la vitesse.
        
        # On va utiliser les données brutes et laisser le modèle (qui a ses scalers internes si pipeline) 
        # ou normaliser manuellement.
        # Comme nos modèles attendent des données normalisées (car entraînés sur scaled),
        # il faut leur donner du scaled.
        
        full_target = data['train_scaled']['target'].append(data['val_scaled']['target']).append(data['test_scaled']['target'])
        full_cov = data['train_scaled']['covariates'].append(data['val_scaled']['covariates']).append(data['test_scaled']['covariates'])
        
        progress_bar.progress(30)
        
        # 3. Backtest
        status_text.text("🔄 Exécution du backtest (ça peut être long)...")
        
        # Déterminer le point de départ
        start_index = int(len(full_target) * start_ratio)
        start_date = full_target.time_index[start_index]
        
        model_name = selected_checkpoint.split('/')[0]
        use_cov = 'tft' in model_name.lower() or 'tcn' in model_name.lower() or 'lstm' in model_name.lower()
        
        backtest_series = model.historical_forecasts(
            series=full_target,
            past_covariates=full_cov if use_cov else None,
            start=start_date,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=False,
            verbose=True,
            last_points_only=False # On veut toute la trajectoire ? Non, historical_forecasts retourne une série concaténée des prédictions
        )
        
        progress_bar.progress(80)
        
        # 4. Métriques & Visu
        status_text.text("📊 Calcul des résultats...")
        
        # Dénormaliser
        scaler = data['scalers']['target']
        backtest_denorm = scaler.inverse_transform(backtest_series)
        
        # Vraies valeurs sur la même période
        actual_series = full_target.slice_intersect(backtest_series)
        actual_denorm = scaler.inverse_transform(actual_series)
        
        # Métriques
        metrics = compute_metrics(actual_series, backtest_series) # Calcul sur scaled pour cohérence, ou denorm ? 
        # compute_metrics gère des arrays ou series, mais les unités (MAE en mètres) dépendent de l'entrée.
        # Il vaut mieux calculer sur denorm pour avoir des mètres.
        metrics_denorm = compute_metrics(actual_denorm, backtest_denorm)
        
        progress_bar.progress(100)
        status_text.text("✅ Backtest terminé !")
        
        # Affichage
        st.success(f"Backtest terminé sur {len(backtest_series)} points")
        
        # Métriques
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("NSE", f"{metrics_denorm['NSE']:.3f}")
        c2.metric("KGE", f"{metrics_denorm['KGE']:.3f}")
        c3.metric("MAE", f"{metrics_denorm['MAE']:.3f} m")
        c4.metric("RMSE", f"{metrics_denorm['RMSE']:.3f} m")
        
        # Graphique
        st.subheader("📈 Prédictions Historiques vs Réalité")
        
        df_true = actual_denorm.pd_dataframe()
        df_pred = backtest_denorm.pd_dataframe()
        
        fig = plot_predictions(df_true, {"Backtest": df_pred}, title="Résultat du Backtest (Validation Glissante)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Navigation
        st.markdown("---")
        st.subheader("🔗 Aller plus loin")
        if st.button("💡 Analyser l'Explicabilité (SHAP)"):
            st.switch_page("pages/8_💡_Explainability.py")
            
    except Exception as e:
        st.error(f"Erreur lors du backtest : {e}")
        st.exception(e)
