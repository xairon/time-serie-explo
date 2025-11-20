"""Page d'Explicabilité (SHAP) - Version Complète."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.config import STATIONS, CHECKPOINTS_DIR, VARIABLE_NAMES
from dashboard.utils.data_loader import prepare_data_for_training
from dashboard.utils.forecasting import load_model_from_checkpoint
from dashboard.utils.state import init_session_state, load_context

st.set_page_config(page_title="Explainability", page_icon="💡", layout="wide")

st.title("💡 Explainability - Comprendre le Modèle")
st.markdown("""
Utilisez SHAP (SHapley Additive exPlanations) pour comprendre quelles variables 
influencent le plus les prédictions de votre modèle de manière **interactive**.
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
        "Modèle à analyser",
        options=checkpoint_names,
        index=default_model_ix
    )

with col2:
    station = st.selectbox(
        "Station",
        options=STATIONS,
        index=default_station_ix
    )

# Options avancées
with st.expander("⚙️ Paramètres SHAP", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        n_background = st.slider("Points de background (train)", 50, 500, 100,
                                help="Plus = plus précis mais plus lent")
        n_explain = st.slider("Points à expliquer (test)", 10, 200, 50,
                            help="Nombre de prédictions à analyser")
    with col_b:
        horizon_to_explain = st.selectbox(
            "Horizon à expliquer",
            options=[1, 3, 7, 14],
            index=2,
            help="À quel pas de prédiction calculer SHAP? (1=lendemain, 7=semaine)"
        )

if st.button("💡 Calculer les valeurs SHAP", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Charger modèle
        status_text.text("📥 Chargement du modèle...")
        checkpoint_path = CHECKPOINTS_DIR / selected_checkpoint
        model = load_model_from_checkpoint(str(checkpoint_path))
        model_name = selected_checkpoint.split('/')[0]
        model_type = None
        for m in ['nbeats', 'tft', 'tcn', 'lstm']:
            if m in model_name.lower():
                model_type = m.upper()
                break
        
        progress_bar.progress(10)
        
        # 2. Charger données
        status_text.text("📊 Chargement des données...")
        data = prepare_data_for_training(station, fill_missing=True)
        
        use_cov = model_type in ['TFT', 'TCN', 'LSTM']
        
        # Background
        background_series = data['train_scaled']['target'][-n_background:]
        background_cov = data['train_scaled']['covariates'][-n_background:] if use_cov else None
        
        # Foreground
        explain_series = data['test_scaled']['target'][:n_explain]
        explain_cov = data['test_scaled']['covariates'][:n_explain] if use_cov else None
        
        progress_bar.progress(20)
        
        # 3. Darts ShapExplainer
        status_text.text(f"🧠 Initialisation de l'Explainer pour {model_type}...")
        
        from darts.explainability import ShapExplainer
        
        explainer = ShapExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_cov
        )
        
        progress_bar.progress(40)
        
        # 4. Calculer SHAP
        status_text.text(f"💡 Calcul SHAP (horizon {horizon_to_explain})...")
        
        # Darts retourne un objet ShapExplanation
        shap_explanation = explainer.explain(
            foreground_series=explain_series,
            foreground_past_covariates=explain_cov,
            horizons=[horizon_to_explain] # On se concentre sur un horizon spécifique
        )
        
        progress_bar.progress(70)
        status_text.text("📊 Génération des visualisations...")
        
        # Récupérer les valeurs SHAP brutes
        # shap_explanation.get_shap_explanation_object(horizon=horizon_to_explain) 
        # retourne un shap.Explanation standard
        
        shap_values = shap_explanation.get_shap_explanation_object(horizon=horizon_to_explain)
        
        # Note: shap_values contient:
        # - .values : array des valeurs SHAP
        # - .base_values : prédiction de base
        # - .data : features originales
        
        progress_bar.progress(90)
        
        # ============================================================
        # VISUALISATIONS INTERACTIVES
        # ============================================================
        
        st.success(f"✅ Analyse SHAP complétée pour {n_explain} prévisions à horizon {horizon_to_explain} jours")
        
        # TAB 1: Importance Globale
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Importance Globale", 
            "🔍 Analyse Locale", 
            "🌊 Waterfall Plot",
            "📈 Dependence Plots"
        ])
        
        with tab1:
            st.subheader(f"Importance des Features (Horizon = {horizon_to_explain} jours)")
            st.info("Quelles variables impactent le plus le modèle en moyenne sur toutes les prédictions?")
            
            # Option 1: Beeswarm plot (meilleur que summary_plot classique)
            col_viz1, col_viz2 = st.columns([2, 1])
            
            with col_viz1:
                st.markdown("**Beeswarm Plot** (distribution des impacts)")
                fig_beeswarm, ax = plt.subplots(figsize=(10, 6))
                shap.plots.beeswarm(shap_values, show=False, max_display=15)
                st.pyplot(fig_beeswarm)
                plt.close()
            
            with col_viz2:
                st.markdown("**Bar Plot** (importance absolue)")
                fig_bar, ax = plt.subplots(figsize=(8, 6))
                shap.plots.bar(shap_values, show=False, max_display=10)
                st.pyplot(fig_bar)
                plt.close()
            
            # Tableau récapitulatif
            st.markdown("### 📋 Tableau d'Importance")
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            feature_names = shap_values.feature_names if hasattr(shap_values, 'feature_names') else [f"Lag {i}" for i in range(len(mean_abs_shap))]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean |SHAP|': mean_abs_shap
            }).sort_values('Mean |SHAP|', ascending=False)
            
            st.dataframe(importance_df, use_container_width=True)
        
        with tab2:
            st.subheader("Explication Locale - À un instant précis")
            st.info("Sélectionnez un point temporel pour voir comment chaque feature a contribué à cette prédiction.")
            
            # Slider pour choisir le point
            point_idx = st.slider(
                "Sélectionner un point à expliquer",
                0, len(explain_series) - 1, 0,
                help=f"Point 0 = {explain_series.time_index[0]}"
            )
            
            point_date = explain_series.time_index[point_idx]
            st.markdown(f"**Date analysée:** {point_date}")
            
            # Force plot (HTML interactif)
            st.markdown("**Force Plot** (contributions cumulatives)")
            try:
                import streamlit.components.v1 as components
                force_plot_html = shap.plots.force(
                    shap_values[point_idx],
                    matplotlib=False
                )
                components.html(shap.getjs() + force_plot_html.html(), height=100)
            except:
                # Fallback si streamlit components ne marche pas
                fig_force, ax = plt.subplots(figsize=(12, 3))
                shap.plots.force(shap_values[point_idx], matplotlib=True, show=False)
                st.pyplot(fig_force)
                plt.close()
        
        with tab3:
            st.subheader("Waterfall Plot - Décomposition Détaillée")
            st.info("Visualisez comment chaque feature pousse la prédiction vers le haut ou vers le bas.")
            
            # Sélection du point (réutiliser le slider de tab2 ou nouveau)
            point_waterfall = st.slider(
                "Point pour Waterfall",
                0, len(explain_series) - 1, 0,
                key="waterfall_slider"
            )
            
            fig_waterfall, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(shap_values[point_waterfall], show=False, max_display=15)
            st.pyplot(fig_waterfall)
            plt.close()
            
            st.caption(f"Date: {explain_series.time_index[point_waterfall]}")
        
        with tab4:
            st.subheader("Dependence Plots - Impact selon la valeur")
            st.info("Comment l'impact d'une feature change selon sa valeur?")
            
            # Sélectionner la feature à analyser
            if len(mean_abs_shap) > 0:
                top_features = importance_df['Feature'].head(10).tolist()
                selected_feature = st.selectbox("Feature à analyser", top_features)
                
                feature_idx = feature_names.index(selected_feature)
                
                fig_dep, ax = plt.subplots(figsize=(10, 6))
                shap.plots.scatter(shap_values[:, feature_idx], show=False)
                st.pyplot(fig_dep)
                plt.close()
            else:
                st.warning("Pas assez de features pour un dependence plot.")
        
        # ============================================================
        # Résumé quantitatif
        # ============================================================
        st.markdown("---")
        st.subheader("📊 Résumé Quantitatif")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Points analysés", n_explain)
        with col_s2:
            st.metric("Features", len(mean_abs_shap))
        with col_s3:
            top_feature_name = importance_df.iloc[0]['Feature']
            st.metric("Feature #1", top_feature_name)
        
        progress_bar.progress(100)
        status_text.text("✅ Analyse complète !")
        
    except Exception as e:
        st.error(f"Erreur lors de l'explicabilité : {e}")
        st.warning(f"""
        **Dépannage:**
        - Pour **N-BEATS**: SHAP peut échouer (architecture très profonde). Essayez TFT/TCN/LSTM.
        - Pour **TFT**: Assurez-vous que les covariates sont présentes.
        - Réduisez `n_background` et `n_explain` si problème de mémoire.
        
        **Type de modèle détecté:** {model_type if 'model_type' in locals() else 'Unknown'}
        """)
        st.exception(e)
