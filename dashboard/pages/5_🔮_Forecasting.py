"""Page de prédictions interactives."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.config import STATIONS, CHECKPOINTS_DIR
from dashboard.utils.data_loader import prepare_data_for_training
from dashboard.utils.forecasting import load_model_from_checkpoint, predict, compute_metrics
from dashboard.utils.plots import plot_predictions
from dashboard.utils.state import init_session_state, load_context

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")

st.title("🔮 Forecasting - Prédictions Interactives")
st.markdown("Générez des prédictions avec vos modèles entraînés.")
st.markdown("---")

# Lister les checkpoints disponibles
checkpoints = list(CHECKPOINTS_DIR.glob("**/*.pth.tar"))

if not checkpoints:
    st.warning("""
    ⚠️ Aucun modèle entraîné trouvé.

    Veuillez d'abord entraîner un modèle dans la page **Train Models**.
    """)
    st.stop()

# Préparer les noms de checkpoints
checkpoint_names = [f"{c.parent.name}/{c.name}" for c in checkpoints]

# Charger le contexte
init_session_state()
ctx = load_context()

# Trouver l'index par défaut pour le modèle
default_model_ix = 0
if ctx['model_path']:
    # Essayer de trouver le chemin dans la liste
    # ctx['model_path'] est absolu, on veut le relatif parent/nom
    try:
        ctx_path = Path(ctx['model_path'])
        rel_name = f"{ctx_path.parent.name}/{ctx_path.name}"
        if rel_name in checkpoint_names:
            default_model_ix = checkpoint_names.index(rel_name)
    except:
        pass

# Trouver l'index par défaut pour la station
default_station_ix = 0
if ctx['station'] and ctx['station'] in STATIONS:
    default_station_ix = STATIONS.index(ctx['station'])

# Sélecteurs
col1, col2 = st.columns(2)

with col1:
    selected_checkpoint = st.selectbox(
        "Modèle entraîné",
        options=checkpoint_names,
        index=default_model_ix,
        help="Sélectionnez un checkpoint à charger"
    )

with col2:
    station = st.selectbox(
        "Station",
        options=STATIONS,
        index=default_station_ix,
        help="Station pour laquelle prédire"
    )

horizon = st.slider(
    "Horizon de prédiction (jours)",
    min_value=7,
    max_value=90,
    value=14,
    help="Nombre de jours futurs à prédire"
)

# Bouton de prédiction
if st.button("🔮 Générer les prédictions", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 1. Charger le checkpoint
        status_text.text("📥 Chargement du modèle...")
        checkpoint_path = CHECKPOINTS_DIR / selected_checkpoint
        model = load_model_from_checkpoint(str(checkpoint_path))
        progress_bar.progress(20)

        # 2. Charger les données
        status_text.text("📊 Chargement des données...")
        data = prepare_data_for_training(station)
        train_data = data['train_scaled']
        test_data = data['test_scaled']
        scalers = data['scalers']
        progress_bar.progress(40)

        # 3. Prédire
        status_text.text("🔮 Génération des prédictions...")
        
        if not model._fit_called:
             st.error("❌ Erreur critique : Le modèle chargé n'a pas été entraîné (poids manquants).")
             st.stop()

        # Déterminer si on utilise les covariates
        model_name = selected_checkpoint.split('/')[0]
        use_covariates = 'tft' in model_name.lower() or 'tcn' in model_name.lower() or 'lstm' in model_name.lower()

        n_pred = len(test_data['target'])

        if use_covariates:
            pred = predict(
                model,
                train_data['target'],
                train_data['covariates'],
                n=n_pred
            )
        else:
            pred = predict(model, train_data['target'], n=n_pred)

        progress_bar.progress(70)

        # 4. Calculer les métriques
        status_text.text("📊 Calcul des métriques...")
        metrics = compute_metrics(test_data['target'], pred)
        progress_bar.progress(90)

        # 5. Dénormaliser
        pred_denorm = scalers['target'].inverse_transform(pred)
        test_denorm = scalers['target'].inverse_transform(test_data['target'])

        progress_bar.progress(100)
        status_text.text("✅ Prédictions générées !")

        st.success("🎉 Prédictions générées avec succès !")

        # Afficher les métriques
        st.markdown("### 📈 Métriques de Performance")

        cols = st.columns(4)
        for i, (metric, value) in enumerate(metrics.items()):
            if i < 4:
                cols[i].metric(metric, f"{value:.4f}")

        cols2 = st.columns(3)
        for i, (metric, value) in enumerate(list(metrics.items())[4:]):
            cols2[i].metric(metric, f"{value:.4f}")

        # Graphique complet
        st.markdown("### 📊 Prédictions vs Réalité (Test Set Complet)")

        pred_df = pred_denorm.pd_dataframe()
        test_df = test_denorm.pd_dataframe()

        model_display_name = selected_checkpoint.split('/')[0].upper()

        fig = plot_predictions(test_df, {model_display_name: pred_df})
        st.plotly_chart(fig, use_container_width=True)

        # Zoom sur les N premiers jours
        st.markdown(f"### 🔍 Zoom sur les {horizon} premiers jours")

        pred_df_zoom = pred_df.iloc[:horizon]
        test_df_zoom = test_df.iloc[:horizon]

        fig_zoom = plot_predictions(
            test_df_zoom,
            {model_display_name: pred_df_zoom},
            title=f"Prédictions - {horizon} jours"
        )
        st.plotly_chart(fig_zoom, use_container_width=True)

        # Export
        st.markdown("---")
        st.markdown("### 💾 Export des Prédictions")

        col1, col2 = st.columns(2)

        with col1:
            # Export CSV
            export_df = pred_df.copy()
            export_df.columns = ['prediction']
            export_df['ground_truth'] = test_df.values

            csv = export_df.to_csv()
            st.download_button(
                label="📥 Télécharger les prédictions (CSV)",
                data=csv,
                file_name=f"predictions_{station}_{model_display_name}.csv",
                mime="text/csv"
            )

        with col2:
            st.info(f"✅ {len(pred_df)} prédictions générées")

    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        import traceback
        st.exception(traceback.format_exc())
