"""Page d'accueil du dashboard Junon."""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.config import STATIONS, VARIABLES, MODELS, METRICS, DEVICE
from dashboard.utils.data_loader import get_all_stations_summary

# Configuration de la page
st.set_page_config(
    page_title="Junon Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🌊 Junon Dashboard - Analyse Piézométrique")
st.markdown("---")

# Sidebar
st.sidebar.title("🧭 Navigation")
st.sidebar.info(f"📍 **{len(STATIONS)} stations** piézométriques françaises")
st.sidebar.markdown(f"💻 Device: **{DEVICE.upper()}**")
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Pages disponibles
- 📊 **Data Explorer** : Visualisation des données
- 📈 **Statistical Analysis** : Tests statistiques
- 🔗 **Correlations** : Analyse des corrélations
- 🎯 **Train Models** : Entraînement + Optuna
- 🔮 **Forecasting** : Prédictions interactives
- 📉 **Model Comparison** : Comparaison des modèles
- 🔄 **Backtesting** : Validation historique
- 💡 **Explainability** : Analyse SHAP
""")

# Métriques en haut
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Stations",
        value=len(STATIONS),
        help="Nombre de stations piézométriques disponibles"
    )

with col2:
    st.metric(
        label="Variables",
        value=len(VARIABLES),
        help="Niveau + 3 covariables (précipitation, température, ETP)"
    )

with col3:
    st.metric(
        label="Modèles",
        value=len(MODELS),
        help="N-BEATS, TFT, TCN, LSTM"
    )

with col4:
    st.metric(
        label="Métriques",
        value=len(METRICS),
        help="MAE, RMSE, R², NSE, KGE, sMAPE, NRMSE, Dir_Acc, MAPE"
    )

st.markdown("---")

# Section : Fonctionnalités
st.subheader("🚀 Fonctionnalités du Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.info("""
        **📊 Data Explorer**

        Visualisez les séries temporelles des 18 stations avec des graphiques interactifs Plotly.
        Sélectionnez plusieurs stations et variables simultanément.
        """)

    with st.container():
        st.info("""
        **📈 Statistical Analysis**

        Tests de stationnarité (ADF, KPSS), détection de saisonnalité, décomposition STL,
        ACF/PACF, tests de normalité.
        """)
    
    with st.container():
        st.info("""
        **🔗 Correlations**

        Matrices de corrélation, cross-correlation, tests de causalité de Granger,
        analyse des lags optimaux.
        """)

with col2:
    with st.container():
        st.info("""
        **🎯 Train Models**

        Entraînez des modèles deep learning (N-BEATS, TFT, TCN, LSTM) avec hyperparamètres
        personnalisés ou optimisation automatique Optuna.
        """)

    with st.container():
        st.info("""
        **🔮 Forecasting**

        Générez des prédictions avec vos modèles entraînés. Visualisez les résultats
        et calculez les métriques de performance (incluant NSE et KGE).
        """)
    
    with st.container():
        st.info("""
        **📉 Model Comparison**

        Comparez les performances de plusieurs modèles sur différentes stations.
        Tableaux, graphiques, radar charts.
        """)

with col3:
    with st.container():
        st.info("""
        **🔄 Backtesting**

        Validation historique robuste via fenêtre glissante. Testez la stabilité
        de vos modèles sur le passé.
        """)
    
    with st.container():
        st.info("""
        **💡 Explainability**

        Comprenez pourquoi votre modèle prédit ce qu'il prédit grâce à SHAP.
        Importance globale et explications locales.
        """)

st.markdown("---")

# Section : Résumé des stations
st.subheader("📊 Résumé des Stations Disponibles")

try:
    df_summary = get_all_stations_summary()

    # Afficher le tableau
    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True
    )

    # Statistiques globales
    col1, col2, col3 = st.columns(3)

    total_samples = df_summary['Samples'].sum()
    avg_duration = df_summary['Duration (years)'].astype(float).mean()

    with col1:
        st.metric("Total d'échantillons", f"{total_samples:,}")
    with col2:
        st.metric("Durée moyenne", f"{avg_duration:.1f} ans")
    with col3:
        st.metric("Plage temporelle", "~30 ans")

except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")

# Footer
st.markdown("---")
st.caption("⚡ Powered by Darts | PyTorch Lightning | Streamlit | Optuna | SHAP")
st.caption("📧 Pour toute question, consultez le README du projet")
