"""Page de comparaison de modèles."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dashboard.config import RESULTS_DIR, METRICS
from dashboard.utils.plots import plot_metrics_comparison, plot_metrics_radar

st.set_page_config(page_title="Model Comparison", page_icon="📉", layout="wide")

st.title("📉 Model Comparison")
st.markdown("Comparez les performances de plusieurs modèles sur différentes stations.")
st.markdown("---")

# Charger les résultats
results_path = RESULTS_DIR / 'darts_comparison_complete.csv'

if not results_path.exists():
    st.warning("""
    ⚠️ Aucun fichier de résultats trouvé.

    Le fichier `results/darts_comparison_complete.csv` n'existe pas encore.

    Vous pouvez :
    1. Entraîner des modèles dans la page **Train Models**
    2. Ou exécuter le notebook `2_forecasting_models.ipynb` pour générer les résultats
    """)
    st.stop()

# Charger les données
try:
    df_results = pd.read_csv(results_path)
    st.success(f"✅ Résultats chargés : {len(df_results)} lignes.")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

# Vérifier les colonnes
required_cols = ['model', 'station'] + METRICS
missing_cols = [col for col in required_cols if col not in df_results.columns]

if missing_cols:
    st.error(f"❌ Colonnes manquantes dans le CSV : {missing_cols}")
    st.stop()

# Filtres
st.sidebar.header("🔍 Filtres")

stations_available = df_results['station'].unique().tolist()
models_available = df_results['model'].unique().tolist()

stations_filter = st.sidebar.multiselect(
    "Stations",
    options=stations_available,
    default=stations_available[:3] if len(stations_available) >= 3 else stations_available
)

models_filter = st.sidebar.multiselect(
    "Modèles",
    options=models_available,
    default=models_available
)

# Appliquer les filtres
df_filtered = df_results[
    (df_results['station'].isin(stations_filter)) &
    (df_results['model'].isin(models_filter))
]

if len(df_filtered) == 0:
    st.warning("⚠️ Aucune donnée après filtrage. Ajustez les filtres.")
    st.stop()

st.info(f"📊 {len(df_filtered)} résultats affichés après filtrage.")

# Onglets
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Tableau",
    "📊 Comparaison par Métrique",
    "🎯 Radar Chart",
    "📈 Boxplots"
])

with tab1:
    st.subheader("Tableau Complet des Résultats")

    # Afficher le tableau
    st.dataframe(
        df_filtered.style.format({metric: '{:.4f}' for metric in METRICS}),
        use_container_width=True,
        hide_index=True
    )

    # Moyennes par modèle
    st.markdown("### 📊 Moyennes par Modèle")

    avg_metrics = df_filtered.groupby('model')[METRICS].mean()

    # Highlight
    def highlight_best(s, reverse=False):
        if reverse:
            is_best = s == s.max()
        else:
            is_best = s == s.min()
        return ['background-color: #90EE90; color: black; font-weight: bold' if v else '' for v in is_best]

    st.dataframe(
        avg_metrics.style
        .format('{:.4f}')
        .apply(highlight_best, subset=['MAE', 'RMSE', 'MAPE', 'sMAPE', 'NRMSE'], axis=0)
        .apply(lambda s: highlight_best(s, reverse=True), subset=['R2', 'Dir_Acc'], axis=0),
        use_container_width=True
    )

    # Meilleur modèle par métrique
    st.markdown("### 🏆 Meilleur Modèle par Métrique")

    best_models = []
    for metric in METRICS:
        if metric in ['R2', 'Dir_Acc']:
            best_model = avg_metrics[metric].idxmax()
            best_value = avg_metrics.loc[best_model, metric]
            direction = "Higher is better"
        else:
            best_model = avg_metrics[metric].idxmin()
            best_value = avg_metrics.loc[best_model, metric]
            direction = "Lower is better"

        best_models.append({
            'Metric': metric,
            'Best Model': best_model,
            'Value': best_value,
            'Direction': direction
        })

    df_best = pd.DataFrame(best_models)

    st.dataframe(
        df_best.style.format({'Value': '{:.4f}'}),
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.subheader("Comparaison par Métrique")

    metric_to_plot = st.selectbox(
        "Sélectionner une métrique",
        options=METRICS,
        key='metric_comp'
    )

    fig = plot_metrics_comparison(df_filtered, metric=metric_to_plot)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Radar Chart Multi-Métriques")

    st.info("Les métriques sont normalisées entre 0 et 1 pour comparaison.")

    # Calculer les moyennes
    avg_metrics = df_filtered.groupby('model')[METRICS].mean()

    # Sélectionner les métriques à afficher
    selected_metrics = st.multiselect(
        "Métriques à afficher",
        options=METRICS,
        default=METRICS[:4]
    )

    if len(selected_metrics) < 3:
        st.warning("⚠️ Sélectionnez au moins 3 métriques pour le radar chart.")
    else:
        fig_radar = plot_metrics_radar(avg_metrics[selected_metrics])
        st.plotly_chart(fig_radar, use_container_width=True)

with tab4:
    st.subheader("Distribution des Métriques entre Stations")

    st.info("Boxplots montrant la variance des performances entre stations.")

    metric_box = st.selectbox(
        "Sélectionner une métrique",
        options=METRICS,
        key='metric_box'
    )

    import plotly.express as px

    fig_box = px.box(
        df_filtered,
        x='model',
        y=metric_box,
        color='model',
        title=f"Distribution de {metric_box} par Modèle"
    )

    fig_box.update_layout(
        template='plotly_white',
        height=500
    )

    st.plotly_chart(fig_box, use_container_width=True)

# Export
st.markdown("---")
st.markdown("### 💾 Export")

if st.button("📥 Télécharger les résultats filtrés (CSV)"):
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="⬇️ Télécharger",
        data=csv,
        file_name="results_filtered.csv",
        mime="text/csv"
    )
