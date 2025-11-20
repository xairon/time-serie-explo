"""Page d'exploration des données."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.config import STATIONS, VARIABLES, VARIABLE_NAMES
from dashboard.utils.data_loader import load_station_data
from dashboard.utils.plots import (
    plot_timeseries, plot_distributions, plot_monthly_boxplot,
    plot_missing_data, plot_outliers, plot_trend_and_seasonality
)

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

st.title("📊 Data Explorer")
st.markdown("Explorez les données brutes des stations piézométriques.")
st.markdown("---")

# Sélecteurs
col1, col2 = st.columns(2)

with col1:
    selected_stations = st.multiselect(
        "Sélectionner les stations",
        options=STATIONS,
        default=['piezo1'],
        help="Choisissez une ou plusieurs stations à visualiser"
    )

with col2:
    selected_variables = st.multiselect(
        "Sélectionner les variables",
        options=VARIABLES,
        default=['level'],
        help="Choisissez les variables à afficher"
    )

if not selected_stations:
    st.warning("⚠️ Veuillez sélectionner au moins une station.")
    st.stop()

if not selected_variables:
    st.warning("⚠️ Veuillez sélectionner au moins une variable.")
    st.stop()

# Charger les données
try:
    dfs = {}
    for station in selected_stations:
        dfs[station] = load_station_data(station)

    st.success(f"✅ {len(dfs)} station(s) chargée(s) avec succès.")

except Exception as e:
    st.error(f"❌ Erreur lors du chargement des données : {e}")
    st.stop()

# Onglets
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Séries Temporelles", 
    "🕵️ Analyse de la Qualité",
    "📊 Statistiques", 
    "📉 Distributions"
])

with tab1:
    st.subheader("Séries Temporelles")

    st.info("""
    👋 **Pourquoi explorer les séries temporelles ?**
    
    Avant toute modélisation, il est crucial de comprendre vos données visuellement.
    
    *   **Trous (gaps)** : Les périodes sans données (lignes plates ou interruptions) peuvent fausser l'entraînement.
    *   **Saisonnalité** : Observez-vous des cycles réguliers (annuels, saisonniers) ?
    *   **Tendances** : Le niveau moyen monte-t-il ou descend-il sur le long terme ?
    *   **Outliers** : Y a-t-il des pics ou des creux aberrants qui semblent être des erreurs de mesure ?
    """)

    # Graphique principal
    try:
        fig = plot_timeseries(dfs, selected_variables, title="Évolution Temporelle")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique : {e}")

    # Date range slider
    if len(selected_stations) == 1:
        station = selected_stations[0]
        df = dfs[station]

        st.markdown("### 🔍 Zoom sur une période")

        min_date = df.index.min().to_pydatetime()
        max_date = df.index.max().to_pydatetime()

        date_range = st.date_input(
            "Sélectionner une plage de dates",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df.loc[start_date:end_date]

            fig_zoom = plot_timeseries(
                {station: df_filtered},
                selected_variables,
                title=f"Zoom : {start_date} → {end_date}"
            )
            st.plotly_chart(fig_zoom, use_container_width=True)

with tab2:
    st.subheader("🕵️ Analyse de la Qualité des Données")
    
    if len(selected_stations) == 1:
        station = selected_stations[0]
        df = dfs[station]
        
        var_quality = st.selectbox(
            "Variable à analyser", 
            options=VARIABLES,  # Toutes les variables disponibles, pas seulement celles sélectionnées
            key='quality_var',
            format_func=lambda x: VARIABLE_NAMES.get(x, x)
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🕳️ Données Manquantes (Gaps)")
            st.info("Le rouge indique les jours où aucune donnée n'est disponible.")
            fig_missing = plot_missing_data(df, var_quality)
            st.plotly_chart(fig_missing, use_container_width=True)
            
        with col2:
            st.markdown("### 🚨 Détection d'Outliers")
            st.info("Points s'écartant de plus de 3 écarts-types de la moyenne glissante (30j).")
            fig_outliers = plot_outliers(df, var_quality, window=30, sigma=3.0)
            st.plotly_chart(fig_outliers, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📈 Tendance et Saisonnalité")
        st.info("""
        **Tendance** : Évolution de fond à long terme (moyenne mobile sur 1 an).
        **Saisonnalité** : Écart à la tendance, révélant les cycles annuels.
        """)
        fig_trend = plot_trend_and_seasonality(df, var_quality, trend_window=365)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🔄 Détection de Changements de Comportement")
        st.info("""
        Cette analyse détecte les périodes où la série change de comportement.
        """)
        st.warning("Analyse en cours d'amélioration – fonctionnalité temporairement désactivée.")
            
    else:
        st.info("📌 Veuillez sélectionner une seule station pour l'analyse détaillée de la qualité.")

with tab3:
    st.subheader("Statistiques Descriptives")

    st.info("""
    📊 **Interprétation des statistiques**
    
    *   **Mean (Moyenne) / Std (Écart-type)** : Donnent une idée du niveau moyen et de la variabilité.
    *   **Min / Max** : Permettent de repérer les valeurs extrêmes.
    *   **Missing** : Le nombre de données manquantes. Si ce chiffre est élevé, la fiabilité des modèles sera impactée.
    """)

    for station in selected_stations:
        with st.expander(f"📍 {station}", expanded=(len(selected_stations) == 1)):
            df = dfs[station]

            # Statistiques
            stats = df[selected_variables].describe().T
            stats['missing'] = df[selected_variables].isnull().sum()

            st.dataframe(
                stats.style.format("{:.2f}").highlight_max(axis=0, props='background-color: #90EE90; color: black; font-weight: bold'),
                use_container_width=True
            )

            # Métriques clés
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Échantillons", f"{len(df):,}")
            with col2:
                st.metric("Début", df.index.min().strftime('%Y-%m-%d'))
            with col3:
                st.metric("Fin", df.index.max().strftime('%Y-%m-%d'))
            with col4:
                duration_years = (df.index.max() - df.index.min()).days / 365.25
                st.metric("Durée", f"{duration_years:.1f} ans")

with tab4:
    st.subheader("Distributions des Variables")

    if len(selected_stations) == 1:
        station = selected_stations[0]
        df = dfs[station]

        # Sélecteur de variable
        var_to_plot = st.selectbox(
            "Variable à analyser",
            options=selected_variables,
            format_func=lambda x: VARIABLE_NAMES.get(x, x)
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Histogramme")
            fig_dist = plot_distributions(df, var_to_plot)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            st.markdown("#### Boxplot Mensuel")
            fig_box = plot_monthly_boxplot(df, var_to_plot)
            st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.info("📌 Sélectionnez une seule station pour voir les distributions détaillées.")

# Export
st.markdown("---")
st.subheader("💾 Export des Données")

if st.button("📥 Télécharger les données sélectionnées (CSV)"):
    for station in selected_stations:
        csv = dfs[station][selected_variables].to_csv()
        st.download_button(
            label=f"⬇️ {station}.csv",
            data=csv,
            file_name=f"{station}_export.csv",
            mime="text/csv"
        )
