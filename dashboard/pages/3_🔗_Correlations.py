"""Page d'analyse des corrélations."""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.config import STATIONS, VARIABLE_NAMES
from dashboard.utils.data_loader import load_station_data
from dashboard.utils.statistics import (
    cross_correlation, granger_causality_test, calculate_lagged_correlations
)
from dashboard.utils.plots import plot_correlation_matrix, plot_cross_correlation

st.set_page_config(page_title="Correlations", page_icon="🔗", layout="wide")

st.title("🔗 Analyse des Corrélations")
st.markdown("Corrélations, cross-correlation et causalité de Granger.")
st.markdown("---")

# Sélecteur de station
selected_station = st.selectbox(
    "Sélectionner une station",
    options=STATIONS
)

# Charger les données
try:
    df = load_station_data(selected_station)
    st.success(f"✅ Station {selected_station} chargée.")
except Exception as e:
    st.error(f"❌ Erreur : {e}")
    st.stop()

# Onglets
tab1, tab2, tab3 = st.tabs([
    "📊 Matrice de Corrélation",
    "🔄 Cross-Correlation",
    "⚡ Causalité de Granger"
])

with tab1:
    st.subheader("Matrice de Corrélation")

    st.info("""
    🔗 **Corrélation de Pearson (linéaire instantanée)**
    
    Ce tableau montre si les variables varient ensemble **au même moment**.
    
    *   **Proche de +1** : Quand l'une monte, l'autre monte (ex: Pluie et Niveau ? Pas forcément instantanément).
    *   **Proche de -1** : Quand l'une monte, l'autre descend (ex: ETP et Niveau en été).
    *   **Proche de 0** : Pas de lien linéaire direct **instantané**.
    
    *Attention : En hydrologie, la pluie met du temps à atteindre la nappe (infiltration). Une faible corrélation ici ne veut pas dire qu'il n'y a pas de lien, mais que le lien est peut-être décalé dans le temps (voir onglet Cross-Correlation).*
    """)

    # Calculer la matrice
    corr_matrix = df[['level', 'PRELIQ_Q', 'T_Q', 'ETP_Q']].corr()

    # Afficher
    fig_corr = plot_correlation_matrix(corr_matrix)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Tableau
    st.markdown("### Coefficients de Corrélation")

    st.dataframe(
        corr_matrix.style
        .format("{:.3f}")
        .background_gradient(cmap='RdBu', vmin=-1, vmax=1),
        use_container_width=True
    )

    st.info("""
    **Interprétation** :
    - Proche de +1 : corrélation positive forte
    - Proche de 0 : pas de corrélation linéaire
    - Proche de -1 : corrélation négative forte
    """)

with tab2:
    st.subheader("Cross-Correlation : Précipitation → Niveau")

    st.info("""
    ⏳ **L'effet retard (Lag)**
    
    L'eau met du temps à s'infiltrer dans le sol. La **Cross-Correlation** cherche le décalage (lag) qui maximise la corrélation.
    
    *   **Lag Optimal (positif)** : Nombre de jours avant que la pluie n'atteigne la nappe (Pluie → Niveau).
    *   **Lag Négatif ?** : Si le lag optimal est négatif, cela signifie mathématiquement que le Niveau précède la Pluie, ce qui n'est pas physique. Cela peut indiquer une saisonnalité complexe, une autre influence ou un artefact de calcul.
    *   **Lag 0** : Effet immédiat (ou résolution journalière insuffisante pour voir le délai).
    """)

    max_lag = st.slider(
        "Lag maximum (jours)",
        min_value=10,
        max_value=90,
        value=60
    )

    with st.spinner("Calcul en cours..."):
        lags, ccf = cross_correlation(
            df['PRELIQ_Q'].values,
            df['level'].values,
            max_lag=max_lag
        )

        # Lag optimal
        optimal_lag_idx = np.argmax(np.abs(ccf))
        optimal_lag = lags[optimal_lag_idx]
        optimal_corr = ccf[optimal_lag_idx]

    # Métriques
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Lag Optimal", f"{optimal_lag} jours")
    with col2:
        st.metric("Corrélation Maximale", f"{optimal_corr:.4f}")
    with col3:
        if optimal_lag > 0:
            direction = "PRELIQ → Niveau"
        elif optimal_lag < 0:
            direction = "Niveau → PRELIQ"
        else:
            direction = "Simultané"
        st.metric("Direction", direction)

    # Graphique
    fig_ccf = plot_cross_correlation(lags, ccf, optimal_lag)
    st.plotly_chart(fig_ccf, use_container_width=True)

    # Analyse des lags pour toutes les covariables
    st.markdown("### Lag Optimal par Covariable")

    lag_results = []
    for cov in ['PRELIQ_Q', 'T_Q', 'ETP_Q']:
        lags_cov, corrs_cov, opt_lag, opt_corr = calculate_lagged_correlations(
            df, 'level', cov, max_lag=max_lag
        )
        lag_results.append({
            'Covariable': VARIABLE_NAMES.get(cov, cov),
            'Lag Optimal (jours)': opt_lag,
            'Corrélation Maximale': opt_corr
        })

    df_lags = pd.DataFrame(lag_results)

    st.dataframe(
        df_lags.style
        .format({'Lag Optimal (jours)': '{:.0f}', 'Corrélation Maximale': '{:.4f}'})
        .background_gradient(subset=['Corrélation Maximale'], cmap='RdYlGn', vmin=-1, vmax=1),
        use_container_width=True,
        hide_index=True
    )

with tab3:
    st.subheader("Tests de Causalité de Granger")

    st.info("""
    ⚡ **Causalité Prédictive (Granger)**
    
    Le test de Granger ne vérifie pas une causalité physique "réelle", mais une **causalité prédictive**.
    
    *   La question posée est : *"Est-ce que connaître le passé de la Pluie aide à mieux prédire le futur du Niveau que si on ne connaissait que le passé du Niveau ?"*
    *   **p-value < 0.05** : Oui, la variable est utile pour la prédiction. C'est un bon candidat pour être utilisé comme "covariate" dans vos modèles.
    """)

    max_lag_granger = st.slider(
        "Lag maximum pour Granger",
        min_value=5,
        max_value=30,
        value=20
    )

    # Tests pour chaque covariable
    granger_results = []

    for cov in ['PRELIQ_Q', 'T_Q', 'ETP_Q']:
        st.markdown(f"#### {VARIABLE_NAMES.get(cov, cov)} → Niveau")

        with st.spinner(f"Test en cours pour {cov}..."):
            gc_result = granger_causality_test(df, 'level', cov, max_lag=max_lag_granger)

        if 'error' in gc_result:
            st.error(f"Erreur : {gc_result['error']}")
            continue

        lags_gc = gc_result['lags']
        pvalues_gc = gc_result['pvalues']
        sig_lags = gc_result['significant_lags']

        # Graphique
        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=lags_gc,
            y=pvalues_gc,
            mode='lines+markers',
            name='p-value',
            line=dict(color='steelblue', width=2),
            marker=dict(size=6)
        ))

        fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                     annotation_text="α = 0.05")

        fig.update_layout(
            title=f"{VARIABLE_NAMES.get(cov, cov)} → Niveau",
            xaxis_title="Lag (jours)",
            yaxis_title="P-value",
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Résumé
        if sig_lags:
            st.success(f"✅ Causalité détectée ! Lags significatifs : {', '.join(map(str, sig_lags[:10]))}")
        else:
            st.warning(f"❌ Pas de causalité détectée.")

        granger_results.append({
            'Covariable': VARIABLE_NAMES.get(cov, cov),
            'Lags Significatifs': len(sig_lags),
            'Meilleur Lag': sig_lags[0] if sig_lags else 'N/A',
            'Causalité': '✅ Oui' if sig_lags else '❌ Non'
        })

    # Résumé
    st.markdown("### Résumé de la Causalité de Granger")

    df_granger = pd.DataFrame(granger_results)

    st.dataframe(df_granger, use_container_width=True, hide_index=True)
