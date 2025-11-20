"""Page d'analyse statistique."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from statsmodels.tsa.stattools import acf, pacf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.config import STATIONS, VARIABLES, VARIABLE_NAMES
from dashboard.utils.data_loader import load_station_data, load_timeseries
from dashboard.utils.statistics import (
    test_stationarity, check_seasonality_darts, stl_decomposition, normality_test
)
from dashboard.utils.plots import plot_acf_pacf, plot_stl_decomposition, plot_seasonal_patterns

st.set_page_config(page_title="Statistical Analysis", page_icon="📈", layout="wide")

st.title("📈 Statistical Analysis")
st.markdown("Tests statistiques pour l'analyse de séries temporelles.")
st.markdown("---")

# Sélecteur de station
selected_station = st.selectbox(
    "Sélectionner une station",
    options=STATIONS,
    help="Choisissez une station à analyser"
)

# Charger les données
try:
    df = load_station_data(selected_station)
    ts_data = load_timeseries(selected_station)
    st.success(f"✅ Station {selected_station} chargée ({len(df):,} échantillons).")
except Exception as e:
    st.error(f"❌ Erreur : {e}")
    st.stop()

# Onglets
tab1, tab2, tab3 = st.tabs([
    "🔍 Stationnarité",
    "📅 Saisonnalité",
    "📊 Décomposition STL"
])

with tab1:
    st.subheader("Tests de Stationnarité")

    st.info("""
    🧐 **Comprendre la Stationnarité**
    
    Une série temporelle est stationnaire si ses propriétés statistiques (moyenne, variance) ne changent pas dans le temps.
    La plupart des modèles classiques (ARIMA) requièrent la stationnarité, mais les réseaux de neurones (LSTM, N-BEATS) peuvent souvent gérer la non-stationnarité.
    
    *   **ADF (Augmented Dickey-Fuller)** : Teste la présence d'une racine unitaire.
        *   Si p-value < 0.05 ➔ On rejette H₀ ➔ La série est **Stationnaire**.
    *   **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)** : Teste l'hypothèse nulle de stationnarité.
        *   Si p-value < 0.05 ➔ On rejette H₀ ➔ La série est **Non-Stationnaire**.
        
    *Si ADF dit "Stationnaire" et KPSS dit "Non-Stationnaire", la série est probablement "Trend Stationary" (stationnaire autour d'une tendance).*
    """)

    # Tests pour toutes les variables
    results = []
    for var in VARIABLES:
        result = test_stationarity(df[var], var)
        results.append(result)

    df_stationarity = pd.DataFrame(results)

    # Affichage avec couleurs
    def highlight_result(val):
        if val == 'Stationary':
            return 'background-color: #90EE90; color: black; font-weight: bold'
        elif val == 'Non-Stationary':
            return 'background-color: #FFB6C1; color: black; font-weight: bold'
        return ''

    # Formater les p-values correctement (éviter 0.0000 et 0.1000 arrondis)
    def format_pvalue(val):
        if val < 0.0001:
            return f"{val:.2e}"  # Notation scientifique pour très petites valeurs
        elif val >= 0.9999:
            return f"{val:.4f}"
        else:
            return f"{val:.6f}"  # Plus de décimales pour les valeurs intermédiaires
    
    # Appliquer le formatage
    df_stationarity_display = df_stationarity.copy()
    df_stationarity_display['ADF p-value'] = df_stationarity_display['ADF p-value'].apply(format_pvalue)
    df_stationarity_display['KPSS p-value'] = df_stationarity_display['KPSS p-value'].apply(format_pvalue)
    
    st.dataframe(
        df_stationarity_display.style
        .format({'ADF Statistic': '{:.4f}', 'KPSS Statistic': '{:.4f}'})
        .applymap(highlight_result, subset=['ADF Result', 'KPSS Result']),
        use_container_width=True,
        hide_index=True
    )

    # ACF/PACF pour le niveau
    st.markdown("### ACF / PACF - Niveau Piézométrique")
    st.info("""
    **ACF (Autocorrelation Function)** et **PACF (Partial Autocorrelation Function)** sont des outils
    pour détecter la persistance et les décalages temporels dans la série. Interprétation :

    * **ACF** : corrélation entre la série et elle-même à différents lags (délais). Des barres qui dépassent
      les bandes rouges indiquent une corrélation significative. Si l'ACF décroît lentement, la série garde
      une mémoire longue (non-stationnaire ou tendance).
    * **PACF** : corrélation entre la série et elle-même à un lag donné, une fois les lags précédents
      retirés. Utile pour déterminer l'ordre AR (AutoRegressive) pertinent.

    ➜ Ces graphiques servent surtout à diagnostiquer la série avant d'entraîner des modèles traditionnels
    (ARIMA, etc.) ou à comprendre la dépendance temporelle de la nappe.
    """ )

    lags = st.slider("Nombre de lags", 10, 200, 100)

    series_level = df['level'].dropna()
    acf_vals = acf(series_level, nlags=lags)
    pacf_vals = pacf(series_level, nlags=lags)

    fig_acf = plot_acf_pacf(acf_vals, pacf_vals, lags=lags)
    st.plotly_chart(fig_acf, use_container_width=True)

with tab2:
    st.subheader("Détection de Saisonnalité")

    st.info("""
    Nous testons plusieurs périodes typiques (hebdo, mensuel, annuel) pour voir si la série
    présente un motif régulier. Le test Darts renvoie :

    * **Détectée** : Oui/Non selon que la saisonnalité est statistiquement significative (α = 5 %).
    * **ACF au lag** : force de la corrélation entre la série et elle-même décalée du nombre de jours indiqué.
      Plus la valeur est proche de 1, plus le motif se répète fidèlement.

    👉 Exemple d’interprétation : “Annual = Oui, ACF 0.82” ➜ le niveau se répète fortement d’une année à l’autre.
    """)

    ts_level = ts_data['target']

    # Tests de saisonnalité
    with st.spinner("Test en cours..."):
        seasonality_results = check_seasonality_darts(
            ts_level,
            periods=[7, 30, 365],
            max_lag=min(730, len(ts_level) - 1)
        )

    # Affichage
    data = []
    for period_name, result in seasonality_results.items():
        data.append({
            'Période': period_name,
            'Jours': result['period'],
            'Détectée': '✅ Oui' if result['detected'] else '❌ Non',
            'ACF au lag': f"{result['acf_at_period']:.4f}"
        })

    df_seasonality = pd.DataFrame(data)

    # Mise en forme pour rendre le tableau plus lisible
    df_seasonality_display = df_seasonality.copy()
    df_seasonality_display['ACF au lag'] = df_seasonality_display['ACF au lag'].apply(lambda x: f"{float(x):.3f}")

    st.dataframe(
        df_seasonality_display.style.applymap(
            lambda val: 'background-color: #90EE90; color: black; font-weight: bold' if val == '✅ Oui'
            else ('background-color: #FFB6C1; color: black; font-weight: bold' if val == '❌ Non' else ''),
            subset=['Détectée']
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    **Comment lire le tableau :**

    * *Détectée = Oui* ➜ la saisonnalité à cette période est confirmée.
    * *ACF au lag* ➜ intensité du motif (0 = aucun motif, 1 = motif parfait). Une valeur > 0.5 est généralement
      un signal fort.
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Visualisation des Patterns Saisonniers")
    st.info("""
    Ces graphiques permettent de voir "l'année type" de la nappe phréatique :
    
    1.  **Cycle Hydrologique Moyen** : C'est le profil moyen du niveau sur une année (de janvier à décembre).
        *   La ligne bleue montre la moyenne historique pour chaque jour.
        *   La zone ombrée montre la variabilité habituelle (écart-type).
    
    2.  **Distribution Mensuelle** : Montre la dispersion des niveaux pour chaque mois.
        *   Permet de repérer les mois de hautes eaux (recharge) et de basses eaux (étiage).
    """)
    
    fig_seasonal = plot_seasonal_patterns(df, 'level')
    st.plotly_chart(fig_seasonal, use_container_width=True)

with tab3:
    st.subheader("Décomposition STL (Seasonal-Trend-Loess)")

    st.info("""
    🧩 **Comment lire la décomposition STL ?**
    
    La série est décomposée en quatre panneaux :
    
    1.  **Original** : données brutes, pour comparer visuellement.
    2.  **Trend** : moyenne glissante à très long terme. Si elle monte/descend, la nappe a une tendance structurelle.
    3.  **Seasonal** : motif qui se répète. La plage verticale indique l’amplitude (ex. cycles annuels).
    4.  **Residual** : ce qu’il reste une fois tendance et saisonnalité retirées. Pics → anomalies potentielles.
    
    👉 Les métriques “Tendance/Saisonnalité/Résidus” sous le graphique indiquent quelle part de la variance totale
    est capturée par chaque composante (plus le pourcentage est élevé, plus cette composante explique la série).
    """)

    col1, col2 = st.columns(2)

    with col1:
        seasonal_period = st.number_input(
            "Période de saisonnalité (jours)",
            min_value=7,
            max_value=730,
            value=365
        )

    with col2:
        # Calculer une valeur par défaut qui soit impair et > seasonal
        default_trend = max(seasonal_period + 1, 3)
        if default_trend % 2 == 0:
            default_trend += 1
        
        trend_period = st.number_input(
            "Période de tendance (jours)",
            min_value=max(seasonal_period + 1, 3),
            max_value=1460,
            value=default_trend,
            step=2,  # Step de 2 pour forcer les valeurs impaires
            help="Doit être impair et supérieur à la période de saisonnalité"
        )
        
        # Ajuster automatiquement si nécessaire
        if trend_period <= seasonal_period:
            st.warning(f"⚠️ La période de tendance doit être > {seasonal_period}. Ajustement automatique.")
            trend_period = seasonal_period + 1
        if trend_period % 2 == 0:
            st.warning("⚠️ La période de tendance doit être impaire. Ajustement automatique.")
            trend_period += 1

    if st.button("🔄 Lancer la décomposition STL"):
        with st.spinner("Décomposition en cours..."):
            try:
                stl_result = stl_decomposition(
                    df['level'],
                    seasonal=seasonal_period,
                    trend=trend_period
                )

                # Graphique
                fig_stl = plot_stl_decomposition(
                    df['level'],
                    stl_result['trend'],
                    stl_result['seasonal'],
                    stl_result['residual']
                )
                st.plotly_chart(fig_stl, use_container_width=True)

                # Variance contributions
                st.markdown("### Contribution à la Variance")

                var_info = stl_result['variance']

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Tendance", f"{var_info['trend_pct']:.1f}%")
                with col2:
                    st.metric("Saisonnalité", f"{var_info['seasonal_pct']:.1f}%")
                with col3:
                    st.metric("Résidus", f"{var_info['residual_pct']:.1f}%")

            except Exception as e:
                st.error(f"Erreur lors de la décomposition : {e}")

