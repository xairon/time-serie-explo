"""Fonctions de visualisation avec Plotly."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from statsmodels.tsa.seasonal import STL
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.config import VARIABLE_NAMES, MODEL_COLORS, PERFORMANCE_CONFIG

# Performance configuration
MAX_PLOT_POINTS = PERFORMANCE_CONFIG['MAX_PLOT_POINTS']


def downsample_data(df: pd.DataFrame, max_points: int = MAX_PLOT_POINTS) -> pd.DataFrame:
    """
    Downsample data if it exceeds max_points for faster plotting.
    
    Args:
        df: DataFrame to downsample
        max_points: Maximum number of points to keep
        
    Returns:
        Downsampled DataFrame
    """
    if len(df) <= max_points:
        return df
    
    # Use every nth point to reduce to approximately max_points
    step = len(df) // max_points
    return df.iloc[::step]


@st.cache_data(ttl=3600)
def plot_timeseries(dfs: dict, variables: list, title: str = "Time Series") -> go.Figure:
    """
    Graphique temporel interactif multi-séries.

    Args:
        dfs: dict {station_name: DataFrame}
        variables: Liste des variables à afficher
        title: Titre du graphique

    Returns:
        Figure Plotly
    """
    fig = go.Figure()

    # Couleurs distinctes pour chaque variable
    variable_colors = {
        'level': '#1f77b4',        # Bleu pour niveau
        'PRELIQ_Q': '#2ca02c',     # Vert pour précipitation
        'T_Q': '#ff7f0e',          # Orange pour température
        'ETP_Q': '#d62728'         # Rouge pour évapotranspiration
    }

    # Si plusieurs stations, varier l'opacité ou le style
    station_styles = [
        dict(dash='solid', width=2),
        dict(dash='dash', width=2),
        dict(dash='dot', width=2),
        dict(dash='dashdot', width=2)
    ]

    for i, (station, df) in enumerate(dfs.items()):
        # Downsample for plotting performance
        df_plot = downsample_data(df)
        
        style = station_styles[i % len(station_styles)]

        for var in variables:
            if var in df_plot.columns:
                color = variable_colors.get(var, '#666666')

                # Si une seule station, ligne pleine. Si plusieurs, varier le style
                if len(dfs) == 1:
                    line_style = dict(color=color, width=2)
                else:
                    line_style = dict(color=color, **style)

                fig.add_trace(go.Scatter(
                    x=df_plot.index,
                    y=df_plot[var],
                    name=f"{station} - {VARIABLE_NAMES.get(var, var)}",
                    mode='lines',
                    line=line_style,
                    hovertemplate=f'<b>{station}</b><br>' +
                                  f'{VARIABLE_NAMES.get(var, var)}: %{{y:.2f}}<br>' +
                                  'Date: %{x}<extra></extra>'
                ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )

    return fig


@st.cache_data(ttl=3600)
def plot_correlation_matrix(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """
    Heatmap de corrélation.

    Args:
        corr_matrix: Matrice de corrélation
        title: Titre

    Returns:
        Figure Plotly
    """
    # Remplacer les noms de colonnes
    labels = [VARIABLE_NAMES.get(col, col) for col in corr_matrix.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=500,
        width=600
    )

    return fig


@st.cache_data(ttl=3600)
def plot_acf_pacf(acf_vals: np.ndarray, pacf_vals: np.ndarray, lags: int = 100,
                  title: str = "ACF/PACF") -> go.Figure:
    """
    Graphique ACF et PACF.

    Args:
        acf_vals: Valeurs ACF
        pacf_vals: Valeurs PACF
        lags: Nombre de lags
        title: Titre

    Returns:
        Figure Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Autocorrelation (ACF)", "Partial Autocorrelation (PACF)")
    )

    # Confidence bounds
    conf_bound = 1.96 / np.sqrt(len(acf_vals))

    # ACF
    fig.add_trace(
        go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF',
               marker_color='steelblue'),
        row=1, col=1
    )
    fig.add_hline(y=conf_bound, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red", row=1, col=1)

    # PACF
    fig.add_trace(
        go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name='PACF',
               marker_color='coral'),
        row=1, col=2
    )
    fig.add_hline(y=conf_bound, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=2)

    fig.update_layout(
        title=title,
        showlegend=False,
        template='plotly_white',
        height=400
    )

    return fig


@st.cache_data(ttl=3600)
def plot_cross_correlation(lags: list, ccf: list, optimal_lag: int = None,
                           title: str = "Cross-Correlation") -> go.Figure:
    """
    Graphique de cross-correlation.

    Args:
        lags: Liste des lags
        ccf: Valeurs de corrélation croisée
        optimal_lag: Lag optimal (sera marqué en rouge)
        title: Titre

    Returns:
        Figure Plotly
    """
    fig = go.Figure()

    # Confidence bounds
    conf_bound = 1.96 / np.sqrt(len(ccf))

    # Stem plot
    for lag, corr in zip(lags, ccf):
        color = 'red' if lag == optimal_lag else 'steelblue'
        fig.add_trace(go.Scatter(
            x=[lag, lag],
            y=[0, corr],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            hovertemplate=f'Lag: {lag}<br>Correlation: {corr:.4f}<extra></extra>'
        ))

    # Markers
    fig.add_trace(go.Scatter(
        x=lags,
        y=ccf,
        mode='markers',
        marker=dict(color=['red' if l == optimal_lag else 'steelblue' for l in lags], size=6),
        showlegend=False
    ))

    # Confidence bounds
    fig.add_hline(y=conf_bound, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=-conf_bound, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=0, line_color="black", line_width=1)

    if optimal_lag is not None:
        fig.add_annotation(
            x=optimal_lag,
            y=ccf[lags.index(optimal_lag)],
            text=f"Optimal lag: {optimal_lag}",
            showarrow=True,
            arrowhead=2,
            bgcolor="red",
            font=dict(color="white")
        )

    fig.update_layout(
        title=title,
        xaxis_title="Lag (days)",
        yaxis_title="Correlation",
        template='plotly_white',
        height=500
    )

    return fig


@st.cache_data(ttl=3600)
def plot_stl_decomposition(original: pd.Series, trend: pd.Series, seasonal: pd.Series,
                           residual: pd.Series, title: str = "STL Decomposition") -> go.Figure:
    """
    Graphique de décomposition STL.

    Args:
        original: Série originale
        trend: Tendance
        seasonal: Saisonnalité
        residual: Résidus
        title: Titre

    Returns:
        Figure Plotly
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=("Original", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.05
    )

    # Original
    fig.add_trace(go.Scatter(x=original.index, y=original.values, name='Original',
                            line=dict(color='blue', width=1)),
                  row=1, col=1)

    # Trend
    fig.add_trace(go.Scatter(x=trend.index, y=trend.values, name='Trend',
                            line=dict(color='red', width=2)),
                  row=2, col=1)

    # Seasonal
    fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal.values, name='Seasonal',
                            line=dict(color='green', width=1)),
                  row=3, col=1)

    # Residual
    fig.add_trace(go.Scatter(x=residual.index, y=residual.values, name='Residual',
                            line=dict(color='purple', width=1)),
                  row=4, col=1)

    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_layout(
        title=title,
        showlegend=False,
        template='plotly_white',
        height=800
    )

    return fig


@st.cache_data(ttl=3600)
def plot_predictions(test_true_df: pd.DataFrame, predictions_dict: dict,
                     title: str = "Predictions vs Reality") -> go.Figure:
    """
    Graphique des prédictions vs réalité.

    Args:
        test_true_df: DataFrame avec les vraies valeurs (index=date, columns=['value'])
        predictions_dict: dict {model_name: DataFrame avec prédictions}
        title: Titre

    Returns:
        Figure Plotly
    """
    fig = go.Figure()

    # Ground truth
    fig.add_trace(go.Scatter(
        x=test_true_df.index,
        y=test_true_df.values.flatten(),
        name='Ground Truth',
        mode='lines',
        line=dict(color='black', width=2),
        hovertemplate='Ground Truth: %{y:.2f}<br>Date: %{x}<extra></extra>'
    ))

    # Prédictions
    for model_name, pred_df in predictions_dict.items():
        color = MODEL_COLORS.get(model_name, 'gray')
        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df.values.flatten(),
            name=model_name,
            mode='lines',
            line=dict(color=color, width=1.5),
            hovertemplate=f'{model_name}: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Water Level (m)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )

    return fig


@st.cache_data(ttl=3600)
def plot_metrics_comparison(df_metrics: pd.DataFrame, metric: str = 'MAE',
                            title: str = None) -> go.Figure:
    """
    Bar chart de comparaison de métriques.

    Args:
        df_metrics: DataFrame avec colonnes: model, metric
        metric: Nom de la métrique à afficher
        title: Titre

    Returns:
        Figure Plotly
    """
    if title is None:
        title = f"Comparison of {metric} Across Models"

    # Grouper par modèle
    avg_metrics = df_metrics.groupby('model')[metric].mean().sort_values()

    colors = [MODEL_COLORS.get(model, 'gray') for model in avg_metrics.index]

    fig = go.Figure(data=[
        go.Bar(
            x=avg_metrics.index,
            y=avg_metrics.values,
            marker_color=colors,
            text=avg_metrics.values.round(4),
            textposition='outside'
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title=metric,
        template='plotly_white',
        height=400
    )

    return fig


@st.cache_data(ttl=3600)
def plot_metrics_radar(avg_metrics: pd.DataFrame, models: list = None) -> go.Figure:
    """
    Radar chart multi-métriques.

    Args:
        avg_metrics: DataFrame avec index=models, columns=metrics
        models: Liste des modèles à afficher (None = tous)

    Returns:
        Figure Plotly
    """
    if models is not None:
        avg_metrics = avg_metrics.loc[models]

    # Normaliser entre 0 et 1 (inverse pour les métriques "lower is better")
    normalized = avg_metrics.copy()

    for col in normalized.columns:
        min_val = normalized[col].min()
        max_val = normalized[col].max()
        if max_val > min_val:
            # Normaliser entre 0 et 1
            normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
            # Inverser pour les métriques "lower is better"
            if col in ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'NRMSE']:
                normalized[col] = 1 - normalized[col]

    fig = go.Figure()

    for model in normalized.index:
        values = normalized.loc[model].values.tolist()
        values.append(values[0])  # Fermer le radar

        categories = normalized.columns.tolist()
        categories.append(categories[0])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=model,
            line=dict(color=MODEL_COLORS.get(model, 'gray'))
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="Multi-Metric Comparison (Normalized)",
        template='plotly_white',
        height=500
    )

    return fig


@st.cache_data(ttl=3600)
def plot_distributions(df: pd.DataFrame, variable: str, title: str = None) -> go.Figure:
    """
    Histogramme + density plot.

    Args:
        df: DataFrame
        variable: Nom de la colonne
        title: Titre

    Returns:
        Figure Plotly
    """
    if title is None:
        title = f"Distribution of {VARIABLE_NAMES.get(variable, variable)}"

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df[variable],
        nbinsx=50,
        name='Histogram',
        histnorm='probability density',
        marker_color='steelblue',
        opacity=0.7
    ))

    fig.update_layout(
        title=title,
        xaxis_title=VARIABLE_NAMES.get(variable, variable),
        yaxis_title="Density",
        template='plotly_white',
        height=400
    )

    return fig


@st.cache_data(ttl=3600)
def plot_monthly_boxplot(df: pd.DataFrame, variable: str, title: str = None) -> go.Figure:
    """
    Boxplot mensuel pour détecter la saisonnalité.

    Args:
        df: DataFrame avec index datetime
        variable: Colonne à afficher
        title: Titre

    Returns:
        Figure Plotly
    """
    if title is None:
        title = f"Monthly Distribution - {VARIABLE_NAMES.get(variable, variable)}"

    # Ajouter colonne mois
    df_temp = df.copy()
    df_temp['month'] = df_temp.index.month

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure()

    for month in range(1, 13):
        data = df_temp[df_temp['month'] == month][variable]
        fig.add_trace(go.Box(
            y=data,
            name=month_names[month - 1],
            marker_color='lightblue'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title=VARIABLE_NAMES.get(variable, variable),
        template='plotly_white',
        height=400
    )

    return fig


@st.cache_data(ttl=3600)
def plot_seasonal_patterns(df: pd.DataFrame, variable: str) -> go.Figure:
    """
    Graphique simplifié des patterns saisonniers (Annuel et Mensuel).
    
    Args:
        df: DataFrame avec index datetime
        variable: Variable à analyser
        
    Returns:
        Figure Plotly avec subplots
    """
    series = df[variable].dropna()
    
    # Ajouter colonnes pour l'analyse
    df_temp = df.copy()
    df_temp['day_of_year'] = df_temp.index.dayofyear
    df_temp['month'] = df_temp.index.month
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Cycle Hydrologique Moyen (Moyenne par jour de l'année)",
            "Distribution Mensuelle (Boxplots)"
        ),
        vertical_spacing=0.15
    )
    
    # 1. Pattern annuel (jour de l'année 1-365)
    # Calculer moyenne et intervalle de confiance (ou écart-type)
    annual_stats = df_temp.groupby('day_of_year')[variable].agg(['mean', 'std', 'min', 'max'])
    
    # Zone de variabilité (min-max ou std)
    fig.add_trace(go.Scatter(
        x=annual_stats.index,
        y=annual_stats['mean'] + annual_stats['std'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=annual_stats.index,
        y=annual_stats['mean'] - annual_stats['std'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 100, 255, 0.1)',
        line=dict(width=0),
        name='Variabilité (±1σ)',
        hoverinfo='skip'
    ), row=1, col=1)
    
    # Moyenne
    fig.add_trace(go.Scatter(
        x=annual_stats.index,
        y=annual_stats['mean'],
        mode='lines',
        name='Niveau Moyen',
        line=dict(color='#1f77b4', width=3)
    ), row=1, col=1)
    
    # 2. Distribution Mensuelle (Boxplots)
    month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin',
                   'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
    
    # Pour boxplot, on doit passer les données brutes, pas agrégées
    fig.add_trace(go.Box(
        x=df_temp['month'].map(lambda x: month_names[x-1]),
        y=df_temp[variable],
        name='Distribution',
        marker_color='#2ca02c',
        boxmean=True # Affiche aussi la moyenne en pointillés
    ), row=2, col=1)
    
    # Mise à jour des axes
    fig.update_xaxes(title_text="Jour de l'année", row=1, col=1)
    fig.update_xaxes(title_text="Mois", row=2, col=1)
    
    fig.update_yaxes(title_text=VARIABLE_NAMES.get(variable, variable), row=1, col=1)
    fig.update_yaxes(title_text=VARIABLE_NAMES.get(variable, variable), row=2, col=1)
    
    fig.update_layout(
        title=f"Analyse de la Saisonnalité - {VARIABLE_NAMES.get(variable, variable)}",
        template='plotly_white',
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


@st.cache_data(ttl=3600)
def plot_missing_data(df: pd.DataFrame, variable: str) -> go.Figure:
    """
    Graphique montrant les données manquantes.

    Args:
        df: DataFrame avec index temporel
        variable: Nom de la variable

    Returns:
        Figure Plotly Heatmap
    """
    # Créer un index complet de toutes les dates dans la plage
    min_date = df.index.min().normalize()  # S'assurer que c'est à minuit
    max_date = df.index.max().normalize()
    full_index = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Créer une série avec l'index complet
    # 0 = données présentes, 1 = données manquantes, NaN = hors période
    is_missing = pd.Series(index=full_index, dtype=float)
    
    # Pour chaque jour dans l'index complet
    for date in full_index:
        if date in df.index:
            # Le jour existe dans les données : vérifier si la valeur est NaN
            if pd.isna(df.loc[date, variable]):
                is_missing.loc[date] = 1  # Vraiment manquant
            else:
                is_missing.loc[date] = 0  # Présent
        else:
            # Le jour n'existe pas dans l'index : considérer comme manquant
            # (car on s'attend à des données quotidiennes)
            is_missing.loc[date] = 1
    
    # Créer une matrice pour heatmap (Année x Jour de l'année)
    years = sorted(is_missing.index.year.unique())
    matrix = []
    
    for year in years:
        year_data = is_missing[is_missing.index.year == year]
        # Créer un array de 366 jours (pour gérer les années bissextiles)
        padded = np.full(366, np.nan)  # NaN = pas de données pour ce jour
        
        for date in year_data.index:
            day_of_year = date.timetuple().tm_yday - 1  # 0-indexed
            if day_of_year < 366:
                padded[day_of_year] = year_data.loc[date]
        
        matrix.append(padded)
        
    matrix = np.array(matrix)
    
    # Couleurs : 0 = présent (gris clair), 1 = manquant (rouge), NaN = pas dans la période (blanc)
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=list(range(1, 367)),
        y=years,
        colorscale=[[0, '#90EE90'], [0.5, '#eeeeee'], [1, 'red']],  # Vert=0, Gris=0.5, Rouge=1
        zmin=0,
        zmax=1,
        showscale=True,
        colorbar=dict(title="", tickvals=[0, 1], ticktext=["Données présentes", "Données manquantes"])
    ))
    
    fig.update_layout(
        title=f"Carte des données manquantes - {VARIABLE_NAMES.get(variable, variable)}",
        xaxis_title="Jour de l'année",
        yaxis_title="Année",
        template='plotly_white',
        height=max(300, len(years) * 20)
    )
    
    return fig


@st.cache_data(ttl=3600)
def detect_behavior_changes(df: pd.DataFrame, variable: str, window: int = 365) -> dict:
    """
    Détecte les changements de comportement dans une série temporelle.
    
    Méthodes utilisées :
    - Changement de variance (rolling std)
    - Changement de moyenne (rolling mean)
    - Changement de tendance (différence de pente)
    - Score d'instabilité global
    
    Args:
        df: DataFrame avec index temporel
        variable: Variable à analyser
        window: Fenêtre glissante (jours)
        
    Returns:
        dict avec dates de changements détectés et scores
    """
    series = df[variable].dropna()
    
    if len(series) < window * 2:
        return {
            'change_points': [],
            'scores': pd.Series(),
            'message': 'Pas assez de données pour l\'analyse'
        }
    
    # 1. Variance glissante
    rolling_std = series.rolling(window=window, center=True).std()
    std_change = rolling_std.diff().abs()
    
    # 2. Moyenne glissante
    rolling_mean = series.rolling(window=window, center=True).mean()
    mean_change = rolling_mean.diff().abs()
    
    # 3. Tendance (première dérivée)
    trend = series.diff()
    rolling_trend_mean = trend.rolling(window=window, center=True).mean()
    trend_change = rolling_trend_mean.diff().abs()
    
    # 4. Score d'instabilité combiné (normalisé)
    std_norm = (std_change - std_change.min()) / (std_change.max() - std_change.min() + 1e-10)
    mean_norm = (mean_change - mean_change.min()) / (mean_change.max() - mean_change.min() + 1e-10)
    trend_norm = (trend_change - trend_change.min()) / (trend_change.max() - trend_change.min() + 1e-10)
    
    instability_score = (std_norm + mean_norm + trend_norm) / 3
    
    # Détecter les changements significatifs (percentile 95)
    threshold = instability_score.quantile(0.95)
    change_points = instability_score[instability_score > threshold].index.tolist()
    
    return {
        'change_points': change_points,
        'scores': instability_score,
        'threshold': threshold,
        'rolling_std': rolling_std,
        'rolling_mean': rolling_mean,
        'std_change': std_change,
        'mean_change': mean_change
    }


@st.cache_data(ttl=3600)
def plot_behavior_changes(df: pd.DataFrame, variable: str, window: int = 365) -> go.Figure:
    """
    Graphique montrant les changements de comportement détectés.
    
    Args:
        df: DataFrame
        variable: Variable
        window: Fenêtre glissante
        
    Returns:
        Figure Plotly avec subplots
    """
    series = df[variable].dropna()
    changes = detect_behavior_changes(df, variable, window)
    
    if 'message' in changes:
        # Pas assez de données
        fig = go.Figure()
        fig.add_annotation(
            text=changes['message'],
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(template='plotly_white', height=400)
        return fig
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Série Originale + Changements Détectés",
            "Variance et Moyenne Glissantes",
            "Score d'Instabilité"
        ),
        vertical_spacing=0.08
    )
    
    # 1. Série originale avec marqueurs de changements
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode='lines', name='Série originale',
        line=dict(color='lightblue', width=1)
    ), row=1, col=1)
    
    # Marquer les changements
    if changes['change_points']:
        change_values = series.loc[changes['change_points']]
        fig.add_trace(go.Scatter(
            x=change_values.index, y=change_values.values,
            mode='markers', name='Changements détectés',
            marker=dict(color='red', size=10, symbol='x', line=dict(width=2))
        ), row=1, col=1)
    
    # 2. Variance et moyenne glissantes
    fig.add_trace(go.Scatter(
        x=changes['rolling_std'].index,
        y=changes['rolling_std'].values,
        mode='lines', name='Écart-type glissant',
        line=dict(color='orange', width=1)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=changes['rolling_mean'].index,
        y=changes['rolling_mean'].values,
        mode='lines', name='Moyenne glissante',
        line=dict(color='green', width=1)
    ), row=2, col=1)
    
    # 3. Score d'instabilité
    fig.add_trace(go.Scatter(
        x=changes['scores'].index,
        y=changes['scores'].values,
        mode='lines', name='Score d\'instabilité',
        line=dict(color='purple', width=2),
        fill='tozeroy', fillcolor='rgba(128,0,128,0.2)'
    ), row=3, col=1)
    
    # Ligne de seuil
    fig.add_hline(
        y=changes['threshold'],
        line_dash="dash", line_color="red",
        annotation_text="Seuil (95e percentile)",
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text=VARIABLE_NAMES.get(variable, variable), row=1, col=1)
    fig.update_yaxes(title_text="Valeur", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=3, col=1)
    
    fig.update_layout(
        title=f"Détection de Changements de Comportement - {VARIABLE_NAMES.get(variable, variable)}",
        template='plotly_white',
        height=800,
        showlegend=True
    )
    
    return fig


@st.cache_data(ttl=3600)
def plot_outliers(df: pd.DataFrame, variable: str, window: int = 30, sigma: float = 3.0) -> go.Figure:
    """
    Graphique détectant les outliers avec Rolling Z-Score.

    Args:
        df: DataFrame
        variable: Variable
        window: Fenêtre glissante
        sigma: Seuil Z-score

    Returns:
        Figure Plotly
    """
    series = df[variable]
    
    # Calculer Rolling Mean & Std
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()
    
    # Détecter outliers
    lower_bound = rolling_mean - (sigma * rolling_std)
    upper_bound = rolling_mean + (sigma * rolling_std)
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    
    fig = go.Figure()
    
    # Série originale
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode='lines', name='Original',
        line=dict(color='gray', width=1)
    ))
    
    # Bounds
    fig.add_trace(go.Scatter(
        x=upper_bound.index, y=upper_bound.values,
        mode='lines', name='Upper Bound',
        line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=lower_bound.index, y=lower_bound.values,
        mode='lines', name='Lower Bound',
        line=dict(width=0), fill='tonexty',
        fillcolor='rgba(0,100,80,0.1)', showlegend=False
    ))
    
    # Outliers
    fig.add_trace(go.Scatter(
        x=outliers.index, y=outliers.values,
        mode='markers', name='Outliers',
        marker=dict(color='red', size=6, symbol='x')
    ))
    
    fig.update_layout(
        title=f"Détection d'Outliers (Rolling Z-Score, window={window}d, σ={sigma})",
        template='plotly_white',
        height=400
    )
    
    return fig


@st.cache_data(ttl=3600)
def plot_trend_and_seasonality(df: pd.DataFrame, variable: str, trend_window: int = 365) -> go.Figure:
    """
    Graphique montrant la série originale, la tendance (STL) et la composante saisonnière.

    Args:
        df: DataFrame
        variable: Variable
        trend_window: Fenêtre indicative pour le calcul (jours) – utilisée pour déterminer la période STL

    Returns:
        Figure Plotly avec subplots
    """
    series = df[variable].dropna()

    if len(series) < 30:
        fig = go.Figure()
        fig.add_annotation(text="Pas assez de données pour la décomposition", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template='plotly_white', height=400)
        return fig

    # Déterminer une période saisonnière (au moins 7 jours, au plus trend_window)
    period = max(7, min(trend_window, len(series) // 6))

    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Série Originale + Tendance", "Composante Saisonnière", "Résidus"),
        vertical_spacing=0.08
    )

    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode='lines', name='Original',
        line=dict(color='lightblue', width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=trend.index, y=trend.values,
        mode='lines', name='Tendance (STL)',
        line=dict(color='red', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=seasonal.index, y=seasonal.values,
        mode='lines', name='Saisonnalité (centrée en 0)',
        line=dict(color='green', width=1.5)
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    fig.add_trace(go.Scatter(
        x=resid.index, y=resid.values,
        mode='lines', name='Résidus',
        line=dict(color='gray', width=1)
    ), row=3, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text=VARIABLE_NAMES.get(variable, variable), row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="Résidus", row=3, col=1)

    fig.update_layout(
        title=f"Tendance et Saisonnalité (STL) - {VARIABLE_NAMES.get(variable, variable)}",
        template='plotly_white',
        height=800,
        showlegend=True
    )

    return fig