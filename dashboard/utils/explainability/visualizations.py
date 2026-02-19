"""Plotly visualizations for explainability results.

Includes:
- Feature importance bar charts
- Temporal saliency heatmaps
- Attention heatmaps
- SHAP waterfall and force plots
- Decomposition comparison plots
- Residual analysis plots
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_feature_importance_bar(
    importance: Dict[str, float],
    title: str = "Feature Importance",
    top_k: Optional[int] = None,
    show_values: bool = True,
    color_scale: str = "Blues"
) -> go.Figure:
    """
    Create horizontal bar chart of feature importance.

    Args:
        importance: Dictionary mapping feature names to importance values
        title: Plot title
        top_k: Show only top k features (None = show all)
        show_values: Whether to show value labels
        color_scale: Plotly color scale name

    Returns:
        Plotly figure
    """
    if not importance:
        fig = go.Figure()
        fig.add_annotation(text="No importance data", x=0.5, y=0.5, showarrow=False)
        return fig

    # Sort by importance
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    if top_k:
        sorted_items = sorted_items[:top_k]

    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Create color scale
    max_val = max(values) if values and max(values) > 0 else 1
    colors = [f'rgba(31, 119, 180, {0.3 + 0.7 * v / max_val})' for v in values]

    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=values,
            orientation='h',
            marker_color=colors,
            text=[f'{v:.1%}' if v < 1 else f'{v:.3f}' for v in values] if show_values else None,
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Relative Importance",
        yaxis_title="",
        height=max(300, len(features) * 35),
        margin=dict(l=10, r=80, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(autorange="reversed")
    )

    return fig


def plot_temporal_saliency_heatmap(
    attributions: np.ndarray,
    feature_names: List[str],
    title: str = "Feature x Time Attribution",
    input_chunk_length: Optional[int] = None
) -> go.Figure:
    """
    Create heatmap of attributions (feature × time).

    Args:
        attributions: Attribution matrix (seq_len × n_features)
        feature_names: List of feature names
        title: Plot title
        input_chunk_length: Input window size for x-axis labels

    Returns:
        Plotly figure
    """
    if attributions is None or attributions.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="No attribution data", x=0.5, y=0.5, showarrow=False)
        return fig

    seq_len, n_features = attributions.shape

    # Transpose for heatmap (features on y-axis, time on x-axis)
    z_data = attributions.T

    # Create labels
    if input_chunk_length:
        x_labels = [f"t-{seq_len - i}" for i in range(seq_len)]
    else:
        x_labels = [f"{i}" for i in range(seq_len)]

    y_labels = feature_names[:n_features] if len(feature_names) >= n_features else \
               feature_names + [f"Feature {i}" for i in range(len(feature_names), n_features)]

    # Symmetric color scale
    max_abs = max(abs(np.nanmin(z_data)), abs(np.nanmax(z_data)), 0.001)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale='RdBu_r',
        zmin=-max_abs,
        zmax=max_abs,
        colorbar=dict(title="Attribution"),
        hovertemplate='Time: %{x}<br>Feature: %{y}<br>Attribution: %{z:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Timestep",
        yaxis_title="Feature",
        height=max(400, len(y_labels) * 25),
        margin=dict(l=10, r=10, t=50, b=30),
    )

    return fig


def plot_attention_heatmap(
    attention: np.ndarray,
    title: str = "Attention Weights",
    x_label: str = "Horizon Step",
    y_label: str = "Input Timestep"
) -> go.Figure:
    """
    Create heatmap of attention weights.

    Args:
        attention: Attention matrix (input_steps × horizon_steps)
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label

    Returns:
        Plotly figure
    """
    if attention is None or attention.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="No attention data", x=0.5, y=0.5, showarrow=False)
        return fig

    # Handle multi-head attention (average across heads)
    if attention.ndim == 3:
        attention = attention.mean(axis=0)

    input_steps, horizon_steps = attention.shape

    x_labels = [f"h+{i+1}" for i in range(horizon_steps)]
    y_labels = [f"t-{input_steps - i}" for i in range(input_steps)]

    fig = go.Figure(data=go.Heatmap(
        z=attention,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        colorbar=dict(title="Attention"),
        hovertemplate='Input: %{y}<br>Horizon: %{x}<br>Attention: %{z:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=max(400, input_steps * 15),
        margin=dict(l=10, r=10, t=50, b=30),
    )

    return fig


def plot_shap_waterfall(
    feature_importance: Dict[str, float],
    base_value: float = 0.0,
    prediction: Optional[float] = None,
    title: str = "SHAP Waterfall"
) -> go.Figure:
    """
    Create SHAP waterfall plot showing cumulative feature contributions.

    Uses Plotly's native Waterfall trace for correct rendering.

    Args:
        feature_importance: Dict mapping features to signed SHAP values
        base_value: Expected value E[f(x)]
        prediction: Final prediction value
        title: Plot title

    Returns:
        Plotly figure
    """
    if not feature_importance:
        fig = go.Figure()
        fig.add_annotation(text="No SHAP data", x=0.5, y=0.5, showarrow=False)
        return fig

    # Sort by absolute value (top contributors first)
    sorted_items = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    # Limit to top 15 for readability
    if len(sorted_items) > 15:
        top_items = sorted_items[:14]
        rest_val = sum(v for _, v in sorted_items[14:])
        top_items.append(("Other features", rest_val))
        sorted_items = top_items

    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    final_prediction = prediction if prediction is not None else base_value + sum(values)

    # Build Waterfall trace
    x_labels = [f"Base ({base_value:.3f})"] + features + [f"Prediction ({final_prediction:.3f})"]
    measures = ["absolute"] + ["relative"] * len(features) + ["total"]
    y_values = [base_value] + values + [0]  # total is auto-computed
    text_labels = [f"{base_value:.3f}"] + [f"{v:+.4f}" for v in values] + [f"{final_prediction:.3f}"]

    # Colors: green for positive, red for negative
    colors = ["rgba(128,128,128,0.7)"]  # base
    for v in values:
        colors.append("rgba(44,160,44,0.8)" if v >= 0 else "rgba(214,39,40,0.8)")
    colors.append("rgba(31,119,180,0.8)")  # prediction

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        text=text_labels,
        textposition="outside",
        textfont=dict(size=10),
        connector=dict(line=dict(color="rgba(100,100,100,0.3)", width=1)),
        increasing=dict(marker=dict(color="rgba(44,160,44,0.8)")),
        decreasing=dict(marker=dict(color="rgba(214,39,40,0.8)")),
        totals=dict(marker=dict(color="rgba(31,119,180,0.8)")),
        hovertemplate='<b>%{x}</b><br>Value: %{text}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title="Value",
        height=max(400, 350),
        margin=dict(l=10, r=10, t=50, b=100),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickangle=-45),
        showlegend=False,
    )

    return fig


def plot_shap_force(
    feature_importance: Dict[str, float],
    base_value: float = 0.0,
    prediction: Optional[float] = None,
    title: str = "Feature Contributions"
) -> go.Figure:
    """
    Create force-style plot with arrows showing contributions.

    Args:
        feature_importance: Dict mapping features to signed SHAP values
        base_value: Expected value
        prediction: Final prediction
        title: Plot title

    Returns:
        Plotly figure
    """
    if not feature_importance:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    # Separate positive and negative contributions
    positive = {k: v for k, v in feature_importance.items() if v > 0}
    negative = {k: v for k, v in feature_importance.items() if v < 0}

    # Sort each group
    pos_sorted = sorted(positive.items(), key=lambda x: x[1], reverse=True)
    neg_sorted = sorted(negative.items(), key=lambda x: x[1])

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Increasing Factors", "Decreasing Factors"),
                       vertical_spacing=0.15)

    # Positive contributions
    if pos_sorted:
        fig.add_trace(go.Bar(
            y=[p[0] for p in pos_sorted],
            x=[p[1] for p in pos_sorted],
            orientation='h',
            marker_color='#2ca02c',
            text=[f'+{p[1]:.3f}' for p in pos_sorted],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Contribution: +%{x:.4f}<extra></extra>'
        ), row=1, col=1)

    # Negative contributions
    if neg_sorted:
        fig.add_trace(go.Bar(
            y=[n[0] for n in neg_sorted],
            x=[abs(n[1]) for n in neg_sorted],
            orientation='h',
            marker_color='#d62728',
            text=[f'{n[1]:.3f}' for n in neg_sorted],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Contribution: %{customdata:.4f}<extra></extra>',
            customdata=[n[1] for n in neg_sorted]
        ), row=2, col=1)

    final_pred = prediction if prediction is not None else base_value + sum(feature_importance.values())

    fig.update_layout(
        title=dict(text=f"{title}<br><sub>Base: {base_value:.3f} → Prediction: {final_pred:.3f}</sub>",
                  font=dict(size=14)),
        height=max(400, (len(positive) + len(negative)) * 30 + 150),
        margin=dict(l=10, r=80, t=80, b=30),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def plot_decomposition_comparison(
    actual_decomp: Dict[str, Any],
    predicted_decomp: Dict[str, Any],
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Decomposition: Actual vs Predicted"
) -> go.Figure:
    """
    Create comparison plot of STL decomposition for actual vs predicted.

    Args:
        actual_decomp: Actual series decomposition (from DecompositionAnalyzer)
        predicted_decomp: Predicted series decomposition
        dates: Optional datetime index for x-axis
        title: Plot title

    Returns:
        Plotly figure with subplots for trend, seasonal, residual
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Trend", "Seasonal", "Residual"),
        vertical_spacing=0.08,
        shared_xaxes=True
    )

    # Get common index
    if dates is not None:
        x_axis = dates
    else:
        x_axis = actual_decomp.get("trend", pd.Series()).index

    # Trend comparison
    if "trend" in actual_decomp:
        fig.add_trace(go.Scatter(
            x=x_axis, y=actual_decomp["trend"],
            mode='lines', name='Actual Trend',
            line=dict(color='#2E86AB', width=2)
        ), row=1, col=1)

    if "trend" in predicted_decomp:
        fig.add_trace(go.Scatter(
            x=x_axis, y=predicted_decomp["trend"],
            mode='lines', name='Predicted Trend',
            line=dict(color='#F24236', width=2, dash='dash')
        ), row=1, col=1)

    # Seasonal comparison
    if "seasonal" in actual_decomp:
        fig.add_trace(go.Scatter(
            x=x_axis, y=actual_decomp["seasonal"],
            mode='lines', name='Actual Seasonal',
            line=dict(color='#2E86AB', width=1.5)
        ), row=2, col=1)

    if "seasonal" in predicted_decomp:
        fig.add_trace(go.Scatter(
            x=x_axis, y=predicted_decomp["seasonal"],
            mode='lines', name='Predicted Seasonal',
            line=dict(color='#F24236', width=1.5, dash='dash')
        ), row=2, col=1)

    # Residual comparison
    if "residual" in actual_decomp:
        fig.add_trace(go.Scatter(
            x=x_axis, y=actual_decomp["residual"],
            mode='lines', name='Actual Residual',
            line=dict(color='#2E86AB', width=1)
        ), row=3, col=1)

    if "residual" in predicted_decomp:
        fig.add_trace(go.Scatter(
            x=x_axis, y=predicted_decomp["residual"],
            mode='lines', name='Predicted Residual',
            line=dict(color='#F24236', width=1, dash='dash')
        ), row=3, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=600,
        margin=dict(l=10, r=10, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def plot_lag_importance(
    lag_importance: Dict[int, float],
    input_chunk_length: int = 30,
    title: str = "Temporal Importance (Autocorrelation)"
) -> go.Figure:
    """
    Create line chart of lag importance.

    Args:
        lag_importance: Dict mapping lag -> correlation value
        input_chunk_length: Input window size for highlighting
        title: Plot title

    Returns:
        Plotly figure
    """
    if not lag_importance:
        fig = go.Figure()
        fig.add_annotation(text="No lag data", x=0.5, y=0.5, showarrow=False)
        return fig

    lags = list(lag_importance.keys())
    values = list(lag_importance.values())

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=lags,
        y=values,
        mode='lines+markers',
        name='Autocorrelation',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)',
        hovertemplate='Lag: t-%{x}<br>Correlation: %{y:.3f}<extra></extra>'
    ))

    # Highlight input window
    fig.add_vrect(
        x0=0, x1=input_chunk_length,
        fillcolor="rgba(255, 200, 0, 0.1)",
        layer="below", line_width=0,
        annotation_text="Input window", annotation_position="top left"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Lag (days)",
        yaxis_title="Absolute Correlation",
        height=350,
        margin=dict(l=10, r=10, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def plot_residual_analysis(
    residuals: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Residual Analysis"
) -> go.Figure:
    """
    Create combined residual analysis plot (histogram + timeline).

    Args:
        residuals: Array of residual values
        dates: Optional datetime index for timeline
        title: Plot title

    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribution", "Over Time"),
        column_widths=[0.4, 0.6]
    )

    # Histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        name='Residuals',
        marker_color='rgba(31, 119, 180, 0.7)',
        hovertemplate='Error: %{x:.3f}<br>Count: %{y}<extra></extra>'
    ), row=1, col=1)

    # Mean line on histogram
    mean_val = np.mean(residuals)
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2, row=1, col=1)
    fig.add_vline(x=mean_val, line_dash="dot", line_color="green", line_width=2, row=1, col=1)

    # Timeline
    x_axis = dates if dates is not None else list(range(len(residuals)))
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=residuals,
        mode='lines',
        name='Residuals',
        line=dict(color='#2E86AB', width=1),
        hovertemplate='Error: %{y:.3f}<extra></extra>'
    ), row=1, col=2)

    # ±1 std band
    std = np.std(residuals)
    fig.add_hrect(
        y0=-std, y1=std,
        fillcolor="rgba(0, 255, 0, 0.1)",
        layer="below", line_width=0,
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=350,
        margin=dict(l=10, r=10, t=60, b=30),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_xaxes(title_text="Error", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=1, col=2)

    return fig


def plot_seasonality_patterns(
    detection_results: Dict[str, Any],
    title: str = "Seasonality Detection"
) -> go.Figure:
    """
    Create bar chart of detected seasonality patterns.

    Args:
        detection_results: Results from detect_seasonality_patterns
        title: Plot title

    Returns:
        Plotly figure
    """
    if not detection_results:
        fig = go.Figure()
        fig.add_annotation(text="No seasonality data", x=0.5, y=0.5, showarrow=False)
        return fig

    periods = []
    acf_values = []
    detected = []

    for period_name, result in detection_results.items():
        if "acf_at_period" in result:
            periods.append(period_name.capitalize())
            acf_values.append(result["acf_at_period"])
            detected.append(result.get("detected", False))

    colors = ['#2ca02c' if d else '#d62728' for d in detected]

    fig = go.Figure(data=[
        go.Bar(
            x=periods,
            y=acf_values,
            marker_color=colors,
            text=[f'{v:.3f}' for v in acf_values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>ACF: %{y:.4f}<br>Detected: %{customdata}<extra></extra>',
            customdata=['Yes' if d else 'No' for d in detected]
        )
    ])

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Period",
        yaxis_title="ACF at Period",
        height=350,
        margin=dict(l=10, r=10, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Add significance threshold line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    return fig


def plot_prediction_vs_actual(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Predictions vs Actual"
) -> go.Figure:
    """
    Create comparison line plot.

    Args:
        dates: Datetime index
        actual: Actual values
        predicted: Predicted values
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        mode='lines', name='Actual',
        line=dict(color='#2E86AB', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=predicted,
        mode='lines', name='Predicted',
        line=dict(color='#F24236', width=2)
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="Value",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig
