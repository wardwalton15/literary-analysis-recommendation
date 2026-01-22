"""Reusable chart components for the dashboard."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional
import pandas as pd


def create_gauge_chart(
    value: float,
    title: str,
    min_val: float = 0,
    max_val: float = 100,
    suffix: str = "",
    color_scale: str = "blues"
) -> go.Figure:
    """Create a gauge chart for a single metric."""
    # Determine color based on value position in range
    normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5

    if color_scale == "sentiment":
        # Red for negative, green for positive
        if value < 0:
            bar_color = f"rgb({int(255 * abs(value))}, 100, 100)"
        else:
            bar_color = f"rgb(100, {int(155 + 100 * value)}, 100)"
    elif color_scale == "blues":
        bar_color = f"rgb(50, {int(100 + 155 * normalized)}, {int(200 + 55 * normalized)})"
    else:
        bar_color = "rgb(100, 150, 200)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        number={"suffix": suffix, "font": {"size": 20}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickfont": {"size": 10}},
            "bar": {"color": bar_color},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "gray",
            "steps": [
                {"range": [min_val, min_val + (max_val - min_val) * 0.33], "color": "rgba(200, 200, 200, 0.3)"},
                {"range": [min_val + (max_val - min_val) * 0.33, min_val + (max_val - min_val) * 0.66], "color": "rgba(200, 200, 200, 0.2)"},
                {"range": [min_val + (max_val - min_val) * 0.66, max_val], "color": "rgba(200, 200, 200, 0.1)"},
            ],
        }
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def create_emotional_arc_chart(arc_data: List[Dict[str, Any]], title: str = "Emotional Arc") -> go.Figure:
    """Create an emotional arc line chart."""
    if not arc_data:
        return go.Figure()

    df = pd.DataFrame(arc_data)

    fig = go.Figure()

    # Main sentiment line
    fig.add_trace(go.Scatter(
        x=df['position'],
        y=df['sentiment'],
        mode='lines',
        name='Sentiment',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate="Position: %{x}%<br>Sentiment: %{y:.3f}<extra></extra>"
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Narrative Progress (%)",
        yaxis_title="Sentiment Score",
        hovermode="x unified",
        showlegend=False,
        height=400,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def create_pie_chart(
    labels: List[str],
    values: List[float],
    title: str,
    colors: Optional[List[str]] = None
) -> go.Figure:
    """Create a pie chart."""
    if colors is None:
        colors = px.colors.qualitative.Set2

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors[:len(labels)],
        textinfo='label+percent',
        textposition='outside',
        hovertemplate="%{label}: %{value}<br>(%{percent})<extra></extra>"
    )])

    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=350,
        margin=dict(l=20, r=20, t=50, b=50),
    )

    return fig


def create_bar_chart(
    x: List[str],
    y: List[float],
    title: str,
    x_label: str = "",
    y_label: str = "",
    color: str = "#1f77b4",
    horizontal: bool = False
) -> go.Figure:
    """Create a bar chart."""
    if horizontal:
        fig = go.Figure(go.Bar(
            x=y,
            y=x,
            orientation='h',
            marker_color=color,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>"
        ))
    else:
        fig = go.Figure(go.Bar(
            x=x,
            y=y,
            marker_color=color,
            hovertemplate="%{x}: %{y:.2f}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label if not horizontal else y_label,
        yaxis_title=y_label if not horizontal else x_label,
        height=400,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def create_grouped_bar_chart(
    categories: List[str],
    data_series: Dict[str, List[float]],
    title: str,
    y_label: str = ""
) -> go.Figure:
    """Create a grouped bar chart for comparisons."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, (name, values) in enumerate(data_series.items()):
        fig.add_trace(go.Bar(
            name=name,
            x=categories,
            y=values,
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=y_label,
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=50),
    )

    return fig


def create_radar_chart(
    categories: List[str],
    values: List[float],
    title: str,
    fill_color: str = "rgba(31, 119, 180, 0.5)"
) -> go.Figure:
    """Create a radar/spider chart."""
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor=fill_color,
        line=dict(color='rgb(31, 119, 180)', width=2),
        name=title
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1] if values else [0, 1]
            )
        ),
        showlegend=False,
        title=title,
        height=400,
        margin=dict(l=60, r=60, t=80, b=60),
    )

    return fig


def create_heatmap(
    x_labels: List[str],
    y_labels: List[str],
    values: List[List[float]],
    title: str,
    color_scale: str = "Blues"
) -> go.Figure:
    """Create a heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=x_labels,
        y=y_labels,
        colorscale=color_scale,
        hoverongaps=False,
        hovertemplate="Topic: %{y}<br>Work: %{x}<br>Value: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        height=400,
        margin=dict(l=100, r=20, t=50, b=50),
    )

    return fig


def create_word_cloud_placeholder(words: List[Dict[str, Any]], title: str = "Word Cloud") -> go.Figure:
    """
    Create a scatter plot visualization of words (simulating word cloud).
    Uses word frequency to determine size.
    """
    if not words:
        return go.Figure()

    import random
    random.seed(42)

    # Sort by weight/frequency
    sorted_words = sorted(words, key=lambda x: x.get('weight', x.get('frequency', 1)), reverse=True)[:30]

    # Create positions
    x_positions = [random.uniform(-1, 1) for _ in sorted_words]
    y_positions = [random.uniform(-1, 1) for _ in sorted_words]
    sizes = [w.get('weight', w.get('frequency', 1)) * 50 for w in sorted_words]
    texts = [w.get('word', w.get('term', '')) for w in sorted_words]

    # Normalize sizes
    max_size = max(sizes) if sizes else 1
    sizes = [s / max_size * 40 + 10 for s in sizes]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_positions,
        y=y_positions,
        mode='text',
        text=texts,
        textfont=dict(
            size=sizes,
            color=[f'rgb({random.randint(50, 150)}, {random.randint(50, 150)}, {random.randint(100, 200)})' for _ in texts]
        ),
        hoverinfo='text',
        hovertext=[f"{t}: {w.get('weight', w.get('frequency', 0)):.3f}" for t, w in zip(texts, sorted_words)]
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def create_histogram(
    values: List[float],
    title: str,
    x_label: str = "",
    bins: int = 20,
    color: str = "#1f77b4"
) -> go.Figure:
    """Create a histogram."""
    fig = go.Figure(go.Histogram(
        x=values,
        nbinsx=bins,
        marker_color=color,
        opacity=0.75,
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Count",
        height=300,
        margin=dict(l=60, r=20, t=50, b=50),
        bargap=0.1,
    )

    return fig


def create_dual_line_chart(
    data1: List[Dict[str, Any]],
    data2: List[Dict[str, Any]],
    label1: str,
    label2: str,
    title: str
) -> go.Figure:
    """Create a dual-line chart for comparison."""
    fig = go.Figure()

    if data1:
        df1 = pd.DataFrame(data1)
        fig.add_trace(go.Scatter(
            x=df1['position'],
            y=df1['sentiment'],
            mode='lines',
            name=label1,
            line=dict(color='#1f77b4', width=2),
        ))

    if data2:
        df2 = pd.DataFrame(data2)
        fig.add_trace(go.Scatter(
            x=df2['position'],
            y=df2['sentiment'],
            mode='lines',
            name=label2,
            line=dict(color='#ff7f0e', width=2),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Narrative Progress (%)",
        yaxis_title="Sentiment Score",
        hovermode="x unified",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=50),
    )

    return fig
