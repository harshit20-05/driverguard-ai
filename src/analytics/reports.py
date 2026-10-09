"""
Plotly Chart Generators and Data Visualizations for Streamlit.
Dark theme customized charts for fatigue trends, eye state, and performance latency.
Accepts list of telemetry dictionaries for robust zero-dependency execution.
"""

from typing import Dict, Any, Optional, List
import plotly.graph_objects as go


DARK_TEMPLATE = "plotly_dark"
COLOR_MAP = {
    "background": "#0F172A",
    "surface": "#1E293B",
    "primary": "#3B82F6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "text": "#F8FAFC"
}


def create_fatigue_timeline_chart(records: List[Dict[str, Any]]) -> go.Figure:
    """Creates an interactive timeline showing fatigue score over elapsed session time."""
    fig = go.Figure()

    if not records:
        fig.add_annotation(text="No telemetry data recorded yet.", showarrow=False, font=dict(size=16, color="#94A3B8"))
        fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor=COLOR_MAP["surface"], plot_bgcolor=COLOR_MAP["surface"])
        return fig

    times = [r["elapsed_seconds"] for r in records]
    scores = [r["fatigue_score"] for r in records]

    # Colored background zones
    fig.add_hrect(y0=0, y1=30, fillcolor="#10B981", opacity=0.10, line_width=0, annotation_text="Low", annotation_position="top left")
    fig.add_hrect(y0=30, y1=60, fillcolor="#EAB308", opacity=0.10, line_width=0, annotation_text="Moderate", annotation_position="top left")
    fig.add_hrect(y0=60, y1=80, fillcolor="#F97316", opacity=0.10, line_width=0, annotation_text="High Risk", annotation_position="top left")
    fig.add_hrect(y0=80, y1=100, fillcolor="#EF4444", opacity=0.15, line_width=0, annotation_text="Critical Alert", annotation_position="top left")

    # Fatigue Score Line
    fig.add_trace(go.Scatter(
        x=times,
        y=scores,
        mode="lines",
        name="Fatigue Score",
        line=dict(color="#38BDF8", width=3),
        hovertemplate="Time: %{x:.1f}s<br>Fatigue: %{y:.1f}/100<extra></extra>"
    ))

    # Mark critical points
    crit_times = [r["elapsed_seconds"] for r in records if r["fatigue_score"] >= 80]
    crit_scores = [r["fatigue_score"] for r in records if r["fatigue_score"] >= 80]
    if crit_times:
        fig.add_trace(go.Scatter(
            x=crit_times,
            y=crit_scores,
            mode="markers",
            name="Critical Event",
            marker=dict(color="#EF4444", size=8, symbol="diamond"),
            hovertemplate="CRITICAL: %{y:.1f}<extra></extra>"
        ))

    fig.update_layout(
        title="AI-Derived Fatigue Indicator Progression",
        xaxis_title="Session Elapsed Time (seconds)",
        yaxis_title="Fatigue Score (0-100)",
        yaxis=dict(range=[0, 105]),
        template=DARK_TEMPLATE,
        paper_bgcolor=COLOR_MAP["surface"],
        plot_bgcolor=COLOR_MAP["surface"],
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_eye_probability_chart(records: List[Dict[str, Any]]) -> go.Figure:
    """Creates a chart showing eye open probability vs closure threshold."""
    fig = go.Figure()

    if not records:
        fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor=COLOR_MAP["surface"])
        return fig

    times = [r["elapsed_seconds"] for r in records]
    probs = [r["eye_prob"] for r in records]

    fig.add_hline(y=0.5, line_dash="dash", line_color="#EF4444", annotation_text="Decision Threshold (0.50)")

    fig.add_trace(go.Scatter(
        x=times,
        y=probs,
        mode="lines",
        name="Open Eye Probability",
        line=dict(color="#A78BFA", width=2),
        hovertemplate="Time: %{x:.1f}s<br>Open Prob: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="Eye Open Probability Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Probability (0 = Closed, 1 = Open)",
        yaxis=dict(range=[-0.05, 1.05]),
        template=DARK_TEMPLATE,
        paper_bgcolor=COLOR_MAP["surface"],
        plot_bgcolor=COLOR_MAP["surface"],
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_performance_chart(records: List[Dict[str, Any]]) -> go.Figure:
    """Displays latency timeline."""
    fig = go.Figure()

    if not records:
        fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor=COLOR_MAP["surface"])
        return fig

    times = [r["elapsed_seconds"] for r in records]
    latencies = [r["latency_ms"] for r in records]

    fig.add_trace(go.Scatter(
        x=times,
        y=latencies,
        mode="lines",
        name="Pipeline Latency (ms)",
        line=dict(color="#34D399", width=2),
        hovertemplate="Time: %{x:.1f}s<br>Latency: %{y:.1f} ms<extra></extra>"
    ))

    fig.update_layout(
        title="Real-Time Processing Latency",
        xaxis_title="Time (seconds)",
        yaxis_title="Latency (ms)",
        template=DARK_TEMPLATE,
        paper_bgcolor=COLOR_MAP["surface"],
        plot_bgcolor=COLOR_MAP["surface"],
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
