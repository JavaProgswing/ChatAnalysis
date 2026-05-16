"""
charts.py — generate Plotly chart JSON + word-cloud PNG (base64) for the dashboard.
"""

import re
import base64
from io import BytesIO
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
)
ACCENT = "#8b5cf6"
ACCENT2 = "#06b6d4"
PINK   = "#ec4899"


def _fig_json(fig) -> dict:
    return fig.to_json()


# ── Hourly activity bar ───────────────────────────────────────────────────────
def hourly_bar(stats: dict) -> str:
    hours = list(range(24))
    counts = [stats["hourly_activity"].get(h, 0) for h in hours]
    labels = [f"{h:02d}:00" for h in hours]

    colors = []
    for h in hours:
        if 0 <= h < 5:   colors.append("#ec4899")   # late night – pink
        elif 5 <= h < 9: colors.append("#f59e0b")   # early morning – amber
        elif 20 <= h:    colors.append("#8b5cf6")   # evening – purple
        else:            colors.append("#06b6d4")   # day – cyan

    fig = go.Figure(go.Bar(x=labels, y=counts, marker_color=colors,
                           hovertemplate="%{x}<br>%{y} messages<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="⏰ Messages by Hour of Day",
                      xaxis=dict(showgrid=False, tickangle=-45),
                      yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"))
    return _fig_json(fig)


# ── Activity heatmap ──────────────────────────────────────────────────────────
def activity_heatmap(stats: dict) -> str:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = list(range(24))
    z = np.zeros((7, 24))
    for row in stats["heatmap"]:
        d, h, c = int(row["dow"]), int(row["hour"]), int(row["count"])
        if 0 <= d < 7 and 0 <= h < 24:
            z[d][h] = c

    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{h:02d}" for h in hours], y=days,
        colorscale="Viridis",
        hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Messages: %{z}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="🔥 Activity Heatmap (Day × Hour)")
    return _fig_json(fig)


# ── Monthly timeline ──────────────────────────────────────────────────────────
def monthly_timeline(stats: dict) -> str:
    data = stats.get("monthly_activity", {})
    if not data:
        return "{}"
    x = sorted(data.keys())
    y = [data[k] for k in x]

    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(color=ACCENT2, size=5),
        fill="tozeroy", fillcolor="rgba(139,92,246,0.12)",
        hovertemplate="%{x}<br>%{y} messages<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="📅 Your Message Volume Over Time",
                      xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"))
    return _fig_json(fig)


# ── Platform donut ────────────────────────────────────────────────────────────
def platform_donut(stats: dict) -> str:
    pb = stats.get("my_platform_breakdown", {})
    colors = ["#8b5cf6", "#06b6d4", "#ec4899", "#f59e0b", "#10b981"]
    fig = go.Figure(go.Pie(
        labels=list(pb.keys()), values=list(pb.values()),
        hole=0.55,
        marker=dict(colors=colors[:len(pb)],
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        hovertemplate="%{label}<br>%{value:,} messages (%{percent})<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="📱 Where You Send Messages",
                      legend=dict(orientation="h", y=-0.1))
    return _fig_json(fig)


# ── Top contacts horizontal bar ──────────────────────────────────────────────
def top_contacts_bar(stats: dict) -> str:
    tc = dict(list(stats.get("top_contacts", {}).items())[:12])
    names = list(tc.keys())
    counts = list(tc.values())
    # Shorten long names
    names = [n[:22] for n in names]

    fig = go.Figure(go.Bar(
        y=names, x=counts, orientation="h",
        marker_color=ACCENT,
        hovertemplate="%{y}<br>%{x:,} messages<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="👥 Your Most Active Chats",
                      xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                      yaxis=dict(showgrid=False, autorange="reversed"))
    return _fig_json(fig)


# ── Day-of-week polar bar ─────────────────────────────────────────────────────
def dow_polar(stats: dict) -> str:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    vals = [stats["daily_activity"].get(d, 0) for d in days]

    fig = go.Figure(go.Barpolar(
        r=vals, theta=days,
        marker_color=ACCENT2,
        hovertemplate="%{theta}<br>%{r:,} messages<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="🗓️ Day of Week Activity",
                      polar=dict(
                          bgcolor="rgba(0,0,0,0)",
                          radialaxis=dict(showticklabels=False, gridcolor="rgba(255,255,255,0.1)"),
                          angularaxis=dict(tickfont=dict(color="#e2e8f0")),
                      ))
    return _fig_json(fig)


# ── Message length distribution ───────────────────────────────────────────────
def msg_length_dist(stats: dict) -> str:
    dist = stats.get("msg_length_dist", {})
    labels = list(dist.keys())
    vals   = list(dist.values())

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=[ACCENT, ACCENT2, PINK, "#10b981", "#f59e0b", "#ef4444"],
        hovertemplate="Length %{x} chars<br>%{y:,} messages<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="📏 Message Length Distribution",
                      xaxis_title="Characters",
                      yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"))
    return _fig_json(fig)


# ── Word cloud (returns base64 PNG) ──────────────────────────────────────────
def wordcloud_b64(stats: dict) -> str:
    text = stats.get("all_text_sample", "")
    if not text.strip():
        return ""
    try:
        wc = WordCloud(
            width=900, height=450,
            background_color="#0f172a", # Dark slate background to match theme
            colormap="viridis",
            max_words=200,
            collocations=False,
        ).generate(text)
        buf = BytesIO()
        wc.to_image().save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[ERROR] WordCloud generation failed: {e}")
        return ""


# ── Emoji chart ───────────────────────────────────────────────────────────────
def emoji_bar(stats: dict) -> str:
    data = stats.get("top_emojis", [])[:15]
    if not data:
        return "{}"
    labels = [d["emoji"] for d in data]
    vals   = [d["count"] for d in data]

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=PINK,
        hovertemplate="%{x}<br>%{y:,} times<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="😂 Your Most-Used Emojis",
                      xaxis=dict(tickfont=dict(size=20)),
                      yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"))
    return _fig_json(fig)


# ── GPT topic word cloud ──────────────────────────────────────────────────────
def gpt_topic_cloud(stats: dict) -> str:
    titles = stats.get("gpt_titles", [])
    if not titles:
        return ""
    text = " ".join(titles)
    try:
        wc = WordCloud(
            width=900, height=400,
            background_color=None,
            mode="RGBA",
            colormap="plasma",
            max_words=100,
            collocations=False,
        ).generate(text)
        buf = BytesIO()
        wc.to_image().save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def build_all(stats: dict) -> dict:
    return {
        "hourly":         hourly_bar(stats),
        "heatmap":        activity_heatmap(stats),
        "monthly":        monthly_timeline(stats),
        "platform":       platform_donut(stats),
        "top_contacts":   top_contacts_bar(stats),
        "dow":            dow_polar(stats),
        "msg_length":     msg_length_dist(stats),
        "emoji":          emoji_bar(stats),
        "wordcloud":      wordcloud_b64(stats),
        "gpt_topics":     gpt_topic_cloud(stats),
    }
