"""
VeritasRAG - Hybrid Retrieval-Augmented Generation with Citations & Evaluation
==============================================================================
Streamlit front-end that ties the six pipeline phases together into a single
demo-friendly application:

    Chat              - grounded Q&A with inline citations and a with/without-RAG
                        toggle that demonstrates hallucination.
    Retrieval Compare - BM25 vs Dense vs Hybrid vs Hybrid+Rerank, side by side,
                        reproducing Table 1 / Figure 4 of the research paper.
    Evaluation        - Ragas-style faithfulness / relevance / precision metrics.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.chunking import chunk_pages
from src.config import AVAILABLE_BACKENDS, EVAL_SET_PATH, GENERATOR, RETRIEVAL, SAMPLE_DOCS_DIR
from src.evaluation import RagEvaluator, aggregate
from src.generator import generate, health_check
from src.ingestion import load_documents
from src.retriever import HybridRetriever

STRATEGY_LABELS = {
    "sparse": "BM25 (Sparse)",
    "dense": "Dense (HNSW)",
    "hybrid": "Hybrid (RRF)",
    "rerank": "Hybrid + Rerank",
}
STRATEGY_ICONS = {"sparse": "🔤", "dense": "🧠", "hybrid": "🔀", "rerank": "🎯"}
BACKEND_META = {
    "gemini": {"label": "Gemini 2.5 Flash", "icon": "✨", "hint": "cloud"},
    "ollama": {"label": "Ollama", "icon": "🖥️", "hint": "local, free"},
    "extractive": {"label": "Extractive", "icon": "📄", "hint": "no LLM"},
}

st.set_page_config(page_title="VeritasRAG", page_icon="🔮", layout="wide")

# --------------------------------------------------------------------------
# Design system — "midnight aurora"
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --vr-violet: #8b5cf6;
    --vr-violet-soft: #a78bfa;
    --vr-fuchsia: #d946ef;
    --vr-cyan: #22d3ee;
    --vr-ink: #eef0ff;
    --vr-muted: #a6abd3;
    --vr-faint: #7c81ab;
    --vr-glass: rgba(255, 255, 255, 0.035);
    --vr-glass-hi: rgba(255, 255, 255, 0.07);
    --vr-line: rgba(255, 255, 255, 0.09);
    --vr-line-soft: rgba(255, 255, 255, 0.055);
    --vr-good: #34d399;
    --vr-warn: #fbbf24;
    --vr-bad: #f87171;
    --vr-radius-xl: 24px;
    --vr-radius-lg: 18px;
    --vr-radius-md: 14px;
    --vr-radius-sm: 10px;
    --vr-glow: 0 0 24px rgba(139, 92, 246, 0.35);
}

/* ================= ANIMATIONS ================= */
@keyframes vr-drift {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50%      { transform: translate(30px, -24px) scale(1.12); }
}
@keyframes vr-drift2 {
    0%, 100% { transform: translate(0, 0) scale(1.05); }
    50%      { transform: translate(-36px, 18px) scale(0.94); }
}
@keyframes vr-gradient-x {
    0%, 100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}
@keyframes vr-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.45); }
    50%      { box-shadow: 0 0 0 7px rgba(52, 211, 153, 0); }
}
@keyframes vr-float {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-9px); }
}
@keyframes vr-fadeup {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ================= BASE CANVAS ================= */
.stApp {
    background:
        radial-gradient(ellipse 55% 38% at 78% -8%, rgba(139, 92, 246, 0.16), transparent 60%),
        radial-gradient(ellipse 45% 32% at 12% 4%, rgba(34, 211, 238, 0.09), transparent 55%),
        radial-gradient(ellipse 60% 45% at 50% 110%, rgba(217, 70, 239, 0.07), transparent 60%),
        #07070f;
    background-attachment: fixed;
}
.block-container { padding-top: 0.4rem; padding-bottom: 3rem; max-width: 1240px; }
/* trim Streamlit's default top chrome so the hero sits near the top */
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stMainMenu"] { display: none !important; }
.stDeployButton, [data-testid="stDeployButton"] { display: none !important; }
[data-testid="stAppViewBlockContainer"] { padding-top: 0.4rem; }
[data-testid="stDecoration"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    color: var(--vr-ink); letter-spacing: -0.3px;
}
p, li { color: var(--vr-muted); }
[data-testid="stCaptionContainer"] p { color: var(--vr-faint); }
a { color: var(--vr-violet-soft); }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139, 92, 246, 0.3); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139, 92, 246, 0.55); }

/* ================= HERO ================= */
.vr-hero {
    position: relative; overflow: hidden;
    background: linear-gradient(150deg, #14142e 0%, #0d0d1f 50%, #170f2e 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: var(--vr-radius-xl);
    padding: 44px 48px 40px;
    margin-bottom: 26px;
    box-shadow: 0 26px 80px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.07);
    animation: vr-fadeup 0.55s ease both;
}
.vr-orb { position: absolute; border-radius: 50%; filter: blur(70px); pointer-events: none; }
.vr-orb.o1 { width: 380px; height: 380px; top: -170px; right: -70px;
             background: radial-gradient(circle, rgba(139,92,246,0.5), transparent 70%);
             animation: vr-drift 9s ease-in-out infinite; }
.vr-orb.o2 { width: 300px; height: 300px; bottom: -160px; left: 12%;
             background: radial-gradient(circle, rgba(34,211,238,0.32), transparent 70%);
             animation: vr-drift2 11s ease-in-out infinite; }
.vr-orb.o3 { width: 240px; height: 240px; top: -100px; left: 42%;
             background: radial-gradient(circle, rgba(217,70,239,0.3), transparent 70%);
             animation: vr-drift 13s ease-in-out infinite reverse; }
.vr-hero-grid {
    position: absolute; inset: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: radial-gradient(ellipse 75% 90% at 50% 0%, black 30%, transparent 75%);
    -webkit-mask-image: radial-gradient(ellipse 75% 90% at 50% 0%, black 30%, transparent 75%);
}
.vr-hero-inner { position: relative; display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap; }
.vr-eyebrow {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--vr-cyan); margin-bottom: 10px; display: flex; align-items: center; gap: 9px;
}
.vr-eyebrow::before { content: ""; width: 26px; height: 1.5px; background: linear-gradient(90deg, var(--vr-cyan), transparent); }
.vr-hero-title {
    font-family: 'Space Grotesk', sans-serif; font-size: 3.1rem; font-weight: 700;
    letter-spacing: -1.5px; line-height: 1.04; margin: 0;
    background: linear-gradient(92deg, #ffffff 0%, #c4b5fd 48%, #67e8f9 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: vr-gradient-x 7s ease infinite;
}
.vr-hero-sub {
    font-size: 1.02rem; color: var(--vr-muted); margin-top: 12px;
    max-width: 620px; line-height: 1.6;
}
.vr-hero-sub b { color: var(--vr-ink); font-weight: 600; }
.vr-hero-tags { margin-top: 22px; display: flex; flex-wrap: wrap; gap: 9px; }
.vr-hero-tags span {
    background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 6px 15px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    color: #cdd1f4; backdrop-filter: blur(6px);
    transition: all 0.2s ease;
}
.vr-hero-tags span:hover {
    border-color: rgba(139, 92, 246, 0.6); color: #fff;
    box-shadow: var(--vr-glow); transform: translateY(-2px);
}
.vr-hero-badge {
    display: inline-flex; align-items: center; gap: 9px;
    background: rgba(255, 255, 255, 0.05); border: 1px solid var(--vr-line);
    border-radius: 999px; padding: 9px 18px; font-size: 0.8rem; font-weight: 700;
    color: var(--vr-muted); white-space: nowrap; backdrop-filter: blur(8px);
}
.vr-hero-badge.ready { border-color: rgba(52, 211, 153, 0.45); color: #a7f3d0; }
.vr-hero-badge .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--vr-faint); }
.vr-hero-badge.ready .dot { background: var(--vr-good); animation: vr-pulse 2s ease-out infinite; }

/* ================= SIDEBAR ================= */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0c1c 0%, #090913 100%);
    border-right: 1px solid var(--vr-line-soft);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.vr-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.vr-brand-mark {
    width: 42px; height: 42px; border-radius: 13px; flex-shrink: 0;
    background: linear-gradient(135deg, var(--vr-violet), var(--vr-fuchsia));
    display: flex; align-items: center; justify-content: center; font-size: 1.25rem;
    box-shadow: 0 0 22px rgba(139, 92, 246, 0.5), inset 0 1px 0 rgba(255,255,255,0.3);
}
.vr-brand-title {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.28rem; font-weight: 700;
    color: var(--vr-ink); letter-spacing: -0.4px; line-height: 1.1;
}
.vr-brand-sub { font-size: 0.68rem; color: var(--vr-faint); font-weight: 700; letter-spacing: 0.14em; margin-top: 2px; }

[data-testid="stSidebar"] h3 {
    font-size: 0.7rem !important; font-weight: 700 !important;
    color: var(--vr-violet-soft) !important;
    text-transform: uppercase; letter-spacing: 0.14em; margin: 0 0 4px 0 !important;
}
[data-testid="stSidebar"] hr { display: none; }
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--vr-glass);
    border: 1px solid var(--vr-line) !important;
    border-radius: var(--vr-radius-lg);
    box-shadow: 0 8px 26px rgba(0, 0, 0, 0.35);
    margin-bottom: 14px;
    backdrop-filter: blur(10px);
}

.vr-status-dot { display: inline-flex; align-items: center; gap: 7px; font-size: 0.78rem; font-weight: 700; }
.vr-status-dot.ok { color: var(--vr-good); }
.vr-status-dot.bad { color: var(--vr-bad); }
.vr-status-dot .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; box-shadow: 0 0 10px currentColor; }

.vr-stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
.vr-stat {
    background: rgba(139, 92, 246, 0.07); border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: var(--vr-radius-sm); padding: 9px 11px;
}
.vr-stat .n {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 700;
    background: linear-gradient(90deg, var(--vr-violet-soft), var(--vr-cyan));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.15;
}
.vr-stat .l { font-size: 0.64rem; color: var(--vr-faint); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 1px; }

/* ================= WIDGETS ================= */
[data-testid="stMetric"] {
    position: relative; overflow: hidden;
    background: var(--vr-glass); border: 1px solid var(--vr-line);
    border-radius: var(--vr-radius-lg); padding: 18px 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(8px);
}
[data-testid="stMetric"]::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2.5px;
    background: linear-gradient(90deg, var(--vr-violet), var(--vr-fuchsia), var(--vr-cyan));
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: var(--vr-ink);
}
[data-testid="stMetricLabel"] { font-weight: 600; color: var(--vr-faint); }

.stTabs [data-baseweb="tab-list"] {
    gap: 10px; border-bottom: none !important; box-shadow: none !important;
    margin: 6px 0 22px; padding: 0; background: transparent;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 14px !important;
    padding: 12px 24px !important; height: auto !important; min-height: 0 !important;
    font-weight: 600; font-size: 0.9rem; line-height: 1.2;
    background: var(--vr-glass); border: 1px solid var(--vr-line); color: var(--vr-muted);
    transition: all 0.18s ease;
}
.stTabs [data-baseweb="tab"]:hover { border-color: rgba(139, 92, 246, 0.5); color: var(--vr-ink); }
.stTabs button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(120deg, var(--vr-violet), var(--vr-fuchsia)) !important;
    color: #fff !important; border-color: transparent !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.4);
}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] [data-testid="stMarkdownContainer"] { color: #fff !important; }
/* kill the sliding underline / baseline indicator — these are div children of the
   tab-list bar (the tabs themselves are <button>, so this never touches them) */
.stTabs [data-baseweb="tab-list"] > div { display: none !important; height: 0 !important; background: transparent !important; }
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; height: 0 !important; background: transparent !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 4px; }

.stButton > button {
    border-radius: var(--vr-radius-sm); font-weight: 600;
    background: var(--vr-glass-hi); border: 1px solid var(--vr-line); color: var(--vr-ink);
    transition: transform 0.06s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px); border-color: rgba(139, 92, 246, 0.55);
    box-shadow: var(--vr-glow); color: #fff;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, var(--vr-violet), var(--vr-fuchsia));
    border: none; color: #fff;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.35);
}
.stButton > button[kind="primary"]:hover { box-shadow: 0 8px 28px rgba(139, 92, 246, 0.55); }

[data-testid="stToggle"] label p { font-weight: 600; color: var(--vr-ink); }

[data-testid="stExpander"] {
    border: 1px solid var(--vr-line-soft); border-radius: var(--vr-radius-md);
    background: var(--vr-glass);
}
[data-testid="stPopoverBody"] { border-radius: var(--vr-radius-md); }

[data-testid="stChatMessage"] {
    background: var(--vr-glass); border: 1px solid var(--vr-line-soft);
    border-radius: var(--vr-radius-lg); box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(6px);
}
[data-testid="stChatInput"] { border-radius: var(--vr-radius-md); }
[data-testid="stChatInput"] textarea { border-radius: var(--vr-radius-md); }

[data-testid="stAlertContainer"] { border-radius: var(--vr-radius-sm); }
[data-testid="stFileUploaderDropzone"] {
    border-radius: var(--vr-radius-md);
    background: rgba(139, 92, 246, 0.05);
    border: 1.5px dashed rgba(139, 92, 246, 0.35);
}

[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--vr-glass); border: 1px solid var(--vr-line) !important;
    border-radius: var(--vr-radius-lg); box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
}

/* ---- citation pill + inline badges ---- */
.vr-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(139, 92, 246, 0.14); color: var(--vr-violet-soft);
    border: 1px solid rgba(139, 92, 246, 0.3);
    padding: 4px 13px; border-radius: 999px; font-size: 0.78rem; font-weight: 700;
}
.vr-cite-badge {
    display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, var(--vr-violet), var(--vr-fuchsia));
    color: #fff !important; font-size: 0.66rem; font-weight: 800;
    width: 17px; height: 17px; border-radius: 50%; vertical-align: super; margin: 0 1px;
    box-shadow: 0 0 8px rgba(139, 92, 246, 0.5);
}

/* ---- source cards ---- */
.vr-src-head { display: flex; align-items: center; gap: 8px; }
.vr-src-num {
    background: rgba(255, 255, 255, 0.06); color: var(--vr-violet-soft);
    font-weight: 800; font-size: 0.74rem;
    border: 1px solid var(--vr-line); border-radius: 999px; width: 22px; height: 22px;
    display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.vr-src-num.cited {
    background: linear-gradient(135deg, var(--vr-violet), var(--vr-fuchsia));
    color: #fff; border-color: transparent; box-shadow: 0 0 10px rgba(139, 92, 246, 0.45);
}
.vr-src-meta { font-size: 0.75rem; color: var(--vr-faint); font-weight: 600; }
.vr-score-bar { height: 4px; border-radius: 999px; background: rgba(255, 255, 255, 0.07); margin-top: 7px; overflow: hidden; }
.vr-score-bar > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--vr-violet), var(--vr-cyan)); box-shadow: 0 0 8px rgba(139, 92, 246, 0.6); }

/* ---- empty state ---- */
.vr-empty {
    text-align: center; padding: 64px 24px;
    background: var(--vr-glass);
    border: 1.5px dashed rgba(139, 92, 246, 0.3);
    border-radius: var(--vr-radius-xl); color: var(--vr-muted);
    animation: vr-fadeup 0.4s ease both;
}
.vr-empty .big { font-size: 2.8rem; margin-bottom: 10px; animation: vr-float 3s ease-in-out infinite; display: inline-block; }
.vr-empty .title {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    color: var(--vr-ink); font-size: 1.15rem; margin-bottom: 5px;
}

/* ---- pipeline flow ---- */
.vr-flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: 10px; margin: 10px 0 22px; }
.vr-step {
    position: relative;
    background: var(--vr-glass); border: 1px solid var(--vr-line);
    border-radius: var(--vr-radius-md); padding: 13px 16px 11px;
    font-weight: 600; font-size: 0.86rem; color: var(--vr-ink);
    transition: all 0.18s ease; min-width: 128px;
}
.vr-step:hover {
    transform: translateY(-3px); border-color: rgba(139, 92, 246, 0.55);
    box-shadow: var(--vr-glow);
}
.vr-step .num {
    font-family: 'Space Grotesk', sans-serif; font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.12em; margin-bottom: 3px;
    background: linear-gradient(90deg, var(--vr-violet-soft), var(--vr-cyan));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.vr-step small { display: block; font-weight: 500; color: var(--vr-faint); font-size: 0.71rem; margin-top: 2px; }
.vr-arrow { color: rgba(139, 92, 246, 0.7); font-weight: 800; font-size: 1.15rem; align-self: center; text-shadow: 0 0 10px rgba(139, 92, 246, 0.6); }

/* ---- section labels ---- */
.vr-section-label {
    font-size: 0.7rem; font-weight: 700; color: var(--vr-violet-soft);
    text-transform: uppercase; letter-spacing: 0.14em; margin: 8px 0 4px;
    display: flex; align-items: center; gap: 9px;
}
.vr-section-label::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(139,92,246,0.35), transparent); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOT_FONT = dict(family="Inter, sans-serif", color="#a6abd3", size=13)
PLOT_GRID = "rgba(255,255,255,0.07)"


def style_fig(fig):
    """Apply the dark 'midnight aurora' look to a plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
        title_font=dict(family="Space Grotesk, Inter, sans-serif", size=17, color="#eef0ff"),
        xaxis=dict(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID, linecolor=PLOT_GRID),
        yaxis=dict(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID, linecolor=PLOT_GRID),
        hoverlabel=dict(bgcolor="#15152b", bordercolor="rgba(139,92,246,0.5)", font=dict(color="#eef0ff")),
    )
    return fig


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def _init_state():
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("build_stats", None)
    st.session_state.setdefault("backend", GENERATOR.backend)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_health", None)  # (ok, msg)


_init_state()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_evaluator():
    return RagEvaluator()


def build_index_from_paths(paths):
    retriever = HybridRetriever()
    stats = retriever.build(chunk_pages(load_documents(paths)))
    st.session_state.retriever = retriever
    st.session_state.build_stats = stats
    return stats


def _citeify(answer: str) -> str:
    """Turn plain [n] markers into small round badges for a cleaner citation look."""
    return re.sub(r"\[(\d+)\]", r"<span class='vr-cite-badge'>\1</span>", answer)


def render_answer_with_citations(answer: str, passages):
    """Show the answer, then a card grid of the numbered sources it can cite."""
    st.markdown(_citeify(answer), unsafe_allow_html=True)
    if not passages:
        return
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    st.markdown(
        f"<span class='vr-pill'>📎 {len(passages)} sources retrieved · {len(cited)} cited</span>",
        unsafe_allow_html=True,
    )
    st.write("")
    max_score = max((p.score for p in passages), default=1.0) or 1.0
    cols = st.columns(min(len(passages), 3))
    for i, p in enumerate(passages, start=1):
        is_cited = i in cited
        with cols[(i - 1) % len(cols)]:
            with st.container(border=True):
                st.markdown(
                    f"<div class='vr-src-head'>"
                    f"<span class='vr-src-num{' cited' if is_cited else ''}'>{i}</span>"
                    f"<span class='vr-src-meta'>{p.citation}</span></div>"
                    f"<div class='vr-score-bar'><div style='width:{100 * p.score / max_score:.0f}%'></div></div>",
                    unsafe_allow_html=True,
                )
                snippet = p.chunk.text.strip()
                if len(snippet) > 220:
                    snippet = snippet[:220].rsplit(" ", 1)[0] + "…"
                st.caption(snippet)
                with st.popover("Read full passage", width="stretch"):
                    st.caption(f"**{p.citation}** · score {p.score:.3f}")
                    st.write(p.chunk.text)


def empty_state(icon: str, title: str, body: str):
    st.markdown(
        f"<div class='vr-empty'><div class='big'>{icon}</div>"
        f"<div class='title'>{title}</div><div>{body}</div></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar - backend + corpus
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div class='vr-brand'><div class='vr-brand-mark'>🔮</div>"
        "<div class='vr-brand-text'><div class='vr-brand-title'>VeritasRAG</div>"
        "<div class='vr-brand-sub'>HYBRID RAG · CITATIONS · EVAL</div></div></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    with st.container(border=True):
        st.subheader("Generator backend")
        backend = st.segmented_control(
            "Answer generator",
            AVAILABLE_BACKENDS,
            default=st.session_state.backend,
            format_func=lambda b: f"{BACKEND_META[b]['icon']} {BACKEND_META[b]['label']}",
            label_visibility="collapsed",
            help="Switch the model that writes the final answer. Retrieval is identical for all.",
        ) or st.session_state.backend
        st.session_state.backend = backend
        GENERATOR.backend = backend
        st.caption(f"{BACKEND_META[backend]['hint']}")

        if st.button("🔌 Test connection", width="stretch"):
            with st.spinner("Pinging backend..."):
                ok, msg = health_check(backend)
            st.session_state.last_health = (ok, msg)

        if st.session_state.last_health:
            ok, msg = st.session_state.last_health
            css_class = "ok" if ok else "bad"
            st.markdown(
                f"<div class='vr-status-dot {css_class}'><span class='dot'></span>{msg}</div>",
                unsafe_allow_html=True,
            )
        st.write("")

    with st.container(border=True):
        st.subheader("Knowledge base")

        uploads = st.file_uploader(
            "Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True,
            label_visibility="visible",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⚙️ Build index", type="primary", width="stretch"):
                if uploads:
                    tmp_paths = []
                    tmpdir = Path(tempfile.mkdtemp())
                    for uf in uploads:
                        p = tmpdir / uf.name
                        p.write_bytes(uf.getbuffer())
                        tmp_paths.append(p)
                    with st.spinner("Ingesting, chunking, embedding..."):
                        build_index_from_paths(tmp_paths)
                    st.success("Index built from uploads.")
                else:
                    st.warning("Upload a file first, or load the sample corpus.")
        with col_b:
            if st.button("🧪 Load sample", width="stretch"):
                sample_paths = sorted(SAMPLE_DOCS_DIR.glob("*"))
                if sample_paths:
                    with st.spinner("Building sample index..."):
                        build_index_from_paths(sample_paths)
                    st.success(f"Loaded {len(sample_paths)} sample docs.")
                else:
                    st.warning("No sample documents found in data/sample_docs/.")

        if st.session_state.build_stats:
            s = st.session_state.build_stats
            st.markdown(
                f"<div class='vr-stat-grid'>"
                f"<div class='vr-stat'><div class='n'>{s['num_chunks']}</div><div class='l'>Chunks</div></div>"
                f"<div class='vr-stat'><div class='n'>{s['embed_seconds']}s</div><div class='l'>Embed</div></div>"
                f"<div class='vr-stat'><div class='n'>{s['bm25_seconds']}s</div><div class='l'>BM25</div></div>"
                f"<div class='vr-stat'><div class='n' style='font-size:0.76rem'>"
                f"{RETRIEVAL.embedding_model.split('/')[-1]}</div><div class='l'>Model</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.write("")

    st.caption("Made with 💜 · VeritasRAG demo")


# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
retriever: HybridRetriever | None = st.session_state.retriever
ready = bool(retriever and retriever.is_ready)

badge_class = "ready" if ready else ""
badge_text = (
    f"INDEX READY · {len(retriever.chunks)} CHUNKS" if ready else "NO INDEX LOADED"
)

st.markdown(
    f"""
    <div class="vr-hero">
        <div class="vr-orb o1"></div>
        <div class="vr-orb o2"></div>
        <div class="vr-orb o3"></div>
        <div class="vr-hero-grid"></div>
        <div class="vr-hero-inner">
            <div>
                <div class="vr-eyebrow">Hybrid Retrieval · Grounded Generation · Evaluation</div>
                <h1 class="vr-hero-title">VeritasRAG</h1>
                <div class="vr-hero-sub">
                    Retrieval-Augmented Generation built to <b>verify instead of trust</b> —
                    every answer is grounded in your documents and backed by
                    <b>inline citations</b> you can inspect.
                </div>
                <div class="vr-hero-tags">
                    <span>🧩 FAISS HNSW</span>
                    <span>🔤 BM25</span>
                    <span>🔀 RRF Fusion</span>
                    <span>🎯 Cross-Encoder Rerank</span>
                    <span>💬 Gemini · Ollama</span>
                    <span>📊 Ragas Evaluation</span>
                </div>
            </div>
            <div class="vr-hero-badge {badge_class}"><span class="dot"></span>{badge_text}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_cmp, tab_eval, tab_about = st.tabs(
    ["💬 Ask", "🔍 Retrieval Comparison", "📊 Evaluation", "ℹ️ How it works"]
)

# ---- Chat -----------------------------------------------------------------
with tab_chat:
    if not ready:
        empty_state("🗂️", "No knowledge base yet", "Build an index or load the sample corpus from the sidebar to begin.")
    else:
        top_row = st.columns([3, 1.1])
        with top_row[0]:
            use_rag = st.toggle(
                "Ground answer in retrieved documents (RAG)",
                value=True,
                help="Turn OFF to see the model answer from memory alone — it will often "
                "hallucinate. Turn ON to force grounding in your documents.",
            )
        with top_row[1]:
            if st.button(
                "🗑️ Clear conversation",
                width="stretch",
                help="Remove all messages and start a fresh conversation.",
            ):
                st.session_state.chat_history = []
                st.rerun()

        query = st.chat_input("Ask a question about your documents...")
        if query:
            st.session_state.chat_history.append(("user", query))

        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                if isinstance(content, tuple):  # assistant payload
                    answer, passages, meta = content
                    render_answer_with_citations(answer, passages)
                    st.caption(meta)
                else:
                    st.markdown(content)

        if query:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating..."):
                    passages = []
                    t0 = time.perf_counter()
                    if use_rag:
                        result = retriever.retrieve(query)
                        passages = result.passages
                        gen = generate(query, passages, backend=backend, grounded=True)
                        retrieval_ms = result.latency_ms
                    else:
                        gen = generate(query, [], backend=backend, grounded=False)
                        retrieval_ms = 0.0
                    total_ms = (time.perf_counter() - t0) * 1000.0

                mode = "RAG-grounded" if use_rag else "⚠️ No-RAG (memory only)"
                meta = (
                    f"{mode} · backend: {gen.backend_used} · "
                    f"retrieval {retrieval_ms:.0f} ms · total {total_ms:.0f} ms"
                )
                if gen.note:
                    st.warning(gen.note)
                render_answer_with_citations(gen.answer, passages)
                st.caption(meta)
                st.session_state.chat_history.append(
                    ("assistant", (gen.answer, passages, meta))
                )

# ---- Retrieval comparison -------------------------------------------------
with tab_cmp:
    st.subheader("Retrieval strategy comparison")
    st.caption(
        "Runs the same query through all four strategies from the research paper "
        "and shows which passages each one surfaces, plus its latency."
    )
    if not ready:
        empty_state("🔍", "No knowledge base yet", "Build an index first to compare retrieval strategies.")
    else:
        cmp_query = st.text_input("Query", key="cmp_query", placeholder="e.g. What is the refund policy?")
        if st.button("Compare strategies", type="primary") and cmp_query:
            rows = []
            passages_by_strategy = {}
            for strat in ["sparse", "dense", "hybrid", "rerank"]:
                res = retriever.run_strategy(strat, cmp_query, top_k=RETRIEVAL.top_k_final)
                passages_by_strategy[strat] = res
                for p in res.passages:
                    rows.append(
                        {
                            "Strategy": STRATEGY_LABELS[strat],
                            "Rank": p.rank,
                            "Score": round(p.score, 4),
                            "Source": p.citation,
                        }
                    )

            lat_df = pd.DataFrame(
                {
                    "Strategy": [STRATEGY_LABELS[s] for s in passages_by_strategy],
                    "Latency (ms)": [passages_by_strategy[s].latency_ms for s in passages_by_strategy],
                    "Top score": [
                        round(passages_by_strategy[s].passages[0].score, 4)
                        if passages_by_strategy[s].passages else 0.0
                        for s in passages_by_strategy
                    ],
                }
            )

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    lat_df, x="Strategy", y="Latency (ms)",
                    title="Query latency by strategy", text="Latency (ms)",
                )
                fig.update_traces(
                    marker_color="#a78bfa",
                    marker_line_width=0,
                    textfont=dict(color="#eef0ff"),
                )
                fig.update_layout(showlegend=False)
                style_fig(fig)
                st.plotly_chart(fig, width="stretch")
            with c2:
                st.dataframe(lat_df, width="stretch", hide_index=True)

            st.markdown("<div class='vr-section-label'>Top passages per strategy</div>", unsafe_allow_html=True)
            cols = st.columns(4)
            for col, strat in zip(cols, ["sparse", "dense", "hybrid", "rerank"]):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{STRATEGY_ICONS[strat]} {STRATEGY_LABELS[strat]}**")
                        strat_passages = passages_by_strategy[strat].passages
                        max_score = max((p.score for p in strat_passages), default=1.0) or 1.0
                        for p in strat_passages:
                            st.markdown(
                                f"<div class='vr-src-meta'>[{p.rank}] {p.citation} · {p.score:.3f}</div>"
                                f"<div class='vr-score-bar'><div style='width:{100 * p.score / max_score:.0f}%'></div></div>",
                                unsafe_allow_html=True,
                            )
                            st.caption(p.chunk.text[:150] + "...")

# ---- Evaluation -----------------------------------------------------------
with tab_eval:
    st.subheader("RAG evaluation dashboard")
    st.caption(
        "Ragas-style metrics computed locally (no paid API): context precision, "
        "faithfulness, answer relevance — Section 5.7 of the research paper."
    )
    if not ready:
        empty_state("📊", "No knowledge base yet", "Build an index first to run the evaluation dashboard.")
    else:
        default_eval = EVAL_SET_PATH.exists()
        st.write(
            "Evaluation questions are read from `data/eval_set.json`."
            if default_eval else
            "No eval_set.json found — using a couple of generic questions."
        )
        if st.button("Run evaluation", type="primary"):
            if default_eval:
                questions = [q["question"] for q in json.loads(EVAL_SET_PATH.read_text())]
            else:
                questions = ["What is this document about?", "Summarise the key points."]

            evaluator = _get_evaluator()
            scores = []
            progress = st.progress(0.0, text="Scoring questions...")
            for i, q in enumerate(questions, start=1):
                res = retriever.retrieve(q)
                gen = generate(q, res.passages, backend=backend, grounded=True)
                scores.append(evaluator.score_one(q, gen.answer, res.passages))
                progress.progress(i / len(questions), text=f"Scored {i}/{len(questions)}")
            progress.empty()

            agg = aggregate(scores)
            m1, m2, m3 = st.columns(3)
            m1.metric("Context Precision", f"{agg['context_precision']:.2f}")
            m2.metric("Faithfulness", f"{agg['faithfulness']:.2f}")
            m3.metric("Answer Relevance", f"{agg['answer_relevance']:.2f}")

            radar = go.Figure()
            radar.add_trace(
                go.Scatterpolar(
                    r=[agg["context_precision"], agg["faithfulness"], agg["answer_relevance"]],
                    theta=["Context Precision", "Faithfulness", "Answer Relevance"],
                    fill="toself",
                    name="VeritasRAG",
                    line_color="#a78bfa",
                    fillcolor="rgba(139,92,246,0.28)",
                )
            )
            radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor=PLOT_GRID, linecolor=PLOT_GRID),
                    angularaxis=dict(gridcolor=PLOT_GRID, linecolor=PLOT_GRID),
                ),
                title="Aggregate quality profile",
            )
            style_fig(radar)
            st.plotly_chart(radar, width="stretch")

            df = pd.DataFrame([s.as_dict() for s in scores])
            st.dataframe(df, width="stretch", hide_index=True)

# ---- About ----------------------------------------------------------------
with tab_about:
    st.subheader("The VeritasRAG pipeline")
    st.markdown(
        """
        <div class="vr-flow">
            <div class="vr-step"><div class="num">PHASE 01</div>📄 Ingest<small>PDF / TXT → text</small></div>
            <div class="vr-arrow">→</div>
            <div class="vr-step"><div class="num">PHASE 02</div>✂️ Chunk<small>overlapping windows</small></div>
            <div class="vr-arrow">→</div>
            <div class="vr-step"><div class="num">PHASE 03</div>🧠 Embed + Index<small>FAISS HNSW + BM25</small></div>
            <div class="vr-arrow">→</div>
            <div class="vr-step"><div class="num">PHASE 04</div>🔀 Fuse (RRF)<small>dense + sparse</small></div>
            <div class="vr-arrow">→</div>
            <div class="vr-step"><div class="num">PHASE 05</div>🎯 Rerank<small>cross-encoder</small></div>
            <div class="vr-arrow">→</div>
            <div class="vr-step"><div class="num">PHASE 06</div>💬 Generate<small>cited answer</small></div>
            <div class="vr-arrow">→</div>
            <div class="vr-step"><div class="num">PHASE 07</div>📊 Evaluate<small>Ragas metrics</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("How it maps to the research paper")
    st.markdown(
        """
| Pipeline phase | Paper section | Where in this app |
|---|---|---|
| **1. Ingestion** | 4.3 | `src/ingestion.py` — PDF/TXT text + page numbers |
| **2. Indexing** | 4.5 | `src/chunking.py`, `src/vector_store.py` (FAISS **HNSW**) |
| **3. Retrieval** | 4.4 | `src/sparse.py` (BM25) + dense + `src/fusion.py` (**RRF**) + `src/reranker.py` |
| **4. Augmentation** | 5.2 / 5.6 | `src/prompt_builder.py` — instruction hardening + delimited context |
| **5. Generation** | 4.3 | `src/generator.py` — Gemini / Ollama / extractive |
| **6. Evaluation** | 5.7 | `src/evaluation.py` — Ragas-style metrics |

**Production retrieval path:** BM25 + Dense → Reciprocal Rank Fusion →
Cross-encoder rerank → top-k → grounded prompt → LLM answer with `[n]` citations.
This is the *"hybrid retrieval with a cross-encoder reranker"* configuration the
paper identifies as the most consistently strong.
        """
    )
    st.info(
        f"Chunk size ≈ {RETRIEVAL.chunk_size_words} words with "
        f"{RETRIEVAL.chunk_overlap_words}-word overlap · "
        f"embedding dim depends on model · HNSW M={RETRIEVAL.hnsw_m}."
    )
