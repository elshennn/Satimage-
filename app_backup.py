from pathlib import Path
import hashlib
import io
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch
import rasterio
from PIL import Image

# -----------------------------------------------------------------------------
# SecureSat-GAN | premium demonstration UI
# Existing model, preprocessing, validation CSV, and SHA-256 features retained.
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SecureSat-GAN | Earth Observation Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from generator import UNetGenerator  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = ROOT / "models/generator_reflectance_best.pth"
CLOUDY_DIR = ROOT / "data/raw/cloudy"
CLEAR_DIR = ROOT / "data/raw/clear"
METRICS_CSV = ROOT / "outputs/reflectance_validation_metrics.csv"
COMPARISON_CSV = ROOT / "outputs/reflectance_model_comparison.csv"

# ---- Global visual system ----------------------------------------------------
st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#070b11; --panel:#0d141e; --panel2:#111c29; --line:#243243; --text:#edf3f8; --muted:#8b9aab; --lime:#c7ff6b; --cyan:#70e6ec; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { color:var(--text); background:radial-gradient(ellipse at 78% -8%,rgba(54,126,153,.16),transparent 35%),radial-gradient(ellipse at 8% 22%,rgba(114,155,69,.06),transparent 31%),#070b11; }
[data-testid="stHeader"] { background:rgba(7,11,17,.92); border-bottom:1px solid rgba(255,255,255,.04); }
[data-testid="stAppViewContainer"] > .main .block-container { max-width:1660px; padding:1.4rem 3rem 3.5rem; }
#MainMenu, footer { visibility:hidden; }
[data-testid="stDecoration"] { display:none; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:var(--text) !important; letter-spacing:-.035em !important; }
h1 { font-size:2.5rem !important; font-weight:700 !important; }
h2 { font-size:1.75rem !important; font-weight:600 !important; }
h3 { font-size:1.12rem !important; font-weight:600 !important; }
p,li,label { color:#c3ceda; line-height:1.65; }
[data-testid="stCaptionContainer"] { color:var(--muted) !important; font-size:.82rem !important; }
hr { border:0 !important; border-top:1px solid var(--line) !important; margin:1.35rem 0 !important; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b111a,#080c12); border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1rem; }
[data-testid="stTabs"] { margin-top:1.5rem; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:7px; border-bottom:1px solid var(--line); background:transparent; }
[data-testid="stTabs"] button[data-baseweb="tab"] { color:#92a0b0 !important; background:transparent !important; border:1px solid transparent !important; border-radius:7px 7px 0 0 !important; padding:12px 16px !important; font-size:.84rem !important; font-weight:700 !important; }
[data-testid="stTabs"] button[data-baseweb="tab"]:hover { color:#fff !important; background:rgba(255,255,255,.035) !important; }
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] { color:var(--lime) !important; background:#101923 !important; border-color:#243243 !important; border-bottom-color:#101923 !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--lime) !important; height:2px !important; }
.stButton > button[kind="primary"] { background:var(--lime) !important; color:#11180a !important; border:1px solid var(--lime) !important; border-radius:7px !important; min-height:46px; padding:0 22px !important; font-weight:800 !important; transition:all .18s ease; }
.stButton > button[kind="primary"]:hover { background:#d7ff98 !important; border-color:#d7ff98 !important; box-shadow:0 0 28px rgba(199,255,107,.14); transform:translateY(-1px); }
.stButton > button,.stDownloadButton > button { background:#111a25 !important; color:#e3ebf3 !important; border:1px solid #2b3b4e !important; border-radius:7px !important; min-height:41px; font-weight:600 !important; transition:all .18s ease; }
.stButton > button:hover,.stDownloadButton > button:hover { border-color:#758ba2 !important; background:#172432 !important; color:#fff !important; }
div[data-baseweb="select"] > div,div[data-baseweb="input"] > div,div[data-baseweb="textarea"] > div { background:#0d141e !important; border:1px solid #2a394a !important; border-radius:7px !important; box-shadow:none !important; }
div[data-baseweb="select"] > div:focus-within,div[data-baseweb="input"] > div:focus-within,div[data-baseweb="textarea"] > div:focus-within { border-color:var(--cyan) !important; box-shadow:0 0 0 1px rgba(112,230,236,.12) !important; }
input,textarea { color:var(--text) !important; }
[data-testid="stFileUploader"] section { background:#0c131d !important; border:1px dashed #35485b !important; border-radius:10px !important; padding:1rem !important; }
[data-testid="stFileUploader"] section:hover { border-color:var(--cyan) !important; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#111c29,#0b121b) !important; border:1px solid #263748 !important; border-radius:9px !important; padding:18px 20px !important; box-shadow:none !important; }
[data-testid="stMetricLabel"] { color:#91a0b2 !important; font-size:.77rem !important; font-weight:700 !important; }
[data-testid="stMetricValue"] { color:var(--lime) !important; font-family:'Space Grotesk',sans-serif !important; font-size:1.7rem !important; font-weight:700 !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:9px; overflow:hidden; }
[data-testid="stImage"] img { border:1px solid #263748; border-radius:8px; }
[data-testid="stAlert"] { border-radius:8px !important; }
[data-testid="stCode"] { background:#070c12 !important; border:1px solid #273747 !important; border-radius:8px !important; }
code,pre { font-family:'DM Mono',monospace !important; }
div[role="radiogroup"] { gap:13px; }
.stProgress > div > div { background:var(--lime) !important; }
/* Brand hero */
.ss-hero { display:grid; grid-template-columns:minmax(0,1.12fr) minmax(360px,.88fr); gap:24px; align-items:center; position:relative; overflow:hidden; min-height:350px; padding:30px 34px 25px; border:1px solid #263544; border-radius:14px; background:linear-gradient(115deg,rgba(15,24,36,.98),rgba(8,14,22,.96) 54%,rgba(9,24,32,.93)); box-shadow:0 22px 70px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.035); }
.ss-hero:before { content:''; position:absolute; inset:0; pointer-events:none; background:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px); background-size:38px 38px; mask-image:linear-gradient(90deg,black,transparent 86%); }
.ss-hero-copy { position:relative; z-index:2; }
.ss-kicker { font:500 10px 'DM Mono',monospace; letter-spacing:3px; color:var(--lime); margin-bottom:20px; }
.ss-wordmark { color:#f1f6fb; font:700 clamp(38px,4.4vw,65px) 'Space Grotesk',sans-serif; line-height:.99; letter-spacing:-3.5px; }
.ss-wordmark span { color:var(--lime); }
.ss-subtitle { max-width:550px; margin-top:18px; color:#a5b3c2; font-size:14px; line-height:1.8; }
.ss-chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:23px; }
.ss-chip { font:10px 'DM Mono',monospace; letter-spacing:.6px; color:#bdcbd7; border:1px solid #2b3d4e; background:rgba(9,15,23,.72); padding:8px 10px; border-radius:4px; }
.ss-live { display:flex; align-items:center; gap:8px; font:10px 'DM Mono',monospace; color:#b9c6d3; letter-spacing:1.1px; margin-top:26px; }
.ss-dot { width:7px; height:7px; display:inline-block; border-radius:50%; background:var(--lime); box-shadow:0 0 14px rgba(199,255,107,.65); }
.ss-visual { position:relative; min-width:0; z-index:1; }
.ss-visual svg { display:block; width:100%; max-width:520px; margin:0 auto; overflow:visible; }
.ss-visual-caption { display:flex; justify-content:space-between; gap:10px; max-width:470px; margin:2px auto 0; color:#718397; font:9px 'DM Mono',monospace; letter-spacing:1.3px; }
.orbit-ring { fill:none; stroke:#4c7380; stroke-width:1; stroke-dasharray:3 7; opacity:.75; }