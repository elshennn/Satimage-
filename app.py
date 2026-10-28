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
# SecureSat-GAN | Earth Observation Intelligence
# Cloud restoration with a trained U-Net and SHA-256 file integrity checking.
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
CHECKPOINT = ROOT / "models" / "generator_reflectance_best.pth"
CLOUDY_DIR = ROOT / "data" / "raw" / "cloudy"
CLEAR_DIR = ROOT / "data" / "raw" / "clear"
METRICS_CSV = ROOT / "outputs" / "reflectance_validation_metrics.csv"
COMPARISON_CSV = ROOT / "outputs" / "reflectance_model_comparison.csv"

# ----------------------------- Premium styling -------------------------------
st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {
  --bg:#070b11; --panel:#0d141e; --panel2:#111c29; --line:#243243;
  --text:#edf3f8; --muted:#8b9aab; --lime:#c7ff6b; --cyan:#70e6ec;
}
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp {
  color:var(--text);
  background:radial-gradient(ellipse at 78% -8%,rgba(54,126,153,.16),transparent 35%),
             radial-gradient(ellipse at 8% 22%,rgba(114,155,69,.06),transparent 31%),#070b11;
}
[data-testid="stHeader"] { background:rgba(7,11,17,.92); border-bottom:1px solid rgba(255,255,255,.04); }
[data-testid="stAppViewContainer"] > .main .block-container { max-width:1660px; padding:1.5rem 3rem 3.5rem; }
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
[data-testid="stTabs"] button[data-baseweb="tab"] {
  color:#92a0b0 !important; background:transparent !important; border:1px solid transparent !important;
  border-radius:7px 7px 0 0 !important; padding:12px 16px !important; font-size:.84rem !important; font-weight:700 !important;
}
[data-testid="stTabs"] button[data-baseweb="tab"]:hover { color:#fff !important; background:rgba(255,255,255,.035) !important; }
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
  color:var(--lime) !important; background:#101923 !important; border-color:#243243 !important; border-bottom-color:#101923 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--lime) !important; height:2px !important; }
.stButton > button[kind="primary"] {
  background:var(--lime) !important; color:#11180a !important; border:1px solid var(--lime) !important;
  border-radius:7px !important; min-height:46px; padding:0 22px !important; font-weight:800 !important; transition:all .18s ease;
}
.stButton > button[kind="primary"]:hover { background:#d7ff98 !important; border-color:#d7ff98 !important; box-shadow:0 0 28px rgba(199,255,107,.14); transform:translateY(-1px); }
.stButton > button,.stDownloadButton > button {
  background:#111a25 !important; color:#e3ebf3 !important; border:1px solid #2b3b4e !important;
  border-radius:7px !important; min-height:41px; font-weight:600 !important; transition:all .18s ease;
}
.stButton > button:hover,.stDownloadButton > button:hover { border-color:#758ba2 !important; background:#172432 !important; color:#fff !important; }
div[data-baseweb="select"] > div,div[data-baseweb="input"] > div,div[data-baseweb="textarea"] > div {
  background:#0d141e !important; border:1px solid #2a394a !important; border-radius:7px !important; box-shadow:none !important;
}
div[data-baseweb="select"] > div:focus-within,div[data-baseweb="input"] > div:focus-within,div[data-baseweb="textarea"] > div:focus-within {
  border-color:var(--cyan) !important; box-shadow:0 0 0 1px rgba(112,230,236,.12) !important;
}
input,textarea { color:var(--text) !important; }
[data-testid="stFileUploader"] section { background:#0c131d !important; border:1px dashed #35485b !important; border-radius:10px !important; padding:1rem !important; }
[data-testid="stFileUploader"] section:hover { border-color:var(--cyan) !important; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#111c29,#0b121b) !important; border:1px solid #263748 !important; border-radius:9px !important; padding:18px 20px !important; }
[data-testid="stMetricLabel"] { color:#91a0b2 !important; font-size:.77rem !important; font-weight:700 !important; }
[data-testid="stMetricValue"] { color:var(--lime) !important; font-family:'Space Grotesk',sans-serif !important; font-size:1.7rem !important; font-weight:700 !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:9px; overflow:hidden; }
[data-testid="stImage"] img { border:1px solid #263748; border-radius:8px; }
[data-testid="stAlert"] { border-radius:8px !important; }
[data-testid="stCode"] { background:#070c12 !important; border:1px solid #273747 !important; border-radius:8px !important; }
code,pre { font-family:'DM Mono',monospace !important; }
div[role="radiogroup"] { gap:13px; }
.stProgress > div > div { background:var(--lime) !important; }
.ss-hero {
  display:grid; grid-template-columns:minmax(0,1.08fr) minmax(300px,.92fr); gap:20px; align-items:center;
  position:relative; overflow:hidden; min-height:340px; padding:28px 32px;
  border:1px solid #263544; border-radius:14px;
  background:linear-gradient(115deg,rgba(15,24,36,.98),rgba(8,14,22,.96) 54%,rgba(9,24,32,.93));
  box-shadow:0 22px 70px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.035);
}
.ss-hero:before {
  content:''; position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);
  background-size:38px 38px; mask-image:linear-gradient(90deg,black,transparent 86%);
}
.ss-hero-copy { position:relative; z-index:2; }
.ss-kicker { font:500 10px 'DM Mono',monospace; letter-spacing:3px; color:var(--lime); margin-bottom:20px; }
.ss-wordmark { color:#f1f6fb; font:700 clamp(36px,4.2vw,64px) 'Space Grotesk',sans-serif; line-height:.99; letter-spacing:-3.5px; }
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
.satellite { transform-origin:250px 155px; animation:orbit 18s linear infinite; }
@keyframes orbit { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
.ss-section-label { color:var(--lime); font:10px 'DM Mono',monospace; letter-spacing:2px; text-transform:uppercase; }
.ss-panel { border:1px solid var(--line); background:linear-gradient(145deg,rgba(17,28,41,.82),rgba(10,16,24,.88)); padding:18px 20px; border-radius:10px; }
.ss-small { color:#8495a8; font:10px 'DM Mono',monospace; letter-spacing:1px; }
@media (max-width:900px) {
 [data-testid="stAppViewContainer"] > .main .block-container { padding:1rem 1rem 2rem; }
 .ss-hero { grid-template-columns:1fr; padding:22px; }
 .ss-visual svg { max-height:280px; }
 .ss-wordmark { letter-spacing:-2px; }
}
</style>
""",
    unsafe_allow_html=True,
)


def make_hero():
    # Inline SVG gives the page a lightweight, self-contained 3D-inspired orbital scene.
    svg = """
    <svg viewBox="0 0 500 330" role="img" aria-label="Illustration of a satellite orbiting Earth">
      <defs>
        <radialGradient id="earth" cx="32%" cy="25%" r="78%">
          <stop offset="0%" stop-color="#75e8d6"/>
          <stop offset="42%" stop-color="#238b9a"/>
          <stop offset="78%" stop-color="#15506d"/>
          <stop offset="100%" stop-color="#0b1b31"/>
        </radialGradient>
        <linearGradient id="panel" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#78f1ed"/><stop offset="100%" stop-color="#24558b"/>
        </linearGradient>
        <linearGradient id="gold" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#e8ff9a"/><stop offset="100%" stop-color="#83ad45"/>
        </linearGradient>
        <clipPath id="earthClip"><circle cx="250" cy="180" r="86"/></clipPath>
        <filter id="glow"><feGaussianBlur stdDeviation="8"/></filter>
      </defs>
      <circle cx="250" cy="180" r="105" fill="#41c9d2" opacity=".08" filter="url(#glow)"/>
      <ellipse class="orbit-ring" cx="250" cy="180" rx="205" ry="78" transform="rotate(-22 250 180)"/>
      <ellipse class="orbit-ring" cx="250" cy="180" rx="170" ry="116" transform="rotate(34 250 180)" opacity=".35"/>
      <circle cx="250" cy="180" r="86" fill="url(#earth)" stroke="#83e5df" stroke-opacity=".55"/>
      <g clip-path="url(#earthClip)" opacity=".9">
        <path d="M167 155 L185 137 207 141 219 158 211 172 224 186 209 204 190 197 183 178 168 173Z" fill="#75b88d" opacity=".82"/>
        <path d="M230 108 L250 119 263 139 251 153 267 168 260 184 274 198 265 226 248 246 233 221 239 201 225 183 232 163 216 146Z" fill="#9ad29d" opacity=".8"/>
        <path d="M288 128 L309 136 326 157 315 175 299 171 290 190 276 181 282 158Z" fill="#5ba985" opacity=".78"/>
        <path d="M169 155 Q213 138 249 161 T332 150" fill="none" stroke="#b6f3d8" stroke-width="2" opacity=".24"/>
        <path d="M172 208 Q219 184 257 208 T329 199" fill="none" stroke="#b6f3d8" stroke-width="2" opacity=".2"/>
        <path d="M173 176 Q245 191 326 171 M180 225 Q250 244 317 218" fill="none" stroke="#8be7df" stroke-width="1" opacity=".24"/>
        <path d="M199 97 Q182 150 205 205 T243 263 M278 96 Q300 150 281 208 T263 263" fill="none" stroke="#9ae7e4" stroke-width="1" opacity=".2"/>
      </g>
      <path d="M182 127 A86 86 0 0 1 330 150" fill="none" stroke="#d0fff1" stroke-width="2" opacity=".4"/>
      <g class="satellite">
        <g transform="translate(250 155) rotate(-20)">
          <path d="M-80 -13 L-34 -13 L-34 13 L-80 13 Z" fill="url(#panel)" stroke="#a8f7f1" stroke-width="1"/>
          <path d="M34 -13 L80 -13 L80 13 L34 13 Z" fill="url(#panel)" stroke="#a8f7f1" stroke-width="1"/>
          <g stroke="#153d5d" stroke-width="1">
            <path d="M-68 -13 V13 M-56 -13 V13 M-44 -13 V13 M-80 0 H-34 M46 -13 V13 M58 -13 V13 M70 -13 V13 M34 0 H80"/>
          </g>
          <path d="M-30 -18 L-14 -24 L17 -18 L28 -7 L23 14 L0 21 L-24 12 L-30 -18Z" fill="url(#gold)" stroke="#d8ff91" stroke-width="1.3"/>
          <path d="M-15 -17 L0 -30 L15 -17" fill="#b6dc67" stroke="#e7ffb3" stroke-width="1.5"/>
          <rect x="-8" y="-8" width="17" height="19" rx="2" fill="#273d4d" stroke="#d4f5a4"/>
          <circle cx="1" cy="1" r="5" fill="#6ce5e7" stroke="#e6ffff" stroke-width="1.2"/>
          <path d="M-1 18 L-1 31 M-10 31 H8" stroke="#b7d4dc" stroke-width="2"/>
        </g>
      </g>
      <g fill="#d9f9ff">
        <circle cx="82" cy="87" r="1.7"/><circle cx="398" cy="78" r="1.3"/><circle cx="408" cy="244" r="1.8"/>
        <circle cx="102" cy="259" r="1.2"/><circle cx="362" cy="45" r="1.4"/><circle cx="138" cy="51" r="1"/>
        <circle cx="430" cy="164" r="1.2"/><circle cx="63" cy="182" r="1.1"/>
      </g>
      <path d="M330 91 L375 66 L416 66" fill="none" stroke="#6ce5e7" stroke-width="1" opacity=".65"/>
      <circle cx="330" cy="91" r="3" fill="#c7ff6b"/>
      <text x="379" y="58" fill="#c7ff6b" font-size="9" font-family="monospace" letter-spacing="1">ORBITAL LINK</text>
      <path d="M158 245 L119 269 L77 269" fill="none" stroke="#6ce5e7" stroke-width="1" opacity=".6"/>
      <circle cx="158" cy="245" r="3" fill="#6ce5e7"/>
      <text x="77" y="286" fill="#8ca1b2" font-size="9" font-family="monospace" letter-spacing="1">EARTH OBSERVATION</text>
    </svg>
    """
    st.markdown(
        f"""
        <section class="ss-hero">
          <div class="ss-hero-copy">
            <div class="ss-kicker">EARTH OBSERVATION / INTELLIGENCE SYSTEMS</div>
            <div class="ss-wordmark">SECURESAT<span>.</span>GAN</div>
            <div class="ss-subtitle">
              Satellite image restoration with a trained U-Net generator and
              SHA-256 integrity verification for geospatial data workflows.
            </div>
            <div class="ss-chips">
              <span class="ss-chip">U-NET RESTORATION</span>
              <span class="ss-chip">MULTISPECTRAL INPUT</span>
              <span class="ss-chip">SHA-256 INTEGRITY</span>
            </div>
            <div class="ss-live"><span class="ss-dot"></span> LOCAL SYSTEM · {DEVICE.type.upper()} COMPUTE</div>
          </div>
          <div class="ss-visual">
            {svg}
            <div class="ss-visual-caption"><span>ORBITAL IMAGING</span><span>SECURE DATA PIPELINE</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading trained restoration model...")
def load_model(checkpoint_path: str, device_name: str):
    model = UNetGenerator(features=16).to(torch.device(device_name))
    state = torch.load(checkpoint_path, map_location=device_name, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def file_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def dataset_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))


def read_satellite_rgb(file_obj_or_path):
    """Read bands 4, 3, 2, normalize to reflectance and return model tensor + preview."""
    if hasattr(file_obj_or_path, "read"):
        raw = file_obj_or_path.read()
        source = rasterio.io.MemoryFile(raw)
        ds = source.open()
    else:
        source = None
        ds = rasterio.open(file_obj_or_path)

    try:
        if ds.count < 4:
            raise ValueError(f"Expected at least 4 raster bands; found {ds.count}.")
        # Match training preprocessing: bands [4, 3, 2], divide by 10000, clip to [0,1].
        rgb = ds.read([4, 3, 2]).astype(np.float32) / 10000.0
        rgb = np.clip(rgb, 0.0, 1.0)
        if rgb.shape[1:] != (256, 256):
            raise ValueError(f"Expected a 256 x 256 GeoTIFF; found {rgb.shape[2]} x {rgb.shape[1]}.")
        tensor = torch.from_numpy(rgb * 2.0 - 1.0).float()
        preview = np.transpose(rgb, (1, 2, 0))
        # Percentile stretch is only for display; model input remains reflectance-scaled.
        lo, hi = np.percentile(preview, [2, 98])
        if hi > lo:
            display = np.clip((preview - lo) / (hi - lo), 0, 1)
        else:
            display = preview
        display = (display * 255).astype(np.uint8)
        return tensor, Image.fromarray(display), ds.profile.copy()
    finally:
        ds.close()
        if source is not None:
            source.close()


def tensor_to_preview(tensor):
    arr = tensor.detach().cpu().float().clamp(-1, 1).numpy()
    arr = np.transpose((arr + 1.0) / 2.0, (1, 2, 0))
    lo, hi = np.percentile(arr, [2, 98])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def model_inference(input_tensor):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT}")
    model = load_model(str(CHECKPOINT), str(DEVICE))
    with torch.inference_mode():
        prediction = model(input_tensor.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
    return prediction


def get_validation_metrics():
    if METRICS_CSV.exists():
        return pd.read_csv(METRICS_CSV)
    return pd.DataFrame()


# ----------------------------- Sidebar ---------------------------------------
with st.sidebar:
    st.markdown("## 🛰️ SECURESAT")
    st.caption("EARTH OBSERVATION INTELLIGENCE")
    st.divider()
    st.markdown("**SYSTEM STATUS**")
    st.markdown(f"🟢 Compute: `{DEVICE.type.upper()}`")
    st.markdown(f"{'🟢' if CHECKPOINT.exists() else '🔴'} Model checkpoint")
    st.markdown(f"{'🟢' if CLOUDY_DIR.exists() else '🟠'} Cloudy dataset")
    st.markdown(f"{'🟢' if CLEAR_DIR.exists() else '🟠'} Clear reference dataset")
    st.divider()
    st.caption("RESTORATION")
    st.write("Trained U-Net · RGB reflectance bands")
    st.caption("INTEGRITY")
    st.write("SHA-256 file digest comparison")
    st.divider()
    st.caption("DEMONSTRATION BUILD · LOCAL PROCESSING")


make_hero()

# Compact overview metrics
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("RESTORATION MODEL", "U-Net", "Trained checkpoint" if CHECKPOINT.exists() else "Checkpoint missing")
with m2:
    st.metric("COMPUTE DEVICE", DEVICE.type.upper(), "CUDA available" if DEVICE.type == "cuda" else "CPU inference")
with m3:
    st.metric("CLOUDY PATCHES", len(dataset_files(CLOUDY_DIR)), "Local dataset")
with m4:
    st.metric("CLEAR PATCHES", len(dataset_files(CLEAR_DIR)), "Reference dataset")

cloud_tab, validation_tab, integrity_tab, about_tab = st.tabs(
    ["01  CLOUD REMOVAL", "02  VALIDATION", "03  INTEGRITY CHECK", "04  PROJECT"]
)

# ----------------------------- Cloud removal ---------------------------------
with cloud_tab:
    st.markdown('<div class="ss-section-label">01 / IMAGE RESTORATION</div>', unsafe_allow_html=True)
    st.header("Restore a cloudy satellite image")
    st.write("Run the trained U-Net on a 256 × 256 multispectral GeoTIFF. Uploads are processed locally by this app.")

    source_choice = st.radio(
        "Image source",
        ["Existing dataset", "Upload GeoTIFF"],
        horizontal=True,
        key="cloud_source_choice",
    )

    selected_path = None
    uploaded_tif = None

    if source_choice == "Existing dataset":
        available = dataset_files(CLOUDY_DIR)
        if not available:
            st.warning("No .tif or .tiff files found in data/raw/cloudy.")
        else:
            names = [p.name for p in available]
            selected_name = st.selectbox("Choose a cloudy dataset patch", names)
            selected_path = CLOUDY_DIR / selected_name
            matching_clear = CLEAR_DIR / selected_name
            if matching_clear.exists():
                st.caption(f"Matching clear reference available: {matching_clear.name}")
            else:
                st.caption("No same-name clear reference was found for this patch.")
    else:
        uploaded_tif = st.file_uploader(
            "Upload a 256 × 256 multispectral GeoTIFF",
            type=["tif", "tiff"],
            key="cloud_geotiff_upload",
        )

    run_disabled = (selected_path is None) if source_choice == "Existing dataset" else (uploaded_tif is None)
    if st.button("RUN CLOUD RESTORATION", type="primary", disabled=run_disabled, use_container_width=False):
        try:
            if source_choice == "Existing dataset":
                input_tensor, input_preview, profile = read_satellite_rgb(selected_path)
                current_name = selected_path.name
            else:
                input_tensor, input_preview, profile = read_satellite_rgb(uploaded_tif)
                current_name = uploaded_tif.name

            with st.spinner("Running U-Net restoration on the selected image..."):
                prediction = model_inference(input_tensor)
                output_preview = tensor_to_preview(prediction)

            st.session_state["last_input_preview"] = input_preview
            st.session_state["last_output_preview"] = output_preview
            st.session_state["last_input_tensor"] = input_tensor
            st.session_state["last_prediction"] = prediction
            st.session_state["last_image_name"] = current_name
            st.session_state["last_profile"] = profile
            st.success(f"Restoration completed for {current_name}.")
        except Exception as exc:
            st.error(f"Restoration failed: {exc}")

    if "last_output_preview" in st.session_state:
        st.divider()
        st.markdown("### Processing results")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**CLOUDY INPUT**")
            st.image(st.session_state["last_input_preview"], use_container_width=True)
        with c2:
            st.markdown("**RESTORED OUTPUT**")
            st.image(st.session_state["last_output_preview"], use_container_width=True)

        st.caption(f"File: {st.session_state.get('last_image_name', 'uploaded image')}")
        output_buffer = io.BytesIO()
        output_array = st.session_state["last_prediction"].detach().cpu().float().clamp(-1, 1).numpy()
        output_array = ((output_array + 1.0) / 2.0 * 10000.0).round().clip(0, 10000).astype(np.uint16)
        output_profile = st.session_state["last_profile"].copy()
        output_profile.update(
            count=3,
            dtype="uint16",
            height=output_array.shape[1],
            width=output_array.shape[2],
        )
        with rasterio.MemoryFile() as memfile:
            with memfile.open(**output_profile) as dst:
                dst.write(output_array)
            output_buffer.write(memfile.read())
        st.download_button(
            "DOWNLOAD RESTORED 3-BAND TIFF",
            data=output_buffer.getvalue(),
            file_name=f"restored_{Path(st.session_state.get('last_image_name', 'satellite.tif')).stem}.tif",
            mime="image/tiff",
        )
        st.caption("The restored file contains the model's three output bands (RGB order). Original geospatial profile metadata is retained where supported.")

# ----------------------------- Validation ------------------------------------
with validation_tab:
    st.markdown('<div class="ss-section-label">02 / MODEL EVALUATION</div>', unsafe_allow_html=True)
    st.header("Validation performance")
    st.write("Metrics below are read from the project's saved validation CSV. They are not recalculated by this dashboard.")
    metrics_df = get_validation_metrics()

    if not metrics_df.empty:
        expected_cols = {"input_mae", "output_mae", "input_psnr", "output_psnr", "input_ssim", "output_ssim"}
        if expected_cols.issubset(metrics_df.columns):
            avg = metrics_df[list(expected_cols)].mean(numeric_only=True)
            a, b, c = st.columns(3)
            with a:
                st.metric("MEAN MAE", f"{avg['output_mae']:.5f}", f"Input {avg['input_mae']:.5f}")
            with b:
                st.metric("MEAN PSNR", f"{avg['output_psnr']:.2f} dB", f"Input {avg['input_psnr']:.2f} dB")
            with c:
                st.metric("MEAN SSIM", f"{avg['output_ssim']:.4f}", f"Input {avg['input_ssim']:.4f}")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.download_button(
            "DOWNLOAD VALIDATION CSV",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="reflectance_validation_metrics.csv",
            mime="text/csv",
        )
    else:
        st.info(f"Validation metrics file not found: {METRICS_CSV}")

    st.divider()
    st.subheader("Model comparison")
    if COMPARISON_CSV.exists():
        comparison = pd.read_csv(COMPARISON_CSV)
        if {"model", "mae", "psnr", "ssim", "sharpness"}.issubset(comparison.columns):
            summary = comparison.groupby("model", as_index=False)[["mae", "psnr", "ssim", "sharpness"]].mean()
            st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.caption("Optional model comparison CSV is not available yet.")

# ----------------------------- Integrity check -------------------------------
with integrity_tab:
    st.markdown('<div class="ss-section-label">03 / DATA TRUST</div>', unsafe_allow_html=True)
    st.header("SHA-256 file integrity verification")
    st.write("Calculate a SHA-256 digest for a file and optionally compare it with a previously recorded trusted digest.")
    st.info("A matching digest indicates that the file bytes match the supplied reference. It does not independently prove who created the file or whether the reference itself is trustworthy.")

    integrity_file = st.file_uploader(
        "Choose a file to verify",
        key="integrity_file_upload",
    )
    reference_hash = st.text_input(
        "Trusted reference SHA-256 (optional)",
        placeholder="Paste a previously recorded 64-character digest",
        max_chars=64,
    ).strip().lower()

    if integrity_file is not None:
        file_bytes = integrity_file.getvalue()
        digest = file_sha256(file_bytes)
        st.markdown(f"**File:** `{integrity_file.name}`")
        st.code(digest, language=None)

        if reference_hash:
            if len(reference_hash) != 64 or any(ch not in "0123456789abcdef" for ch in reference_hash):
                st.error("The reference must be exactly 64 hexadecimal characters.")
            elif digest == reference_hash:
                st.success("INTEGRITY MATCH — the file digest matches the supplied reference.")
            else:
                st.error("INTEGRITY MISMATCH — the file digest differs from the supplied reference.")

        report = (
            "SecureSat-GAN SHA-256 Integrity Report\n"
            "======================================\n"
            f"File name: {integrity_file.name}\n"
            f"File size (bytes): {len(file_bytes)}\n"
            f"SHA-256: {digest}\n"
            f"Reference SHA-256: {reference_hash or 'Not supplied'}\n"
            f"Result: {'NOT CHECKED - no reference supplied' if not reference_hash else ('MATCH' if digest == reference_hash else 'MISMATCH')}\n"
            "\nNote: SHA-256 compares file bytes to the supplied digest; it does not prove the provenance of the reference.\n"
        )
        st.download_button(
            "DOWNLOAD INTEGRITY REPORT",
            data=report.encode("utf-8"),
            file_name="securesat_integrity_report.txt",
            mime="text/plain",
        )
    else:
        st.caption("Upload any file to calculate its SHA-256 digest.")

# -------------------------------- About ---------------------------------------
with about_tab:
    st.markdown('<div class="ss-section-label">04 / SYSTEM OVERVIEW</div>', unsafe_allow_html=True)
    st.header("About SecureSat-GAN")
    st.write(
        "SecureSat-GAN is a demonstration project that combines satellite-image cloud restoration "
        "with file-integrity verification. A trained U-Net generates a three-channel restored image "
        "from selected multispectral GeoTIFF bands."
    )
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
            <div class="ss-panel">
              <div class="ss-small">RESTORATION PIPELINE</div>
              <h3>U-Net image reconstruction</h3>
              <p>Reads bands 4, 3, and 2, applies the same reflectance scaling used during training, and predicts a restored three-channel output.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="ss-panel">
              <div class="ss-small">SECURITY PIPELINE</div>
              <h3>SHA-256 integrity check</h3>
              <p>Computes a file digest and compares it with a user-provided trusted reference hash to detect byte-level differences.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Important: the orbital illustration is decorative. The restoration output comes from the local trained checkpoint. Results depend on input quality and training data.")
    st.caption(f"Project root: {ROOT}")
