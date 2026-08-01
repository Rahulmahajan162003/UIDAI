import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim


DEFAULT_BASE_URL = "http://127.0.0.1:5001"
TIMEOUT_SECONDS = 90
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
LOGO_PATH = REPO_ROOT / "assets" / "images" / "logo11.png"
SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "webp"]


def load_logo_base64() -> str:
    if not LOGO_PATH.exists():
        return ""
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")


def post_image(base_url: str, endpoint: str, uploaded_file, data=None):
    uploaded_file.seek(0)
    files = {
        "image": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    return requests.post(
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        files=files,
        data=data,
        timeout=TIMEOUT_SECONDS,
    )


def post_named_images(base_url: str, endpoint: str, uploaded_files: dict[str, object], data: dict = None):
    files = {}
    for field_name, uploaded_file in uploaded_files.items():
        uploaded_file.seek(0)
        files[field_name] = (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    return requests.post(
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        files=files,
        data=data,
        timeout=TIMEOUT_SECONDS,
    )


def decode_base64_image(data: str):
    try:
        return Image.open(BytesIO(base64.b64decode(data)))
    except Exception:
        return None


def find_base64_image(payload):
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str):
                image = decode_base64_image(value)
                if image is not None:
                    return key, image
            elif isinstance(value, dict):
                nested = find_base64_image(value)
                if nested is not None:
                    return nested
            elif isinstance(value, list):
                for item in value:
                    nested = find_base64_image(item)
                    if nested is not None:
                        return nested
    return None


def clean_payload_for_display(payload):
    if payload is None:
        return None
    if isinstance(payload, dict):
        cleaned = {}
        for k, v in payload.items():
            if isinstance(v, str) and (len(v) > 200 or k in ["processed", "visualization", "cropped", "template"]):
                cleaned[k] = f"<Base64 Truncated, Length: {len(v)}>"
            else:
                cleaned[k] = clean_payload_for_display(v)
        return cleaned
    elif isinstance(payload, list):
        return [clean_payload_for_display(item) for item in payload]
    else:
        return payload


def parse_json_response(response):
    try:
        payload = response.json()
    except ValueError:
        st.error(f"Backend returned non-JSON response (HTTP {response.status_code}).")
        st.code(response.text)
        return None

    if response.ok:
        return payload

    message = payload.get("error") or payload.get("message") or "Request failed"
    st.error(f"{message} (HTTP {response.status_code})")
    with st.expander("Backend response"):
        st.json(clean_payload_for_display(payload))
    return None


def read_uploaded_bgr(uploaded_file):
    uploaded_file.seek(0)
    data = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def compute_orb_matches(image_a_gray, image_b_gray):
    orb = cv2.ORB_create(500)
    _, des1 = orb.detectAndCompute(image_a_gray, None)
    _, des2 = orb.detectAndCompute(image_b_gray, None)
    if des1 is None or des2 is None:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    return sum(1 for m in matches if m.distance < 50)


def compute_sift_matches(image_a_gray, image_b_gray):
    if not hasattr(cv2, "SIFT_create"):
        return None
    sift = cv2.SIFT_create()
    _, des1 = sift.detectAndCompute(image_a_gray, None)
    _, des2 = sift.detectAndCompute(image_b_gray, None)
    if des1 is None or des2 is None:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return len(good_matches)


def compute_local_similarity(contact_file, contactless_file):
    contact_bgr = read_uploaded_bgr(contact_file)
    contactless_bgr = read_uploaded_bgr(contactless_file)
    if contact_bgr is None or contactless_bgr is None:
        return None

    contact_gray = cv2.cvtColor(contact_bgr, cv2.COLOR_BGR2GRAY)
    contactless_gray = cv2.cvtColor(contactless_bgr, cv2.COLOR_BGR2GRAY)
    target_h = min(contact_gray.shape[0], contactless_gray.shape[0])
    target_w = min(contact_gray.shape[1], contactless_gray.shape[1])
    if target_h == 0 or target_w == 0:
        return None

    contact_gray = cv2.resize(contact_gray, (target_w, target_h))
    contactless_gray = cv2.resize(contactless_gray, (target_w, target_h))
    ssim_score = ssim(contact_gray, contactless_gray, data_range=255)

    return {
        "ssim": round(float(ssim_score), 4),
        "orb_matches": int(compute_orb_matches(contact_gray, contactless_gray)),
        "sift_matches": compute_sift_matches(contact_gray, contactless_gray),
    }


def metric_card(label: str, value: str, hint: str = "", tone: str = "neutral"):
    st.markdown(
        f"""
        <div class="metric-card metric-card--{tone}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_breakdown(pipeline_metrics):
    if not pipeline_metrics or "steps" not in pipeline_metrics:
        return
    
    steps = pipeline_metrics["steps"]
    inference_sum = pipeline_metrics.get("model_inference_sum_ms", 0.0)
    overhead = pipeline_metrics.get("system_overhead_ms", 0.0)
    total_time = pipeline_metrics.get("total_time_ms", 0.0)
    
    step_rows = ""
    for step in steps:
        step_rows += (
            f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px dashed rgba(184, 120, 0, 0.12);">'
            f'<span style="font-weight: 600; color: inherit; font-size: 0.92rem;">{step["step"]}</span>'
            f'<span style="font-family: monospace; font-weight: 800; color: #b87800; font-size: 0.92rem;">{step["time_ms"]:.1f} ms</span>'
            f'</div>'
        )
        
    html_content = (
        f'<div style="font-family: \'DM Sans\', sans-serif;">'
        f'<div style="display: flex; flex-direction: column; gap: 2px; margin-bottom: 12px;">'
        f'{step_rows}'
        f'</div>'
        f'<div style="background: rgba(184, 120, 0, 0.04); border-radius: 10px; padding: 10px 14px; border: 1px solid rgba(184, 120, 0, 0.08); margin-top: 10px;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.88rem; padding: 3px 0;">'
        f'<span style="font-weight: 600; color: var(--muted, #5b6470);">Model-inference sum</span>'
        f'<span style="font-family: monospace; font-weight: 800; color: inherit;">~{inference_sum/1000.0:.2f} s</span>'
        f'</div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.88rem; padding: 3px 0;">'
        f'<span style="font-weight: 600; color: var(--muted, #5b6470);">System Overhead</span>'
        f'<span style="font-family: monospace; font-weight: 800; color: #d32f2f;">+{overhead:.1f} ms</span>'
        f'</div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.94rem; font-weight: 800; border-top: 1px solid rgba(184, 120, 0, 0.18); margin-top: 8px; padding-top: 8px;">'
        f'<span style="color: inherit;">Aggregate projection</span>'
        f'<span style="font-family: monospace; color: #2e7d32;">~{total_time/1000.0:.2f} s</span>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    with st.expander("⏱️ Latency & Pipeline Breakdown", expanded=False):
        st.markdown(html_content, unsafe_allow_html=True)


def grade_colors(grade: str):
    palette = {
        "Excellent": ("#1e6f45", "#e7f6ed"),
        "Good": ("#8c5a07", "#fff1d8"),
        "Marginal": ("#a66218", "#fff4e8"),
        "Rejected": ("#b33a33", "#fdecea"),
    }
    return palette.get(grade or "", ("#475467", "#f2f4f7"))


def build_breakdown_rows(breakdown: dict):
    rows = []
    if not isinstance(breakdown, dict):
        return rows

    for key, value in breakdown.items():
        if isinstance(value, bool):
            rows.append({"Metric": key.replace("_", " ").title(), "Score": 1.0 if value else 0.0})
            continue
        try:
            rows.append({"Metric": key.replace("_", " ").title(), "Score": float(value)})
        except (TypeError, ValueError):
            continue
    return rows


def inject_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;800&display=swap');

            :root {
                --bg: #f7f2e8;
                --panel: rgba(255, 252, 247, 0.96);
                --panel-strong: #fffdfa;
                --stroke: rgba(184, 120, 0, 0.16);
                --stroke-strong: rgba(184, 120, 0, 0.30);
                --text: #1b1d21;
                --muted: #5b6470;
                --amber: #e0a020;
                --amber-deep: #b87800;
                --amber-soft: #fff3d7;
                --cream: #fffaf3;
                --green: #1e6f45;
                --red: #b33a33;
                --shadow: 0 14px 34px rgba(110, 79, 19, 0.08);
                --radius-lg: 18px;
                --radius-md: 12px;
            }

            .stApp {
                background:
                    linear-gradient(180deg, #fffaf3 0%, #f7f2e8 48%, #f2ecdf 100%);
                color: var(--text);
                font-family: 'DM Sans', sans-serif;
            }

            .block-container {
                max-width: 1440px;
                padding-top: 1rem;
                padding-bottom: 2rem;
            }

            #MainMenu,
            footer,
            header[data-testid="stHeader"] {
                visibility: hidden;
            }

            h1, h2, h3, h4, h5, p, label {
                font-family: 'DM Sans', sans-serif !important;
                letter-spacing: 0;
            }

            span:not([data-testid*="Icon"]):not([class*="Icon"]):not([class*="Material"]),
            div:not([data-testid*="Icon"]):not([class*="Icon"]):not([class*="Material"]) {
                font-family: 'DM Sans', sans-serif !important;
            }

            .tool-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                min-height: 84px;
                padding: 1rem 1.2rem;
                margin-bottom: 1rem;
                border-radius: var(--radius-lg);
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(255, 247, 231, 0.98));
                border: 1px solid var(--stroke);
                box-shadow: var(--shadow);
            }

            .brand-lockup {
                display: flex;
                align-items: center;
                gap: 0.9rem;
                min-width: 0;
            }

            .brand-logo {
                width: 56px;
                height: 56px;
                flex: 0 0 auto;
                border-radius: 12px;
                box-shadow: 0 8px 18px rgba(224, 160, 32, 0.18);
            }

            .brand-title {
                margin: 0;
                color: var(--text);
                font-size: 1.72rem;
                line-height: 1.08;
                font-weight: 800;
            }

            .brand-subtitle {
                margin-top: 0.15rem;
                color: #8a5b11;
                font-size: 0.9rem;
                font-weight: 700;
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.5rem 0.72rem;
                border-radius: 999px;
                background: #fff3d7;
                border: 1px solid rgba(184, 120, 0, 0.18);
                color: #6e4700;
                font-size: 0.85rem;
                font-weight: 800;
                white-space: nowrap;
            }

            [data-baseweb="tab-list"] {
                gap: 0.55rem;
                margin-bottom: 1rem;
                padding: 0.34rem;
                border-radius: 16px;
                background: rgba(255, 252, 247, 0.84);
                border: 1px solid rgba(184, 120, 0, 0.12);
                width: fit-content;
            }

            [data-baseweb="tab"] {
                height: 2.72rem;
                padding: 0 0.95rem;
                border-radius: 999px;
                background: transparent;
                border: 1px solid transparent;
                color: #7a5922 !important;
                font-weight: 800;
            }

            [data-baseweb="tab"][aria-selected="true"] {
                background: linear-gradient(135deg, rgba(224, 160, 32, 0.22), rgba(255, 241, 214, 0.98));
                border-color: rgba(184, 120, 0, 0.18);
                color: #4f3407 !important;
                box-shadow: 0 6px 14px rgba(184, 120, 0, 0.10);
            }

            [data-baseweb="tab"] p,
            [data-baseweb="tab"] span,
            [data-baseweb="tab"] div {
                color: inherit !important;
            }

            .section-card,
            .workflow-panel,
            .metric-card,
            .result-frame,
            .uploader-shell {
                border-radius: var(--radius-md);
                border: 1px solid var(--stroke);
                box-shadow: var(--shadow);
            }

            .section-card {
                padding: 1.15rem 1.2rem;
                margin-bottom: 0.9rem;
                background: var(--panel);
            }

            .section-title {
                margin: 0 0 0.25rem;
                color: var(--text);
                font-size: 1.14rem;
                line-height: 1.18;
                font-weight: 800;
            }

            .section-copy {
                margin: 0;
                color: var(--muted);
                line-height: 1.48;
                font-size: 0.95rem;
            }

            .workflow-stack {
                display: flex;
                flex-direction: column;
                gap: 0.9rem;
            }

            .workflow-panel {
                padding: 1.05rem;
                background: var(--panel);
            }

            .workflow-panel h4 {
                margin: 0 0 0.3rem;
                color: var(--text);
                font-size: 0.98rem;
                font-weight: 800;
            }

            .workflow-panel p {
                margin: 0;
                color: var(--muted);
                line-height: 1.48;
                font-size: 0.92rem;
            }

            .action-note {
                margin-top: 0.55rem;
                color: #86612a;
                font-size: 0.86rem;
                line-height: 1.45;
            }

            .uploader-shell {
                padding: 0.95rem;
                background: rgba(255, 254, 250, 0.92);
            }

            [data-testid="stFileUploader"] {
                border: 1px dashed rgba(184, 120, 0, 0.34);
                border-radius: var(--radius-md);
                background: rgba(255, 252, 247, 0.84);
                padding: 0.32rem;
            }

            [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] {
                display: none !important;
            }

            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploaderDropzone"] {
                background: #fffdfa !important;
                border: 1px solid rgba(184, 120, 0, 0.18) !important;
                border-radius: 10px !important;
                color: var(--text) !important;
                min-height: 4.8rem !important;
                padding: 0.9rem !important;
                display: flex !important;
                align-items: center !important;
                gap: 0.85rem !important;
                flex-wrap: wrap !important;
            }

            [data-testid="stFileUploader"] section div,
            [data-testid="stFileUploader"] section span,
            [data-testid="stFileUploader"] section p,
            [data-testid="stFileUploader"] section small {
                color: var(--muted) !important;
            }

            [data-testid="stFileUploader"] button {
                background: #1b1d21 !important;
                color: #fffdfa !important;
                border: 1px solid rgba(27, 29, 33, 0.12) !important;
                border-radius: 8px !important;
                min-width: 6.4rem !important;
                min-height: 2.45rem !important;
                padding: 0.55rem 1rem !important;
                font-weight: 800 !important;
                line-height: 1 !important;
                box-shadow: 0 10px 18px rgba(27, 29, 33, 0.12) !important;
            }

            [data-testid="stFileUploader"] button div,
            [data-testid="stFileUploader"] button span,
            [data-testid="stFileUploader"] button p {
                color: #fffdfa !important;
                margin: 0 !important;
                line-height: 1 !important;
                white-space: nowrap !important;
            }

            [data-testid="stFileUploader"] [data-testid="stIconMaterial"] {
                display: none !important;
            }

            [data-testid="stFileUploaderDropzoneInstructions"] {
                flex: 1 1 13rem !important;
                min-width: 11rem !important;
            }

            .stButton > button {
                width: 100%;
                min-height: 2.82rem;
                border: 0;
                border-radius: 12px;
                padding: 0.78rem 1rem;
                background: linear-gradient(135deg, #d98f0f 0%, #f0b537 100%);
                color: #1b1d21;
                font-weight: 800;
                box-shadow: 0 12px 24px rgba(217, 143, 15, 0.18);
            }

            .stButton > button:hover {
                color: #1b1d21;
                border: 0;
                filter: brightness(0.98);
            }

            .metric-card {
                min-height: 118px;
                padding: 0.95rem;
                background: var(--panel-strong);
            }

            .metric-card--success {
                border-color: rgba(30, 111, 69, 0.22);
                background: linear-gradient(135deg, #f7fff9, #e9f8ef);
            }

            .metric-card--danger {
                border-color: rgba(179, 58, 51, 0.22);
                background: linear-gradient(135deg, #fff8f6, #ffece7);
            }

            .metric-label {
                color: #8a5b11;
                font-size: 0.77rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .metric-value {
                margin-top: 0.32rem;
                color: var(--text);
                font-size: 1.72rem;
                line-height: 1.05;
                font-weight: 800;
                overflow-wrap: anywhere;
            }

            .metric-hint {
                margin-top: 0.36rem;
                color: var(--muted);
                font-size: 0.86rem;
                line-height: 1.36;
            }

            .grade-chip {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 0.48rem 0.72rem;
                font-size: 0.92rem;
                font-weight: 800;
            }

            .result-frame {
                padding: 0.82rem;
                background: rgba(255, 255, 255, 0.82);
            }

            .result-frame img {
                border-radius: 10px;
            }

            .empty-state {
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 220px;
                padding: 1.15rem;
                border-radius: var(--radius-md);
                background: rgba(255, 253, 248, 0.94);
                border: 1px dashed rgba(184, 120, 0, 0.22);
                color: var(--muted);
                text-align: center;
                line-height: 1.55;
            }

            [data-testid="stAlert"] {
                border-radius: 12px;
                border-width: 1px;
                border-style: solid;
                box-shadow: 0 8px 18px rgba(110, 79, 19, 0.06);
            }

            [data-testid="stAlert"] p,
            [data-testid="stAlert"] div,
            [data-testid="stAlert"] span,
            [data-testid="stAlert"] label {
                color: var(--text) !important;
            }

            [data-testid="stAlert"][kind="warning"] {
                background: linear-gradient(135deg, #fff7cc, #fff1ad);
                border-color: rgba(184, 120, 0, 0.24);
            }

            [data-testid="stAlert"][kind="success"] {
                background: linear-gradient(135deg, #eef9f1, #ddf2e5);
                border-color: rgba(30, 111, 69, 0.20);
            }

            [data-testid="stAlert"][kind="info"] {
                background: linear-gradient(135deg, #eef5ff, #e3efff);
                border-color: rgba(67, 118, 194, 0.18);
            }

            [data-testid="stAlert"][kind="error"] {
                background: linear-gradient(135deg, #fff0ed, #ffe5df);
                border-color: rgba(179, 58, 51, 0.20);
            }

            [data-testid="stJson"] {
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid rgba(184, 120, 0, 0.12);
            }

            /* Premium styled expander overrides for YellowSense theme */
            [data-testid="stExpander"] {
                border: 1px solid rgba(184, 120, 0, 0.28) !important;
                border-radius: 12px !important;
                background: linear-gradient(135deg, #fffdfa 0%, #fffbf2 100%) !important;
                box-shadow: 0 6px 18px rgba(110, 79, 19, 0.05) !important;
                margin-top: 1rem !important;
                margin-bottom: 1.2rem !important;
                overflow: hidden !important;
            }

            [data-testid="stExpander"] summary {
                background: linear-gradient(135deg, rgba(255, 247, 231, 0.6) 0%, rgba(224, 160, 32, 0.08) 100%) !important;
                border-bottom: 1px solid rgba(184, 120, 0, 0.08) !important;
            }

            [data-testid="stExpander"] summary:hover {
                background: linear-gradient(135deg, rgba(255, 247, 231, 0.9) 0%, rgba(224, 160, 32, 0.15) 100%) !important;
            }

            [data-testid="stExpander"] summary svg {
                fill: #b87800 !important;
                color: #b87800 !important;
            }

            [data-testid="stExpander"] summary p {
                font-weight: 800 !important;
                color: #4f3407 !important;
                font-family: 'DM Sans', sans-serif !important;
            }

            [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
                padding: 1.1rem 1.25rem !important;
                background: #fffdfa !important;
            }

            .app-footer {
                margin-top: 1.2rem;
                padding: 0.9rem 0 0.25rem;
                text-align: center;
                color: #8a6a33;
                font-size: 0.86rem;
            }

            @media (max-width: 900px) {
                .tool-header {
                    align-items: flex-start;
                    flex-direction: column;
                }

                [data-baseweb="tab-list"] {
                    width: 100%;
                    overflow-x: auto;
                }

                .brand-title {
                    font-size: 1.42rem;
                }

                .status-pill {
                    white-space: normal;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(base_url: str):
    logo_b64 = load_logo_base64()
    logo_html = ""
    if logo_b64:
        logo_html = f"<img class='brand-logo' src='data:image/png;base64,{logo_b64}' alt='YellowSense logo' />"

    st.markdown(
        f"""
        <section class="tool-header">
            <div class="brand-lockup">
                {logo_html}
                <div>
                    <h1 class="brand-title">UIDAI SITAA Desktop Tool</h1>
                    <div class="brand-subtitle">YellowSense fingerprint quality console</div>
                </div>
            </div>
            <div class="status-pill">Backend: {base_url}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, copy: str):
    st.markdown(
        f"""
        <div class="section-card">
            <h3 class="section-title">{title}</h3>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_intro(title: str, copy: str, note: str):
    st.markdown(
        f"""
        <div class="workflow-panel">
            <h4>{title}</h4>
            <p>{copy}</p>
            <div class="action-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quality_tab(base_url: str):
    render_section(
        "Fingerprint Image Quality",
        "Upload a fingerprint image to run blur, brightness, glare, and capture-readiness checks.",
    )
    left, right = st.columns([0.88, 1.35], gap="large")

    with left:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        render_upload_intro(
            "Input Image",
            "Select one candidate capture for quality screening.",
            "Supported formats: PNG, JPG, JPEG, BMP and WEBP.",
        )
        st.markdown("<div class='uploader-shell'>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Quality image",
            type=SUPPORTED_IMAGE_TYPES,
            key="quality_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.markdown("<div class='result-frame'>", unsafe_allow_html=True)
            st.image(uploaded, caption="Uploaded fingerprint image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        run = st.button("Run quality check", key="run_quality")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='workflow-panel'><h4>Quality Result</h4><p>Capture guidance and backend metrics appear here after analysis.</p></div>",
            unsafe_allow_html=True,
        )
        if not run:
            st.markdown(
                "<div class='empty-state'>Upload an image and run the quality check to review the capture result.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if uploaded is None:
            st.warning("Upload a fingerprint image before running the quality check.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        with st.spinner("Submitting image to /quality_check..."):
            try:
                response = post_image(base_url, "/quality_check", uploaded)
            except requests.RequestException as exc:
                st.error(f"Could not reach /quality_check: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

        payload = parse_json_response(response)
        if payload is None:
            st.markdown("</div>", unsafe_allow_html=True)
            return

        blur = payload.get("blur", {})
        brightness = payload.get("brightness", {})
        glare = payload.get("glare", {})
        passed = bool(payload.get("passed"))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Status", "Pass" if passed else "Fail", payload.get("guidance", ""), "success" if passed else "danger")
        with col2:
            metric_card("Blur", str(blur.get("blur_score", "N/A")), "Higher is sharper")
        with col3:
            metric_card("Brightness", str(brightness.get("brightness", "N/A")), "Mean luminance")
        with col4:
            metric_card("Glare", "Yes" if glare.get("has_glare") else "No", glare.get("method", "Method unavailable"))

        render_pipeline_breakdown(payload.get("pipeline_metrics"))

        detail_col, data_col = st.columns([1, 1.15], gap="large")
        with detail_col:
            st.markdown("#### Quality Flags")
            issues = payload.get("issues") or []
            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("No blocking quality issues were reported.")
        with data_col:
            st.markdown("#### Response Payload")
            st.json(clean_payload_for_display(payload))
        st.markdown("</div>", unsafe_allow_html=True)


def render_readiness_tab(base_url: str):
    render_section(
        "Fingerprint Readiness Assessment",
        "Score a fingerprint image on a 0 to 100 readiness scale with grade and metric breakdown.",
    )
    left, right = st.columns([0.88, 1.35], gap="large")

    with left:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        render_upload_intro(
            "Input Image",
            "Select one fingerprint image for readiness scoring.",
            "Use a clear, centered finger image for the most meaningful score.",
        )
        st.markdown("<div class='uploader-shell'>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Fingerprint image",
            type=SUPPORTED_IMAGE_TYPES,
            key="readiness_upload",
            label_visibility="collapsed",
        )
        validate_liveness = st.checkbox(
            "Enforce Liveness Validation",
            value=False,
            key="readiness_validate_liveness",
            help="If enabled, a spoof/fake detection caps the readiness score to a maximum of 30."
        )
        if uploaded is not None:
            st.markdown("<div class='result-frame'>", unsafe_allow_html=True)
            st.image(uploaded, caption="Uploaded fingerprint image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        run = st.button("Run readiness assessment", key="run_readiness")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='workflow-panel'><h4>Assessment Result</h4><p>Readiness score, grade, and metric summary appear here.</p></div>",
            unsafe_allow_html=True,
        )
        if not run:
            st.markdown(
                "<div class='empty-state'>Upload an image and run the assessment to view the readiness summary.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if uploaded is None:
            st.warning("Upload a fingerprint image before running the assessment.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        with st.spinner("Submitting image to /readiness..."):
            try:
                data = {"validate_liveness": "true" if validate_liveness else "false"}
                response = post_image(base_url, "/readiness", uploaded, data=data)
            except requests.RequestException as exc:
                st.error(f"Could not reach /readiness: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

        payload = parse_json_response(response)
        if payload is None:
            st.markdown("</div>", unsafe_allow_html=True)
            return

        score = payload.get("readiness_score", payload.get("score"))
        grade = payload.get("grade", "Unknown")
        breakdown = payload.get("breakdown", {})
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = None

        is_live = payload.get("is_live")
        liveness_conf = payload.get("liveness_confidence")

        grade_fg, grade_bg = grade_colors(grade)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            metric_card("Readiness", f"{score_value:.1f}" if score_value is not None else "N/A", "0 to 100 scale")
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Grade</div>
                    <div style="margin-top:0.75rem;">
                        <span class="grade-chip" style="color:{grade_fg}; background:{grade_bg};">{grade}</span>
                    </div>
                    <div class="metric-hint">Backend readiness band</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            metric_card("Metrics", str(len(breakdown) if isinstance(breakdown, dict) else 0), "Breakdown items")
        with col4:
            if is_live is not None:
                live_text = "Live" if is_live else "Spoof"
                live_fg, live_bg = ("#1e6f45", "#e7f6ed") if is_live else ("#b33a33", "#fdecea")
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Liveness</div>
                        <div style="margin-top:0.75rem;">
                            <span class="grade-chip" style="color:{live_fg}; background:{live_bg};">{live_text}</span>
                        </div>
                        <div class="metric-hint">Liveness check status</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                metric_card("Liveness", "N/A", "Liveness check status")
        with col5:
            if liveness_conf is not None:
                metric_card("Liveness Conf", f"{liveness_conf*100:.1f}%", "Confidence probability")
            else:
                metric_card("Liveness Conf", "N/A", "Confidence probability")

        if score_value is not None:
            st.progress(min(max(score_value / 100.0, 0.0), 1.0), text=f"Overall readiness: {score_value:.1f}/100")

        render_pipeline_breakdown(payload.get("pipeline_metrics"))

        chart_col, data_col = st.columns([1.15, 1], gap="large")
        with chart_col:
            st.markdown("#### Quality Breakdown")
            rows = build_breakdown_rows(breakdown)
            if rows:
                st.bar_chart(rows, x="Metric", y="Score", horizontal=True, color="#D98F0F")
            else:
                st.info("No numeric breakdown values were returned.")
        with data_col:
            st.markdown("#### Response Payload")
            st.json(clean_payload_for_display(payload))
        st.markdown("</div>", unsafe_allow_html=True)


def render_comparison_tab(base_url: str):
    render_section(
        "Contact vs Contactless Comparison",
        "Compare a contact fingerprint image with a contactless capture using backend similarity and local image metrics.",
    )
    left, right = st.columns([1.02, 1.22], gap="large")

    with left:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        render_upload_intro(
            "Comparison Pair",
            "Choose one contact image and one contactless image of the same finger.",
            "Matching samples make SSIM, ORB, and SIFT scores meaningful.",
        )
        contact = st.file_uploader("Contact image", type=SUPPORTED_IMAGE_TYPES, key="compare_contact_upload")
        contactless = st.file_uploader("Contactless image", type=SUPPORTED_IMAGE_TYPES, key="compare_contactless_upload")
        if contact is not None or contactless is not None:
            preview_left, preview_right = st.columns(2)
            with preview_left:
                if contact is not None:
                    st.markdown("<div class='result-frame'>", unsafe_allow_html=True)
                    st.image(contact, caption="Contact", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            with preview_right:
                if contactless is not None:
                    st.markdown("<div class='result-frame'>", unsafe_allow_html=True)
                    st.image(contactless, caption="Contactless", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        high_sec = st.checkbox("Enforce UIDAI High-Security Mode (Matches >= 12, Score >= 25%)", value=False, key="high_sec_compare")
        run = st.button("Run comparison", key="run_comparison")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='workflow-panel'><h4>Comparison Result</h4><p>Similarity metrics and backend response appear here after comparison.</p></div>",
            unsafe_allow_html=True,
        )
        if not run:
            st.markdown(
                "<div class='empty-state'>Upload both images and run comparison to view SSIM, ORB, and SIFT results.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if contact is None or contactless is None:
            st.warning("Upload both a contact image and a contactless image before running comparison.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        with st.spinner("Submitting images to /compare_contact..."):
            try:
                response = post_named_images(
                    base_url,
                    "/compare_contact",
                    {"contact": contact, "contactless": contactless},
                    data={"high_security": "true" if high_sec else "false"}
                )
            except requests.RequestException as exc:
                st.error(f"Could not reach /compare_contact: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

        payload = parse_json_response(response)
        if payload is None:
            st.markdown("</div>", unsafe_allow_html=True)
            return

        comparison = payload.get("comparison", {})
        ssim_val = comparison.get("ssim", 0.0)
        orb_val = comparison.get("orb_matches", 0)
        sift_val = comparison.get("sift_matches")
        sift_str = "N/A" if sift_val is None else str(sift_val)
        backend_score = comparison.get("similarity_score", 0.0)
        
        contact_minutiae = comparison.get("contact_minutiae_count", "N/A")
        contactless_minutiae = comparison.get("contactless_minutiae_count", "N/A")
        matched_minutiae = comparison.get("matched_minutiae_count", 0)
        verdict = comparison.get("verdict", "FAIL")
        high_sec_mode_active = comparison.get("high_security_mode", False)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Verdict", verdict, "Match verdict status", "success" if verdict == "PASS" else "danger")
        with col2:
            metric_card("Matched Minutiae", str(matched_minutiae), "Matching minutiae count")
        with col3:
            metric_card("Minutiae Counts", f"{contact_minutiae} / {contactless_minutiae}", "Contact / Contactless count")
        with col4:
            metric_card("Match Score", f"{backend_score:.1f}%", "MCC template similarity")



        with st.expander("Show Image-Level Similarity Metrics (SSIM, ORB, SIFT)"):
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                metric_card("SSIM", f"{ssim_val:.4f}", "Structural similarity")
            with col_s2:
                metric_card("ORB Matches", str(orb_val), "Local keypoint matches")
            with col_s3:
                metric_card("SIFT Matches", sift_str, "Local keypoint matches")

        render_pipeline_breakdown(payload.get("pipeline_metrics"))

        chart_col, data_col = st.columns([1.3, 0.95], gap="large")
        with chart_col:
            st.markdown("#### Minutiae Matching Alignment")
            images = payload.get("images", {})
            vis_b64 = images.get("comparison_visualization")
            if vis_b64:
                vis_img = decode_base64_image(vis_b64)
                if vis_img is not None:
                    st.markdown("<div class='result-frame'>", unsafe_allow_html=True)
                    st.image(vis_img, caption="Minutiae Alignment (Left: Contact, Right: Contactless). Green points and connecting lines show matched minutiae pairs; red points show unmatched minutiae.", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("Failed to decode minutiae matching visualization.")
            else:
                st.info("No matching visualization available in response.")
            
            st.markdown(
                "##### 💡 How to Read the Alignment Map\n"
                "* **Parallel Green Lines**: Connect matching minutiae pairs. A dense set of parallel lines indicates that the spatial distribution of landmarks is highly consistent across both prints, indicating a high-confidence match.\n"
                "* **Green Points**: Represent corresponding features (Ridge Endings or Bifurcations) found in both captures.\n"
                "* **Red Points**: Represent local features that were either obscured, not captured, or not present in the other scan due to resolution/optical bounds."
            )
        with data_col:
            st.markdown("#### Metric Summary")
            rows = [
                {"Metric": "SSIM (x100)", "Value": float(ssim_val * 100)},
                {"Metric": "ORB Matches", "Value": float(orb_val)},
                {"Metric": "SIFT Matches", "Value": float(sift_val or 0)},
                {"Metric": "Match Score (%)", "Value": float(backend_score)},
                {"Metric": "Matched Minutiae", "Value": float(matched_minutiae)},
            ]
            st.bar_chart(rows, x="Metric", y="Value", horizontal=True, color="#D98F0F")
            
            st.markdown("#### Response Payload")
            st.json(clean_payload_for_display(payload))
        st.markdown("</div>", unsafe_allow_html=True)


def render_contact_tab(base_url: str):
    render_section(
        "Contact Image Processing",
        "Preprocess a contact fingerprint image and inspect extracted minutiae with the processed preview.",
    )
    left, right = st.columns([0.88, 1.35], gap="large")

    with left:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        render_upload_intro(
            "Contact Image",
            "Select a contact fingerprint image for backend preprocessing.",
            "Use a clear single-frame scan with visible ridge structure.",
        )
        st.markdown("<div class='uploader-shell'>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Contact fingerprint image",
            type=SUPPORTED_IMAGE_TYPES,
            key="contact_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.markdown("<div class='result-frame'>", unsafe_allow_html=True)
            st.image(uploaded, caption="Uploaded contact image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        run = st.button("Run contact processing", key="run_contact")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='workflow-panel'><h4>Processing Result</h4><p>Processed preview, minutiae count, and response details appear here.</p></div>",
            unsafe_allow_html=True,
        )
        if not run:
            st.markdown(
                "<div class='empty-state'>Upload a contact image and run processing to view the output.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if uploaded is None:
            st.warning("Upload a contact fingerprint image before running this workflow.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        with st.spinner("Submitting image to /process_contact..."):
            try:
                response = post_image(base_url, "/process_contact", uploaded)
            except requests.RequestException as exc:
                st.error(f"Could not reach /process_contact: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

        payload = parse_json_response(response)
        if payload is None:
            st.markdown("</div>", unsafe_allow_html=True)
            return

        metric_card("Minutiae", str(payload.get("minutiae_count", "N/A")), "Detected contact fingerprint points")

        render_pipeline_breakdown(payload.get("pipeline_metrics"))

        found_image = find_base64_image(payload)
        preview_col, result_col = st.columns([1.12, 1], gap="large")
        with preview_col:
            st.markdown("#### Processed Preview")
            if found_image is None:
                st.info("The response did not contain a decodable base64 image.")
            else:
                image_key, image = found_image
                st.markdown("<div class='result-frame'>", unsafe_allow_html=True)
                st.image(image, caption=f"Decoded image from `{image_key}`", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        with result_col:
            st.markdown("#### Response Payload")
            st.json(clean_payload_for_display(payload))
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    page_icon = Image.open(LOGO_PATH) if LOGO_PATH.exists() else None
    st.set_page_config(
        page_title="UIDAI SITAA Desktop Tool",
        page_icon=page_icon,
        layout="wide",
    )

    inject_styles()
    base_url = DEFAULT_BASE_URL
    render_header(base_url)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Tab 1: Image Quality",
            "Tab 2: Readiness",
            "Tab 3: Comparison",
            "Tab 4: Contact Processing",
        ]
    )

    with tab1:
        render_quality_tab(base_url)
    with tab2:
        render_readiness_tab(base_url)
    with tab3:
        render_comparison_tab(base_url)
    with tab4:
        render_contact_tab(base_url)

    st.markdown(
        "<div class='app-footer'>YellowSense Technologies - SITAA Cohort 1 - 2025</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
