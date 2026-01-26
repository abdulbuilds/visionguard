import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
from streamlit_lottie import st_lottie
import requests
import streamlit.components.v1 as components
from labels import FINAL_LABELS

# ==========================================
# 1. INITIAL CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="VisionGuard AI | Pro",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "history" not in st.session_state:
    st.session_state.history = []

IMG_SIZE = 64
MODEL_PATH = "model_fixed.keras"
CONFIDENCE_THRESHOLD = 70.0

# ==========================================
# 2. CORE LOGIC
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except: return None

model = load_model()
lottie_radar = load_lottieurl("https://lottie.host/8038755b-427d-419b-9807-63a562453c0d/fN1R20Kq1X.json")

# WEB-COMPATIBLE VOICE BROADCAST
def speak_web(text):
    js_code = f"""
        <script>
        var msg = new SpeechSynthesisUtterance('{text}');
        window.speechSynthesis.speak(msg);
        </script>
    """
    components.html(js_code, height=0)

def preprocess_image(img):
    img_res = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.array(img_res) / 255.0, axis=0)

# ==========================================
# 3. DYNAMIC STYLING
# ==========================================
def apply_styles(bar_color="#00f2fe"):
    st.markdown(f"""
        <style>
        @keyframes gradient {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        .stApp {{
            background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            color: white;
            transition: all 0.5s ease;
        }}
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(30px) scale(0.95); }}
            to {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        .result-card {{
            background: rgba(255, 255, 255, 0.07);
            padding: 30px;
            border-radius: 20px;
            border: 1px solid rgba(0, 242, 254, 0.3);
            text-align: center;
            backdrop-filter: blur(15px);
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            animation: slideIn 0.6s cubic-bezier(0.23, 1, 0.32, 1);
        }}
        .sign-text {{
            font-size: 3.2rem; font-weight: 800; color: #00f2fe; 
            text-shadow: 0 0 15px rgba(0,242,254,0.5);
            display: block; margin: 15px 0;
        }}
        section[data-testid="stSidebar"] {{
            background-color: rgba(15, 12, 41, 0.95) !important;
            backdrop-filter: blur(15px);
        }}
        div[data-testid="stProgress"] > div > div > div > div {{
            background-color: {bar_color} !important;
            transition: width 0.8s ease-in-out;
        }}
        </style>
        """, unsafe_allow_html=True)

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🕒 Activity Log")
    st.markdown("---")
    if not st.session_state.history:
        st.caption("Awaiting detections...")
    else:
        for item in reversed(st.session_state.history[-10:]):
            cols = st.columns([1, 3])
            if item['thumb']: cols[0].image(item['thumb'], width=50)
            with cols[1]:
                st.markdown(f"**{item['label']}**")
                st.caption(f"{item['conf']:.1f}% | {item['time']}")
            st.markdown("<div style='margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.05);'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ==========================================
# 5. MAIN UI
# ==========================================
st.title("🚦 VisionGuard Pro")
st.caption("AI-Powered Traffic Safety & Recognition System")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📥 Input Feed")
    tabs = st.tabs(["📁 File Upload", "📷 Live Camera"])
    img_final = None
    
    with tabs[0]:
        f = st.file_uploader("Upload", type=["jpg","png","jpeg"], key="up_final", label_visibility="collapsed")
        if f: img_final = Image.open(f)
    with tabs[1]:
        c = st.camera_input("Capture", key="cam_final", label_visibility="collapsed")
        if c: img_final = Image.open(c)
    
    if img_final:
        st.image(img_final, use_container_width=True, caption="Analyzed Image")

with col_right:
    st.subheader("🧠 Intelligence Output")
    output_placeholder = st.empty()

    if img_final:
        # Prediction
        batch = preprocess_image(img_final)
        preds = model.predict(batch, verbose=0)
        cid = np.argmax(preds)
        conf = float(np.max(preds)) * 100
        label = FINAL_LABELS[cid]
        
        # Color & Style
        bar_color = "#00ff88" if conf >= 85 else "#ffee00" if conf >= 60 else "#ff4b4b"
        apply_styles(bar_color)
        
        # History
        t_now = datetime.now().strftime("%H:%M:%S")
        thumb = img_final.copy()
        thumb.thumbnail((80, 80))
        if not st.session_state.history or st.session_state.history[-1]['label'] != label:
            st.session_state.history.append({"label": label, "conf": conf, "time": t_now, "thumb": thumb})

        with output_placeholder.container():
            st.markdown(f"""
                <div class="result-card">
                    <p style="color:gray; letter-spacing:2px; margin-bottom:0; font-size:0.8rem;">REAL-TIME DETECTION</p>
                    <span class="sign-text">{label}</span>
                    <p style="color:{bar_color}; font-weight:bold;">Confidence: {conf:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            st.progress(conf/100)
            if conf >= CONFIDENCE_THRESHOLD:
                st.success("✅ Match Verified")
                if st.button("🔊 Voice Broadcast"):
                    speak_web(f"Detection confirmed. This is a {label} sign.")
            else:
                st.warning("⚠️ Analysis Uncertain - Check Clarity")
    else:
        apply_styles("#00f2fe")
        with output_placeholder.container():
            if lottie_radar:
                st_lottie(lottie_radar, height=250, key="radar_vFinal")
            st.info("System Standby. Waiting for visual input...")

st.markdown("---")
st.caption("Final Version | Web-Compatible Voice | Streamlit Cloud Ready")
