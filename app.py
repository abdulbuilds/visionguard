import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go
import plotly.express as px
from labels import FINAL_LABELS
import pandas as pd
import json
import base64
from gtts import gTTS
import io

# ==========================================
# 1. INITIAL CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="VisionGuard AI | Pro",
    page_icon="🚦",
    layout="wide"
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

IMG_SIZE = 64
MODEL_PATH = "model_fixed.keras"
CONFIDENCE_THRESHOLD = 70.0

# Online audio URLs (replace with your hosted audio files or use placeholder)
AUDIO_URLS = {
    "high": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",  # Success sound
    "medium": "https://assets.mixkit.co/active_storage/sfx/2571/2571-preview.mp3",  # Notification
    "low": "https://assets.mixkit.co/active_storage/sfx/2572/2572-preview.mp3"  # Error beep
}

def create_3d_sphere(confidence, label_name):
    """Enhanced 3D sphere with better visual feedback"""
    color = "#00ff88" if confidence >= 85 else "#ffee00" if confidence >= 70 else "#ff4b4b"
    
    phi = np.linspace(0, 2*np.pi, 30)
    theta = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(phi), np.sin(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.ones(np.size(phi)), np.cos(theta))

    fig = go.Figure()

    # Glowing shell
    fig.add_trace(go.Surface(
        x=x, y=y, z=z, 
        opacity=0.15, 
        showscale=False, 
        colorscale=[[0, '#00f2fe'], [1, '#00f2fe']],
        hoverinfo='skip'
    ))

    # Core pulse point
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=20, color=color, opacity=0.7, symbol='diamond', 
                    line=dict(width=5, color='white')),
        name="Neural Core"
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='cube'
        ),
        showlegend=False,
        height=350,
        annotations=[dict(
            text=f"<b>{label_name.upper()}</b>", 
            showarrow=False, 
            font=dict(color=color, size=16), 
            yref="paper", 
            y=0.1
        )]
    )
    return fig

def create_probability_chart(predictions, top_n=5):
    """Create a horizontal bar chart showing top N predictions"""
    # Get top N predictions
    top_indices = np.argsort(predictions[0])[-top_n:][::-1]
    top_labels = [FINAL_LABELS[i] for i in top_indices]
    top_probs = [predictions[0][i] * 100 for i in top_indices]
    
    # Create color gradient
    colors = ['#00ff88' if p >= 70 else '#ffee00' if p >= 50 else '#ff4b4b' for p in top_probs]
    
    fig = go.Figure(go.Bar(
        x=top_probs,
        y=top_labels,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[f'{p:.1f}%' for p in top_probs],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,12,41,0.3)',
        font=dict(color='white', size=12),
        xaxis=dict(
            title='Confidence (%)',
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, 100]
        ),
        yaxis=dict(
            title='',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        height=300,
        margin=dict(l=150, r=50, t=20, b=40),
        showlegend=False
    )
    
    return fig

def play_voice_condition(conf_level):
    """Play audio based on confidence level using online URLs"""
    if conf_level >= 90:
        audio_url = AUDIO_URLS["high"]
    elif conf_level >= 70:
        audio_url = AUDIO_URLS["medium"]
    else:
        audio_url = AUDIO_URLS["low"]
    
    # Auto-play audio using HTML5 audio element
    md = f"""
        <audio autoplay="true" style="display:none;">
            <source src="{audio_url}" type="audio/mp3">
        </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

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
    except: 
        return None

model = load_model()

def speak(text):
    """Text-to-speech using gTTS (works on Streamlit Cloud!)"""
    try:
        # Generate speech using Google TTS
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Convert to base64
        audio_bytes = audio_buffer.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        # Play using HTML5 audio
        audio_html = f"""
            <audio autoplay="true" style="display:none;">
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except:
        pass  # Silently fail if gTTS unavailable

def preprocess_image(img):
    """Preprocess image for model input"""
    img_res = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.array(img_res) / 255.0, axis=0)

def detect_image_type(img):
    """Simple heuristic to detect image type"""
    # Convert to numpy array
    img_array = np.array(img.convert("RGB"))
    
    # Calculate basic statistics
    mean_color = np.mean(img_array)
    std_color = np.std(img_array)
    
    # Very dark or very bright images
    if mean_color < 30:
        return "Very dark image"
    elif mean_color > 225:
        return "Very bright/white image"
    
    # Low contrast (likely uniform/blank)
    if std_color < 20:
        return "Low contrast / blank image"
    
    # Check if image has face-like features (very basic)
    # This is a placeholder - you'd need a proper face detection model for accuracy
    h, w = img_array.shape[:2]
    if h > 100 and w > 100:
        center_region = img_array[h//3:2*h//3, w//3:2*w//3]
        if np.std(center_region) > 40:
            return "Possible human/object (not a traffic sign)"
    
    return "Unknown non-sign image"

# ==========================================
# 3. DYNAMIC STYLING WITH THEME SUPPORT
# ==========================================
def apply_styles(bar_color="#00f2fe", theme="dark"):
    """Apply dynamic styles with theme support"""
    bg_gradient = "linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29)" if theme == "dark" else "linear-gradient(-45deg, #e0eafc, #cfdef3, #e0eafc, #ffffff)"
    text_color = "white" if theme == "dark" else "#1a1a1a"
    card_bg = "rgba(255, 255, 255, 0.07)" if theme == "dark" else "rgba(255, 255, 255, 0.9)"
    
    st.markdown(f"""
        <style>
        @keyframes gradient {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        .stApp {{
            background: {bg_gradient};
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            color: {text_color};
        }}

        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(30px) scale(0.95); }}
            to {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}

        .result-card {{
            background: {card_bg};
            padding: 30px;
            border-radius: 20px;
            border: 1px solid rgba(0, 242, 254, 0.3);
            text-align: center;
            backdrop-filter: blur(15px);
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            animation: slideIn 0.6s cubic-bezier(0.23, 1, 0.32, 1);
        }}

        .sign-text {{
            font-size: 3.2rem; 
            font-weight: 800; 
            color: #00f2fe; 
            text-shadow: 0 0 15px rgba(0,242,254,0.5);
            display: block; 
            margin: 15px 0;
        }}

        section[data-testid="stSidebar"] {{
            background-color: rgba(15, 12, 41, 0.9) !important;
            backdrop-filter: blur(15px);
        }}

        div[data-testid="stProgress"] > div > div > div > div {{
            background-color: {bar_color} !important;
            transition: width 0.8s ease-in-out;
        }}
        
        /* Simplified radio button styling */
        div[role="radiogroup"] {{
            background: rgba(0, 242, 254, 0.1) !important;
            padding: 12px !important;
            border-radius: 15px !important;
            border: 1px solid rgba(0, 242, 254, 0.3) !important;
        }}
        
        div[role="radiogroup"] label[data-checked="true"] {{
            color: #00f2fe !important;
            text-shadow: 0 0 10px rgba(0, 242, 254, 0.5);
        }}
        </style>
        """, unsafe_allow_html=True)

# ==========================================
# 4. SIDEBAR - ACTIVITY LOG + EXPORT
# ==========================================
with st.sidebar:
    st.title("🕒 Activity Log")
    
    # Theme toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    
    with col2:
        if st.button("🔊 Voice", use_container_width=True):
            if st.session_state.history:
                latest = st.session_state.history[-1]
                speak(f"Latest detection: {latest['label']} at {latest['conf']:.1f}% confidence")
            else:
                speak("No detections yet")
    
    st.markdown("---")
    
    if not st.session_state.history:
        st.caption("Awaiting detections...")
    else:
        for item in reversed(st.session_state.history[-10:]):
            cols = st.columns([1, 3])
            if item['thumb']: 
                cols[0].image(item['thumb'], width=50)
            with cols[1]:
                st.markdown(f"**{item['label']}**")
                st.caption(f"{item['conf']:.1f}% | {item['time']}")
            st.markdown("<div style='margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.05);'></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Export options
    st.subheader("📥 Export History")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 CSV", use_container_width=True):
            if st.session_state.history:
                df = pd.DataFrame([
                    {
                        'Label': h['label'],
                        'Confidence': f"{h['conf']:.2f}%",
                        'Time': h['time']
                    } for h in st.session_state.history
                ])
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    csv,
                    "visionguard_history.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    with col2:
        if st.button("📄 JSON", use_container_width=True):
            if st.session_state.history:
                json_data = json.dumps([
                    {
                        'label': h['label'],
                        'confidence': h['conf'],
                        'timestamp': h['time']
                    } for h in st.session_state.history
                ], indent=2)
                st.download_button(
                    "⬇️ Download JSON",
                    json_data,
                    "visionguard_history.json",
                    "application/json",
                    use_container_width=True
                )
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ==========================================
# 5. MAIN DASHBOARD UI
# ==========================================
st.title("🚦 VisionGuard Pro")
st.caption("Advanced Neural Traffic Recognition System | v5.0")

# Create tabs for different modes
tab1, tab2 = st.tabs(["🎯 Single Detection", "📦 Batch Processing"])

# ==========================================
# TAB 1: SINGLE DETECTION (Original UI)
# ==========================================
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📥 Input Selection")
        
        mode = st.radio(
            "Mode",
            ["📁 File Upload", "📷 Live Camera"],
            horizontal=True
        )

        img_final = None

        if "📁 File Upload" in mode:
            f_input = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"], key="f_v7")
            if f_input:
                img_final = Image.open(f_input)
        else:
            c_input = st.camera_input("Take photo", key="c_v7")
            if c_input:
                img_final = Image.open(c_input)

        if img_final:
            st.markdown("---")
            st.image(img_final, use_container_width=True, caption=f"Active Target ({mode})")
            if st.button("🗑️ Clear Selection"):
                st.rerun()

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
            
            # Confidence check
            if conf < CONFIDENCE_THRESHOLD:
                apply_styles("#ff4b4b", st.session_state.theme)
                with output_placeholder.container():
                    st.plotly_chart(create_3d_sphere(conf, "Low Clarity"), use_container_width=True, config={'displayModeBar': False})
                    
                    # Enhanced unknown detection
                    image_type = detect_image_type(img_final)
                    st.warning(f"⚠️ No traffic sign detected")
                    st.info(f"**Detected:** {image_type}")
                    st.info(f"**Signal Strength:** {conf:.1f}% (Required: {CONFIDENCE_THRESHOLD}%)")
                    
                    # Show probability distribution even for unknown
                    with st.expander("📊 View Probability Distribution"):
                        st.plotly_chart(create_probability_chart(preds, top_n=5), use_container_width=True)
                    
                    play_voice_condition(conf)
            else:
                # Success
                bar_color = "#00ff88" if conf >= 85 else "#ffee00"
                apply_styles(bar_color, st.session_state.theme)
                play_voice_condition(conf)
                
                # Wait a moment then speak the sign name
                import time
                time.sleep(0.5)  # Small delay so sound effect starts first
                speak(f"Detected: {label}")
                
                # History tracking - always add to show live updates
                t_now = datetime.now().strftime("%H:%M:%S")
                thumb = img_final.copy()
                thumb.thumbnail((80, 80))
                
                # Only prevent duplicate if it's the exact same detection within 2 seconds
                should_add = True
                if st.session_state.history:
                    last_item = st.session_state.history[-1]
                    if last_item['label'] == label and abs(float(last_item['conf']) - conf) < 1.0:
                        should_add = False
                
                if should_add:
                    st.session_state.history.append({
                        "label": label, 
                        "conf": conf, 
                        "time": t_now, 
                        "thumb": thumb
                    })
                
                with output_placeholder.container():
                    st.plotly_chart(create_3d_sphere(conf, label), use_container_width=True, config={'displayModeBar': False})

                    st.markdown(f"""
                        <div class="result-card">
                            <p style="color:gray; letter-spacing:2px; margin-bottom:0; font-size:0.8rem;">NEURAL MATCH FOUND</p>
                            <span class="sign-text">{label}</span>
                            <p style="color:{bar_color}; font-weight:bold;">Precision Index: {conf:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(conf/100)
                    
                    # Show detailed probability distribution
                    with st.expander("📊 View Full Probability Distribution"):
                        st.plotly_chart(create_probability_chart(preds, top_n=10), use_container_width=True)
        else:
            apply_styles("#00f2fe", st.session_state.theme)
            with output_placeholder.container():
                st.plotly_chart(create_3d_sphere(0, "Scanning..."), use_container_width=True, config={'displayModeBar': False})
                st.info("System Ready. Waiting for visual data...")

# ==========================================
# TAB 2: BATCH PROCESSING
# ==========================================
with tab2:
    st.subheader("📦 Batch Image Processing")
    st.caption("Upload multiple images for simultaneous detection")
    
    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if batch_files:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🚀 Process All Images", type="primary", use_container_width=True):
                st.session_state.batch_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(batch_files):
                    status_text.text(f"Processing {idx + 1}/{len(batch_files)}: {file.name}")
                    
                    img = Image.open(file)
                    batch = preprocess_image(img)
                    preds = model.predict(batch, verbose=0)
                    cid = np.argmax(preds)
                    conf = float(np.max(preds)) * 100
                    label = FINAL_LABELS[cid]
                    
                    # Create thumbnail
                    thumb = img.copy()
                    thumb.thumbnail((150, 150))
                    
                    st.session_state.batch_results.append({
                        'filename': file.name,
                        'label': label if conf >= CONFIDENCE_THRESHOLD else "Unknown",
                        'confidence': conf,
                        'image': thumb,
                        'passed': conf >= CONFIDENCE_THRESHOLD
                    })
                    
                    progress_bar.progress((idx + 1) / len(batch_files))
                
                status_text.text("✅ Batch processing complete!")
        
        # Display results
        if st.session_state.batch_results:
            st.markdown("---")
            st.subheader("📊 Batch Results")
            
            # Summary stats
            total = len(st.session_state.batch_results)
            passed = sum(1 for r in st.session_state.batch_results if r['passed'])
            failed = total - passed
            avg_conf = np.mean([r['confidence'] for r in st.session_state.batch_results if r['passed']])
            
            metric_cols = st.columns(4)
            metric_cols[0].metric("Total Images", total)
            metric_cols[1].metric("Detected Signs", passed, delta=f"{(passed/total)*100:.1f}%")
            metric_cols[2].metric("Unknown", failed, delta=f"{(failed/total)*100:.1f}%", delta_color="inverse")
            metric_cols[3].metric("Avg Confidence", f"{avg_conf:.1f}%" if passed > 0 else "N/A")
            
            st.markdown("---")
            
            # Results grid
            cols_per_row = 3
            for i in range(0, len(st.session_state.batch_results), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(st.session_state.batch_results):
                        result = st.session_state.batch_results[idx]
                        with cols[j]:
                            st.image(result['image'], use_container_width=True)
                            
                            status_color = "🟢" if result['passed'] else "🔴"
                            st.markdown(f"{status_color} **{result['label']}**")
                            st.caption(f"Confidence: {result['confidence']:.1f}%")
                            st.caption(f"File: {result['filename']}")
            
            # Export batch results
            st.markdown("---")
            if st.button("📥 Export Batch Results as CSV"):
                df = pd.DataFrame([
                    {
                        'Filename': r['filename'],
                        'Label': r['label'],
                        'Confidence': f"{r['confidence']:.2f}%",
                        'Status': 'Detected' if r['passed'] else 'Unknown'
                    } for r in st.session_state.batch_results
                ])
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    csv,
                    f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )

st.markdown("---")
st.caption("VisionGuard Pro v5.0 | Enhanced Edition with Batch Processing & Analytics")
