"""
=============================================================================
  TrafficSight AI  v2.0 (Gradio Version)
  Real-Time Traffic Sign Recognition using YOLOv8 + CNN
=============================================================================
"""

import gradio as gr
import cv2
import numpy as np
import time
import datetime
import os
import sys

from utils.detection import (
    load_model, run_inference, enhance_results_with_cnn, draw_detections,
    parse_results, get_color_for_category
)
from utils.tracker import TrackerState
from utils.voice_alert import VoiceAlertEngine

# Suppress TensorFlow logging and oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.layers import Dense
from labels import FINAL_LABELS
import base64

# Start voice engine
_voice = VoiceAlertEngine(rate=155, volume=1.0)
_voice.start()

# Monkey patch Dense __init__ to strip quantization_config and support newer keras models
original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init

# Load Models
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
try:
    yolo_model = load_model(MODEL_PATH)
    yolo_status = "🟢 Online"
except Exception as e:
    yolo_model = None
    yolo_status = "🔴 Offline"

try:
    cnn_model = load_keras_model("models/model_retrained.keras")
    cnn_status = "🟢 Online"
except Exception as e:
    cnn_model = None
    cnn_status = "🔴 Offline"

# CSS
css = """
body { background: #0f1117 !important; color: white !important; font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1400px !important; }

/* Disable default webcam mirroring */
video { transform: none !important; -webkit-transform: none !important; }

/* Hide Gradio default camera source select dropdown */
.gr-image select, .gr-image .camera-select, .gr-image button[aria-label*="camera" i], .gr-image button[aria-label*="source" i] {
    display: none !important;
}

/* Dashboard Header */
.top-bar {
    display: flex; justify-content: space-between; align-items: center;
    background: #1e2130; padding: 15px 25px; border-radius: 12px;
    border: 1px solid #2e3250; margin-bottom: 20px;
}
.top-bar-left { font-size: 1.5rem; font-weight: 700; display: flex; align-items: center; gap: 10px; }
.top-bar-right { display: flex; align-items: center; gap: 15px; }

.pill { padding: 5px 12px; border-radius: 50px; font-size: 0.85rem; font-weight: 600; }
.pill-green { background: rgba(0,212,170,0.2); color: #00d4aa; border: 1px solid rgba(0,212,170,0.4); }
.pill-red { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
.pill-gray { background: rgba(148,163,184,0.2); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.3); }

/* Recent Detections List */
.recent-list { max-height: 580px; overflow-y: auto; padding-right: 6px; }
.recent-card {
    display: flex; gap: 12px; background: #1e2130;
    border: 1px solid #2e3250; padding: 10px; border-radius: 12px; margin-bottom: 8px; align-items: center;
    animation: slideDownFade 0.3s ease-out forwards;
}
.recent-card img { border-radius: 4px; width: 48px; height: 48px; object-fit: cover; border: 1px solid rgba(250,250,250,0.1); }
.recent-info { display: flex; flex-direction: column; flex-grow: 1; min-width: 0; }
.recent-title { font-weight: 600; font-size: 0.9rem; overflow: hidden; white-space: nowrap; animation: typeWriter 0.4s steps(20, end) forwards; }
.recent-meta { font-size: 0.75rem; color: #94a3b8; display: flex; justify-content: space-between; margin-top: 4px; }
.conf-bar-bg { background: #2e3250; height: 4px; border-radius: 2px; width: 100%; margin-top: 4px; overflow: hidden; }
.conf-bar-fill { height: 100%; transform-origin: left; animation: fillBar 0.5s ease-out forwards; }
.empty-msg { color: #94a3b8; font-size: 0.85rem; padding: 10px; }

/* Bottom Analytics Strip */
.bottom-strip {
    background: #1e2130; padding: 15px 20px; border-radius: 12px;
    border: 1px solid #2e3250; margin-top: 20px; display: flex; align-items: center; gap: 20px;
}
.strip-total { font-size: 1.2rem; font-weight: 700; border-right: 1px solid #2e3250; padding-right: 20px; color: #fafafa; }
.strip-bars { display: flex; gap: 15px; flex-wrap: wrap; flex: 1; }
.mini-bar { display: flex; align-items: center; gap: 6px; font-size: 0.85rem; color: #e2e8f0; }
.color-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

/* Animations */
@keyframes slideDownFade {
    0% { opacity: 0; transform: translateY(-20px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes typeWriter {
    from { width: 0; }
    to { width: 100%; }
}
@keyframes fillBar {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
}

/* ===== Mobile Responsive ===== */
* { box-sizing: border-box; }

@media (max-width: 768px) {
    /* Container */
    .gradio-container { padding: 8px !important; }
    h1 { font-size: 1.2rem !important; }

    /* Stack columns vertically */
    .gr-row, .gr-group, .contain > .flex {
        flex-direction: column !important;
    }
    .gr-column { min-width: 100% !important; max-width: 100% !important; flex: 1 1 100% !important; }

    /* Full-width controls */
    .gr-button, button.primary, button.secondary, button.stop {
        width: 100% !important;
        margin-bottom: 6px !important;
    }
    .gr-slider, .gr-input, .gr-check-radio {
        width: 100% !important;
    }

    /* Video feed full width */
    .gr-image, .gr-image img, .gr-image video {
        width: 100% !important;
        height: auto !important;
    }

    /* Fix mobile camera rotation */
    video {
        transform: none !important;
        -webkit-transform: none !important;
        object-fit: cover !important;
    }
    img.svelte-1pijsyv, video.svelte-1pijsyv {
        transform: none !important;
    }

    /* Header compact */
    .top-bar {
        flex-direction: column; gap: 8px;
        padding: 10px 14px;
    }
    .top-bar-left { font-size: 1.1rem; }

    /* Detection cards compact */
    .recent-card {
        padding: 8px !important;
        gap: 8px;
    }
    .recent-card img { width: 36px; height: 36px; }
    .recent-title { font-size: 0.8rem; }
    .recent-meta { font-size: 0.65rem; }
    .recent-list { max-height: 300px; }

    /* Analytics strip compact */
    .bottom-strip {
        flex-direction: column;
        padding: 10px 12px;
        gap: 10px;
    }
    .strip-total {
        border-right: none;
        border-bottom: 1px solid #2e3250;
        padding-right: 0;
        padding-bottom: 10px;
        font-size: 1rem;
    }
    .strip-bars { gap: 8px; }
    .mini-bar { font-size: 0.75rem; }
}
"""

def generate_html_recent(recent_list):
    html = '<div class="recent-list">'
    if not recent_list:
        html += '<div class="empty-msg">No recent detections.</div>'
    for item in recent_list:
        color = '#3b82f6'
        color_tuple = get_color_for_category(item['class'])
        if color_tuple: # Convert BGR to hex
            color = f"#{color_tuple[2]:02x}{color_tuple[1]:02x}{color_tuple[0]:02x}"
            
        conf = item['conf']
        if conf > 70: bar_color = "#00d4aa"
        elif conf > 40: bar_color = "#f59e0b"
        else: bar_color = "#ef4444"
        
        html += f'''
        <div class="recent-card card-{item["track_id"]}">
            <img src="data:image/png;base64,{item["img_b64"]}" alt="{item["class"]}">
            <div class="recent-info">
                <span class="recent-title" style="color:{color}">{item["class"]}</span>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="background:{bar_color}; width:{conf}%;"></div>
                </div>
                <div class="recent-meta">
                    <span>{conf}%</span>
                    <span>{item["time"]}</span>
                </div>
            </div>
        </div>
        '''
    html += '</div>'
    return html

def generate_html_strip(counts, total):
    html = f'<div class="bottom-strip"><div class="strip-total">Total: {total}</div><div class="strip-bars">'
    if not counts:
        html += '<span style="color:#94a3b8; font-size:0.85rem;">No stats yet.</span>'
    else:
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for cls, count in sorted_counts:
            color = '#3b82f6'
            color_tuple = get_color_for_category(cls)
            if color_tuple:
                color = f"#{color_tuple[2]:02x}{color_tuple[1]:02x}{color_tuple[0]:02x}"
            html += f'<div class="mini-bar"><span class="color-dot" style="background-color:{color}"></span>{cls}: {count}</div>'
    html += '</div></div>'
    return html

def image_to_base64(img_np):
    success, encoded_image = cv2.imencode('.png', img_np)
    if success:
        return base64.b64encode(encoded_image).decode()
    return ""

try:
    import spaces
except ImportError:
    spaces = None

def process_frame(frame_rgb, conf_threshold, iou_threshold, show_boxes, show_labels, voice_enabled, tracker_state, recent_detections, class_counts, total_dets):
    if frame_rgb is None:
        return None, tracker_state, recent_detections, class_counts, total_dets, generate_html_recent(recent_detections), generate_html_strip(class_counts, total_dets)
        
    try:
        # Fix mirrored/flipped image via OpenCV (since Gradio 5+ removed mirror_webcam param)
        frame_rgb = cv2.flip(frame_rgb, 1)
        
        # Inference & Classification
        results, elapsed = run_inference(yolo_model, frame_rgb, conf_threshold, iou_threshold, use_tracker=True)
        results = enhance_results_with_cnn(results, frame_rgb, cnn_model, FINAL_LABELS)
        
        detections = parse_results(results)
        
        for d in detections:
            track_id = d.get("track_id", -1)
            if tracker_state.check_and_add(track_id):
                cls = d["class_name"]
                
                # Voice Alerts
                if voice_enabled and _voice.is_available():
                    _voice.speak(cls)
                    
                class_counts[cls] = class_counts.get(cls, 0) + 1
                total_dets += 1
                
                x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
                h, w = frame_rgb.shape[:2]
                x1, y1 = max(0, x1-5), max(0, y1-5)
                x2, y2 = min(w, x2+5), min(h, y2+5)
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    b64 = image_to_base64(crop_bgr)
                    recent_detections.insert(0, {
                        "class": cls,
                        "conf": round(d["confidence"], 1),
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "img_b64": b64,
                        "track_id": track_id
                    })
                    if len(recent_detections) > 15:
                        recent_detections.pop()
        
        annotated_rgb = draw_detections(frame_rgb, results, show_boxes=show_boxes, show_labels=show_labels, tracker_state=tracker_state)
        return annotated_rgb, tracker_state, recent_detections, class_counts, total_dets, generate_html_recent(recent_detections), generate_html_strip(class_counts, total_dets)
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame_rgb, tracker_state, recent_detections, class_counts, total_dets, generate_html_recent(recent_detections), generate_html_strip(class_counts, total_dets)

if spaces is not None:
    process_frame = spaces.GPU(process_frame)

# Monkey-patch navigator.mediaDevices.getUserMedia to forcefully request the back camera
head_js = """
<script>
    const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async function(constraints) {
        if (constraints && constraints.video) {
            if (typeof constraints.video === 'boolean') {
                constraints.video = { facingMode: "environment" };
            } else if (typeof constraints.video === 'object') {
                constraints.video.facingMode = "environment";
            }
        }
        return originalGetUserMedia(constraints);
    };
</script>
"""

with gr.Blocks(head=head_js) as demo:
    # State Variables
    tracker_state = gr.State(TrackerState())
    recent_detections = gr.State([])
    class_counts = gr.State({})
    total_dets = gr.State(0)
    
    # Header
    header_html = gr.HTML('<div class="top-bar"><div class="top-bar-left">🚦 TrafficSight Dashboard</div><div class="top-bar-right"><span class="pill pill-green">● WebRTC Camera Online</span></div></div>')
    
    with gr.Row():
        with gr.Column(scale=7):
            with gr.Accordion("📷 Click to Start Camera (Hidden Feed)", open=False):
                webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Live Browser Camera")
            
            video_feed = gr.Image(label="Processed Detections", interactive=False)
            
            with gr.Accordion("Controls", open=False):
                with gr.Row():
                    conf_threshold = gr.Slider(label="Confidence", minimum=0.1, maximum=1.0, value=0.25, step=0.05)
                    iou_threshold = gr.Slider(label="IoU", minimum=0.1, maximum=1.0, value=0.45, step=0.05)
                with gr.Row():
                    show_boxes = gr.Checkbox(label="Show Boxes", value=True)
                    show_labels = gr.Checkbox(label="Show Labels", value=True)
                    voice_enabled = gr.Checkbox(label="🔊 Voice Alerts", value=True)
                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset Session", variant="secondary")
                    
        with gr.Column(scale=3):
            gr.Markdown("**Recent Detections**")
            recent_html = gr.HTML(generate_html_recent([]))
            
    strip_html = gr.HTML(generate_html_strip({}, 0))
    
    def reset_state(tracker):
        tracker.reset()
        return tracker, [], {}, 0, generate_html_recent([]), generate_html_strip({}, 0)

    # Stream event handler
    webcam_input.stream(
        fn=process_frame,
        inputs=[webcam_input, conf_threshold, iou_threshold, show_boxes, show_labels, voice_enabled, tracker_state, recent_detections, class_counts, total_dets],
        outputs=[video_feed, tracker_state, recent_detections, class_counts, total_dets, recent_html, strip_html]
    )
    
    reset_btn.click(
        reset_state,
        inputs=[tracker_state],
        outputs=[tracker_state, recent_detections, class_counts, total_dets, recent_html, strip_html]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=css)
