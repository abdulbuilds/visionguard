"""
Detection utilities for Traffic Sign Detection App.
Handles model loading, inference, image processing, and result formatting.
"""

import cv2
import numpy as np
import time
import json
import pandas as pd
from PIL import Image
import io
import base64


def load_model(model_path: str):
    """
    Load YOLOv8 model from the given path.
    Returns the model object or raises an exception.
    """
    from ultralytics import YOLO
    model = YOLO(model_path)
    return model


def run_inference(model, image: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45, use_tracker: bool = False):
    """
    Run YOLOv8 inference on a numpy image (BGR or RGB).
    When use_tracker=True, uses ByteTrack for persistent object tracking.
    Returns results object and elapsed time in seconds.
    """
    start = time.time()
    if use_tracker:
        results = model.track(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )
    else:
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
    elapsed = time.time() - start
    return results, elapsed


def enhance_results_with_cnn(results, image: np.ndarray, cnn_model, labels_dict: dict):
    """
    Mutate YOLO results object by cropping bounding boxes of specific classes,
    passing them to a CNN, and updating the class ID and name.
    """
    if cnn_model is None or labels_dict is None:
        return results

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return results

    # Only speed limit signs go to CNN for detailed classification
    cnn_target_classes = ["speedlimit"]
    
    # Direct YOLO-to-display-name mapping (these skip CNN entirely)
    direct_rename = {
        "crosswalk": "Zebra crossing",
        "stop": "Stop",
        "trafficlight": "Traffic signals",
    }
    
    # We will need the next available ID for dynamically adding classes
    current_max_id = max(list(results[0].names.keys())) if results[0].names else -1
    
    # Create a clone of the boxes data so we can modify it safely
    new_data = boxes.data.clone()
    new_names = dict(results[0].names)
    modified = False
    
    # Determine the column index for 'cls'. It is 6 if track_id is present (shape 7), else 5.
    cls_idx = 6 if new_data.shape[1] == 7 else 5
    
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        cls_name = results[0].names.get(cls_id, "").lower()
        cls_name_clean = cls_name.replace(" ", "")
        
        # Check if this is a direct-rename class (crosswalk, stop, trafficlight)
        renamed = False
        for key, display_name in direct_rename.items():
            if key in cls_name_clean:
                new_id = current_max_id + 1
                current_max_id += 1
                new_names[new_id] = display_name
                new_data[i, cls_idx] = float(new_id)
                modified = True
                renamed = True
                break
        
        if renamed:
            continue
        
        # Only speed limit signs go to CNN for detailed classification
        is_speed = any(target in cls_name_clean for target in cnn_target_classes)
        
        if is_speed:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Bound coordinates to image shape
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            # Resize for CNN
            crop_resized = cv2.resize(crop, (64, 64))
            
            # Normalize and format input
            crop_normalized = crop_resized.astype(np.float32) / 255.0
            crop_input = np.expand_dims(crop_normalized, axis=0)
            
            # Predict using model call instead of predict() for better thread safety and performance in Streamlit
            predictions = cnn_model(crop_input, training=False).numpy()
            cnn_cls_id = int(np.argmax(predictions[0]))
            
            # Print predicted details as requested
            pred = np.argmax(predictions[0])
            print("Predicted Index:", pred)
            if 'train_dataset' in globals() and hasattr(train_dataset, 'class_names'):
                print("Predicted Folder:", train_dataset.class_names[pred])
            elif labels_dict is not None and pred in labels_dict:
                print("Predicted Folder:", labels_dict[pred])
            
            if cnn_cls_id in labels_dict:
                detailed_name = labels_dict[cnn_cls_id]
                
                # Update YOLO results object in place
                new_id = current_max_id + 1
                current_max_id += 1
                
                new_names[new_id] = detailed_name
                new_data[i, cls_idx] = float(new_id)
                modified = True

    if modified:
        from ultralytics.engine.results import Results
        new_res = Results(orig_img=results[0].orig_img, path=results[0].path, names=new_names, boxes=new_data)
        results[0] = new_res

    return results


def get_color_for_category(class_name: str) -> tuple:
    """Return BGR color tuple based on sign category."""
    name_lower = class_name.lower()
    # Regulatory (Red)
    if any(x in name_lower for x in ["stop", "speed", "yield", "crossing", "end"]):
        return (44, 68, 239) # #ef4444 in BGR
    # Warning (Amber)
    elif any(x in name_lower for x in ["signal", "light"]):
        return (11, 158, 245) # #f59e0b in BGR
    # Informational (Blue)
    else:
        return (246, 130, 59) # #3b82f6 in BGR

def draw_detections(image: np.ndarray, results, show_boxes: bool = True, show_labels: bool = True, tracker_state=None) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of the image.
    Returns annotated image as numpy array (RGB).
    """
    annotated = image.copy()
    
    if not show_boxes and not show_labels:
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        track_id = int(box.id[0].item()) if box.id is not None else -1
        
        cls_name = results[0].names.get(cls_id, f"Class {cls_id}")
        color = get_color_for_category(cls_name)
        if tracker_state:
            color = tracker_state.get_pulse_color(track_id, color)
        
        if show_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
        if show_labels:
            label = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb


def parse_results(results, class_names: dict = None) -> list[dict]:
    """
    Parse YOLO results into a list of detection dictionaries.
    Each dict: {class_id, class_name, confidence, x1, y1, x2, y2, width, height, track_id}
    track_id is -1 when ByteTrack hasn't assigned an ID yet.
    """
    detections = []
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Extract ByteTrack track ID (None if tracker not active or not yet assigned)
        track_id = -1
        if box.id is not None:
            track_id = int(box.id[0].item())

        # Resolve class name
        if class_names and cls_id in class_names:
            cls_name = class_names[cls_id]
        elif results[0].names and cls_id in results[0].names:
            cls_name = results[0].names[cls_id]
        else:
            cls_name = f"Class {cls_id}"

        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": round(conf * 100, 2),
            "x1": round(x1, 1),
            "y1": round(y1, 1),
            "x2": round(x2, 1),
            "y2": round(y2, 1),
            "width": round(x2 - x1, 1),
            "height": round(y2 - y1, 1),
            "track_id": track_id,
        })

    return detections


def detections_to_dataframe(detections: list[dict]) -> pd.DataFrame:
    """Convert list of detection dicts to a styled pandas DataFrame."""
    if not detections:
        return pd.DataFrame(columns=["Class", "Confidence (%)", "X1", "Y1", "X2", "Y2", "W", "H"])

    rows = []
    for d in detections:
        rows.append({
            "Class": d["class_name"],
            "Confidence (%)": d["confidence"],
            "X1": d["x1"],
            "Y1": d["y1"],
            "X2": d["x2"],
            "Y2": d["y2"],
            "W (px)": d["width"],
            "H (px)": d["height"],
        })
    return pd.DataFrame(rows)


def detections_to_csv(detections: list[dict]) -> bytes:
    """Serialize detections to CSV bytes."""
    df = detections_to_dataframe(detections)
    return df.to_csv(index=False).encode("utf-8")


def detections_to_json(detections: list[dict], metadata: dict = None) -> bytes:
    """Serialize detections to JSON bytes with optional metadata."""
    payload = {
        "metadata": metadata or {},
        "detections": detections,
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def pil_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert a PIL image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a numpy RGB array to PIL Image."""
    return Image.fromarray(arr.astype(np.uint8))


def get_image_info(uploaded_file) -> dict:
    """Extract metadata from a Streamlit UploadedFile object."""
    img = Image.open(uploaded_file)
    size_kb = len(uploaded_file.getvalue()) / 1024
    return {
        "filename": uploaded_file.name,
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "size_kb": round(size_kb, 2),
        "size_mb": round(size_kb / 1024, 3),
    }


def compute_metrics(detections: list[dict]) -> dict:
    """Compute summary metrics from detections."""
    if not detections:
        return {
            "total": 0,
            "highest_conf": 0.0,
            "avg_conf": 0.0,
            "unique_classes": 0,
        }
    confs = [d["confidence"] for d in detections]
    classes = set(d["class_name"] for d in detections)
    return {
        "total": len(detections),
        "highest_conf": round(max(confs), 2),
        "avg_conf": round(sum(confs) / len(confs), 2),
        "unique_classes": len(classes),
    }
