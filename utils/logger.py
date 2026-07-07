"""
=============================================================================
  logger.py  —  Persistent CSV detection logger
  
  Every detection event is appended to outputs/detection_log.csv with
  columns: Timestamp, Class Name, Confidence (%), Source
=============================================================================
"""

import os
import csv
import threading
from datetime import datetime

# Output directory (outputs/ beside this package)
_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),   # project root
    "outputs"
)
_CSV_COLUMNS = ["Timestamp", "Class Name", "Confidence (%)", "Source"]

# Thread lock for concurrent appends
_lock = threading.Lock()


def _ensure_dir(mode: str = "images") -> str:
    """Create outputs/<mode>/ directory if it does not exist."""
    mode_dir = os.path.join(_OUTPUT_DIR, mode)
    os.makedirs(mode_dir, exist_ok=True)
    return mode_dir


def _ensure_header(mode: str = "images") -> str:
    """Write CSV header if the file is new / empty."""
    mode_dir = _ensure_dir(mode)
    log_file = os.path.join(mode_dir, "detection_log.csv")
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
    return log_file


def log_detections(detections: list[dict], source: str = "Upload", mode: str = "images") -> int:
    """
    Append *detections* to the CSV log.

    Parameters
    ----------
    detections : list[dict]
        Each dict must have keys: class_name, confidence
    source : str
        Label for the Source column (e.g. 'Upload', 'Webcam')
    mode : str
        Subdirectory mode ('images' or 'webcam')

    Returns
    -------
    int
        Number of rows written.
    """
    if not detections:
        return 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        {
            "Timestamp":       now,
            "Class Name":      d.get("class_name", "Unknown"),
            "Confidence (%)":  round(d.get("confidence", 0.0), 2),
            "Source":          source,
        }
        for d in detections
    ]

    with _lock:
        log_file = _ensure_header(mode)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writerows(rows)

    return len(rows)


def read_log(mode: str = "images") -> list[dict]:
    """
    Read all rows from the CSV log.

    Returns
    -------
    list[dict]
        Newest rows first.  Empty list if no log exists.
    """
    log_file = _ensure_header(mode)
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        return []

    with _lock:
        with open(log_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    return list(reversed(rows))   # newest first


def clear_log(mode: str = "images") -> None:
    """Delete the log file and reinitialise a fresh header."""
    _ensure_dir(mode)
    log_file = os.path.join(_OUTPUT_DIR, mode, "detection_log.csv")
    with _lock:
        if os.path.exists(log_file):
            os.remove(log_file)
        _ensure_header(mode)


def log_bytes(mode: str = "images") -> bytes:
    """Return the CSV log as bytes (for Streamlit download button)."""
    log_file = _ensure_header(mode)
    if not os.path.exists(log_file):
        return b""
    with open(log_file, "rb") as f:
        return f.read()


def get_output_dir(mode: str = "images") -> str:
    """Return the absolute path to the outputs/<mode>/ directory."""
    return _ensure_dir(mode)
