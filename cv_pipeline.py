"""
cv_pipeline.py — YOLOv8 vehicle detection + time-of-day fallback.

Background thread runs get_junction_density() for every junction every 10 s.
Results are stored directly into junction_graph.junction_state via push_density().

Depends on: junction_graph.py
"""

import os
import random
import threading
import time

from junction_graph import JUNCTIONS, junction_state, push_density

# ---------------------------------------------------------------------------
# YOLOv8 setup
# ---------------------------------------------------------------------------

_YOLO_MODEL = None
_YOLO_LOADED = False
VEHICLE_CLASSES = {2, 3, 5, 7}   # COCO: car, motorcycle, bus, truck

_YOLOV8_PATH = os.path.join(os.path.dirname(__file__), "..", "mini project", "yolov8n.pt")
if not os.path.exists(_YOLOV8_PATH):
    _YOLOV8_PATH = "yolov8n.pt"   # fallback: same dir

def _load_yolo():
    global _YOLO_MODEL, _YOLO_LOADED
    try:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO(_YOLOV8_PATH)
        _YOLO_LOADED = True
        print(f"[CV] YOLOv8 loaded from {_YOLOV8_PATH}")
    except Exception as e:
        print(f"[CV] WARNING: YOLOv8 not available ({e}). Using synthetic fallback only.")

_load_yolo()

# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------

_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "junction_images")

# Also check the old project's traffic_images directory
_OLD_IMAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mini project", "traffic_images")
)

def _time_bucket(hour: int) -> str:
    if 6 <= hour < 9:   return "peak_morning"
    if 9 <= hour < 12:  return "high"
    if 12 <= hour < 17: return "medium"
    if 17 <= hour < 20: return "peak_evening"
    return "low"


def _find_image(junction_id: str, hour: int) -> str | None:
    """Return path to a junction image if one exists, else None."""
    bucket = _time_bucket(hour)

    # Look in new project images first, then old project
    for base_dir in [_IMAGE_DIR, os.path.join(_OLD_IMAGE_DIR, f"junction_{junction_id}")]:
        if not os.path.isdir(base_dir):
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            path = os.path.join(base_dir, f"{junction_id}_{bucket}{ext}")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path
            # Try old naming scheme: just bucket name
            path2 = os.path.join(base_dir, f"{bucket}{ext}")
            if os.path.exists(path2) and os.path.getsize(path2) > 0:
                return path2
    return None


# ---------------------------------------------------------------------------
# Synthetic density (fallback)
# ---------------------------------------------------------------------------

def _synthetic_density(junction_id: str, hour: int) -> float:
    """Time-of-day synthetic density with per-junction variation + noise."""
    cap = JUNCTIONS[junction_id]["lane_capacity"]
    base_cap_factor = cap / 30.0   # normalised against max capacity junction

    if 7 <= hour < 9 or 17 <= hour < 19:    # peaks
        base = random.uniform(0.65, 0.92)
    elif 12 <= hour < 14:                    # midday
        base = random.uniform(0.30, 0.55)
    elif 22 <= hour or hour < 6:             # night
        base = random.uniform(0.03, 0.18)
    else:
        base = random.uniform(0.28, 0.58)

    noise = random.gauss(0, 0.04)
    return min(max(base * base_cap_factor + noise, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Core density function
# ---------------------------------------------------------------------------

def get_junction_density(junction_id: str, hour: int | None = None) -> float:
    """
    PRIMARY: Run YOLOv8 on junction image → count vehicles → normalise.
    FALLBACK: Use time-of-day synthetic density if image missing or YOLO fails.

    Always stores result in junction_state via push_density().
    Returns float in [0.0, 1.0].
    """
    if hour is None:
        hour = time.localtime().tm_hour

    cap = JUNCTIONS[junction_id]["lane_capacity"]
    density = None

    # --- Try YOLO on real image ---
    if _YOLO_LOADED:
        img_path = _find_image(junction_id, hour)
        if img_path:
            try:
                results = _YOLO_MODEL(img_path, verbose=False)
                count = sum(
                    1 for box in results[0].boxes
                    if int(box.cls.item()) in VEHICLE_CLASSES
                )
                density = min(count / cap, 1.0)
            except Exception as e:
                print(f"[CV] YOLOv8 inference error at {junction_id}: {e}")

    # --- Fallback ---
    if density is None:
        density = _synthetic_density(junction_id, hour)

    push_density(junction_id, density)
    return density


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------

def _cv_loop(stop_event: threading.Event, socketio=None) -> None:
    """
    Every 10 seconds: enumerate all 17 junctions and refresh density.
    If socketio is provided, emit 'density_update' to 'controller' room.
    """
    while not stop_event.is_set():
        hour = time.localtime().tm_hour
        density_snapshot: dict[str, float] = {}

        for jid in JUNCTIONS:
            d = get_junction_density(jid, hour)
            density_snapshot[jid] = round(d, 3)

        if socketio is not None:
            try:
                socketio.emit("density_update", density_snapshot, room="controller")
            except Exception:
                pass

        stop_event.wait(10)   # sleep 10 s, wakes early if stop_event set


_cv_stop_event: threading.Event | None = None
_cv_thread: threading.Thread | None = None


def start_cv_pipeline(socketio=None) -> threading.Event:
    """Start the background CV loop. Returns the stop_event for shutdown."""
    global _cv_stop_event, _cv_thread
    _cv_stop_event = threading.Event()
    _cv_thread = threading.Thread(
        target=_cv_loop,
        args=(_cv_stop_event, socketio),
        daemon=True,
        name="cv-pipeline",
    )
    _cv_thread.start()
    print("[CV] Pipeline started (10s tick)")
    return _cv_stop_event


def stop_cv_pipeline() -> None:
    if _cv_stop_event:
        _cv_stop_event.set()
