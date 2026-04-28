"""
============================================================
Configuration Module
============================================================
Centralizes all application settings: paths, model parameters,
uniform color ranges, scoring weights, and device selection.
============================================================
"""

import os
import json
import torch
from pathlib import Path

# ── Base Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
FACES_DIR = DATA_DIR / "student_faces"
UNIFORM_SAMPLES_DIR = DATA_DIR / "uniform_samples"
DB_PATH = BASE_DIR / "database.db"
CALIBRATION_FILE = DATA_DIR / "uniform_colors.json"

# Create directories if they don't exist
for d in [DATA_DIR, MODELS_DIR, FACES_DIR, UNIFORM_SAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Device Configuration (GPU / CPU) ───────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Face Recognition Settings ──────────────────────────────
FACE_EMBEDDING_DIM = 512
FACE_CONFIDENCE_THRESHOLD = 0.70
FACE_FALLBACK_THRESHOLD = 0.50
NUM_STUDENTS = 5

# ── Uniform Detection Settings ─────────────────────────────
# CALIBRATION MODE: If a calibration file exists, load colors from it.
# Otherwise, use these defaults. The calibration file is created via
# POST /api/calibrate endpoint on the frontend.
_DEFAULT_SHIRT_HSV_LOW = (120, 20, 100)
_DEFAULT_SHIRT_HSV_HIGH = (165, 180, 255)
_DEFAULT_PANT_HSV_LOW = (0, 0, 0)
_DEFAULT_PANT_HSV_HIGH = (179, 100, 70)


def _load_calibrated_colors():
    """Load calibrated uniform colors from JSON file if it exists."""
    if CALIBRATION_FILE.exists():
        try:
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            return {
                "shirt_low": tuple(data.get("shirt_hsv_low", list(_DEFAULT_SHIRT_HSV_LOW))),
                "shirt_high": tuple(data.get("shirt_hsv_high", list(_DEFAULT_SHIRT_HSV_HIGH))),
                "pant_low": tuple(data.get("pant_hsv_low", list(_DEFAULT_PANT_HSV_LOW))),
                "pant_high": tuple(data.get("pant_hsv_high", list(_DEFAULT_PANT_HSV_HIGH))),
            }
        except Exception:
            pass
    return None


def save_calibrated_colors(shirt_low, shirt_high, pant_low, pant_high):
    """Save calibrated HSV color ranges to disk."""
    data = {
        "shirt_hsv_low": list(shirt_low),
        "shirt_hsv_high": list(shirt_high),
        "pant_hsv_low": list(pant_low),
        "pant_hsv_high": list(pant_high),
    }
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    # Update live config
    global UNIFORM_SHIRT_HSV_LOW, UNIFORM_SHIRT_HSV_HIGH
    global UNIFORM_PANT_HSV_LOW, UNIFORM_PANT_HSV_HIGH
    UNIFORM_SHIRT_HSV_LOW = tuple(shirt_low)
    UNIFORM_SHIRT_HSV_HIGH = tuple(shirt_high)
    UNIFORM_PANT_HSV_LOW = tuple(pant_low)
    UNIFORM_PANT_HSV_HIGH = tuple(pant_high)


# Load calibrated colors or use defaults
_cal = _load_calibrated_colors()
if _cal:
    UNIFORM_SHIRT_HSV_LOW = _cal["shirt_low"]
    UNIFORM_SHIRT_HSV_HIGH = _cal["shirt_high"]
    UNIFORM_PANT_HSV_LOW = _cal["pant_low"]
    UNIFORM_PANT_HSV_HIGH = _cal["pant_high"]
else:
    UNIFORM_SHIRT_HSV_LOW = _DEFAULT_SHIRT_HSV_LOW
    UNIFORM_SHIRT_HSV_HIGH = _DEFAULT_SHIRT_HSV_HIGH
    UNIFORM_PANT_HSV_LOW = _DEFAULT_PANT_HSV_LOW
    UNIFORM_PANT_HSV_HIGH = _DEFAULT_PANT_HSV_HIGH

UNIFORM_CNN_CONFIDENCE_THRESHOLD = 0.60
TUCKIN_OVERLAP_THRESHOLD_PX = 15
UNIFORM_IMAGE_SIZE = (224, 224)

# ── Tesseract OCR Configuration ────────────────────────────
# Tesseract must be installed separately on Windows.
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Default install path on Windows:
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
# Configure pytesseract to use the correct path
try:
    import pytesseract
    if os.path.exists(TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except ImportError:
    pass  # pytesseract will be imported later

# ── ID Card Detection Settings ─────────────────────────────
ID_CARD_CONFIDENCE_THRESHOLD = 0.50
ID_CARD_OCR_KEYWORDS = [
    "id", "student", "college", "university", "identity", "card"
]

# ── Scoring Weights ────────────────────────────────────────
SCORING_WEIGHTS = {
    "shirt":   0.30,
    "pant":    0.30,
    "tucked":  0.20,
    "id_card": 0.20,
}

CONFIDENCE_PENALTY_THRESHOLD = 0.80

SCORE_LABELS = {
    95: "Best Professional Attire",
    80: "Good",
    60: "Needs Improvement",
    0:  "Non-Compliant",
}

# ── API Settings ────────────────────────────────────────────
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

STREAM_TARGET_FPS = 5

# ── YOLOv8 Settings ────────────────────────────────────────
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONFIDENCE = 0.25
YOLO_PERSON_CLASS_ID = 0
