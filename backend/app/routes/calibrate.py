"""
============================================================
Calibration Route -- POST /api/calibrate
============================================================
Allows the user to upload a reference uniform image and
sample colors to set the HSV ranges for uniform detection.
============================================================
"""

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

from app.config import (
    save_calibrated_colors, CALIBRATION_FILE,
    UNIFORM_SHIRT_HSV_LOW, UNIFORM_SHIRT_HSV_HIGH,
    UNIFORM_PANT_HSV_LOW, UNIFORM_PANT_HSV_HIGH,
)
from app.utils.image_utils import bytes_to_cv2, encode_image_base64

router = APIRouter()


@router.get("/calibration")
async def get_current_calibration():
    """Get current uniform color calibration values."""
    return {
        "calibrated": CALIBRATION_FILE.exists(),
        "shirt_hsv_low": list(UNIFORM_SHIRT_HSV_LOW),
        "shirt_hsv_high": list(UNIFORM_SHIRT_HSV_HIGH),
        "pant_hsv_low": list(UNIFORM_PANT_HSV_LOW),
        "pant_hsv_high": list(UNIFORM_PANT_HSV_HIGH),
    }


@router.post("/calibrate/auto")
async def auto_calibrate(
    file: UploadFile = File(...),
    component: str = Form(default="shirt"),  # "shirt" or "pant"
):
    """
    Auto-calibrate uniform colors from a reference image.

    Upload a photo showing the uniform component (shirt or pant).
    The system will:
    1. Convert to HSV
    2. Compute the dominant color cluster
    3. Set HSV range as dominant ± tolerance

    This creates a calibration file that persists across restarts.
    """
    contents = await file.read()
    image = bytes_to_cv2(contents)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute mean and std of HSV channels (center 60% of image to avoid edges)
    h, w = hsv.shape[:2]
    margin_h, margin_w = int(h * 0.2), int(w * 0.2)
    roi = hsv[margin_h:h-margin_h, margin_w:w-margin_w]

    mean_hsv = np.mean(roi.reshape(-1, 3), axis=0)
    std_hsv = np.std(roi.reshape(-1, 3), axis=0)

    # Set range: mean ± 1.5 * std (clamped to valid HSV ranges)
    tolerance = 1.5
    hsv_low = np.clip(mean_hsv - tolerance * std_hsv, [0, 0, 0], [179, 255, 255]).astype(int)
    hsv_high = np.clip(mean_hsv + tolerance * std_hsv, [0, 0, 0], [179, 255, 255]).astype(int)

    # Update the appropriate component
    import app.config as cfg
    if component == "shirt":
        save_calibrated_colors(
            hsv_low.tolist(), hsv_high.tolist(),
            list(cfg.UNIFORM_PANT_HSV_LOW), list(cfg.UNIFORM_PANT_HSV_HIGH)
        )
    else:
        save_calibrated_colors(
            list(cfg.UNIFORM_SHIRT_HSV_LOW), list(cfg.UNIFORM_SHIRT_HSV_HIGH),
            hsv_low.tolist(), hsv_high.tolist()
        )

    # Generate preview mask
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    preview = cv2.bitwise_and(image, image, mask=mask)
    preview_b64 = encode_image_base64(preview)

    return {
        "message": f"{component} colors calibrated successfully",
        "component": component,
        "hsv_low": hsv_low.tolist(),
        "hsv_high": hsv_high.tolist(),
        "mean_hsv": mean_hsv.tolist(),
        "preview_mask": f"data:image/jpeg;base64,{preview_b64}",
    }


@router.post("/calibrate/manual")
async def manual_calibrate(
    shirt_h_low: int = Form(default=100),
    shirt_s_low: int = Form(default=50),
    shirt_v_low: int = Form(default=50),
    shirt_h_high: int = Form(default=130),
    shirt_s_high: int = Form(default=255),
    shirt_v_high: int = Form(default=255),
    pant_h_low: int = Form(default=0),
    pant_s_low: int = Form(default=0),
    pant_v_low: int = Form(default=40),
    pant_h_high: int = Form(default=180),
    pant_s_high: int = Form(default=50),
    pant_v_high: int = Form(default=180),
):
    """Manually set HSV color ranges for uniform detection."""
    save_calibrated_colors(
        (shirt_h_low, shirt_s_low, shirt_v_low),
        (shirt_h_high, shirt_s_high, shirt_v_high),
        (pant_h_low, pant_s_low, pant_v_low),
        (pant_h_high, pant_s_high, pant_v_high),
    )
    return {
        "message": "Calibration saved",
        "shirt_hsv_low": [shirt_h_low, shirt_s_low, shirt_v_low],
        "shirt_hsv_high": [shirt_h_high, shirt_s_high, shirt_v_high],
        "pant_hsv_low": [pant_h_low, pant_s_low, pant_v_low],
        "pant_hsv_high": [pant_h_high, pant_s_high, pant_v_high],
    }
