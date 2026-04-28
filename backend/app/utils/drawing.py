"""Bounding box and annotation drawing utilities."""

import cv2
import numpy as np


# Color palette (BGR format for OpenCV)
COLORS = {
    "face":     (255, 178, 50),    # Orange
    "shirt":    (50, 205, 50),     # Green
    "pant":     (50, 50, 205),     # Red-ish
    "tucked":   (205, 205, 50),    # Cyan
    "id_card":  (255, 50, 255),    # Magenta
    "person":   (200, 200, 200),   # Gray
    "unknown":  (0, 0, 255),       # Red
}


def draw_bbox(image: np.ndarray, bbox: list, label: str,
              confidence: float = None, color_key: str = "person") -> np.ndarray:
    """Draw a labeled bounding box on the image."""
    img = image.copy()
    x1, y1, x2, y2 = [int(c) for c in bbox]
    color = COLORS.get(color_key, (200, 200, 200))

    # Draw rectangle with slight transparency effect
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Build label text
    text = label
    if confidence is not None:
        text += f" {confidence:.0%}"

    # Draw label background
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def annotate_full_results(image: np.ndarray, analysis: dict) -> np.ndarray:
    """
    Draw all detections on the image.
    
    Args:
        image: BGR image
        analysis: Full pipeline result dict
    Returns:
        Annotated image copy
    """
    img = image.copy()

    # Draw face detections
    for face in analysis.get("face_results", {}).get("faces", []):
        label = face.get("name", "Unknown")
        if face.get("student_id") == "UNKNOWN":
            color_key = "unknown"
        else:
            color_key = "face"
        img = draw_bbox(img, face["bbox"], label,
                       face.get("confidence"), color_key)

    # Draw uniform detections
    for person in analysis.get("uniform_results", {}).get("persons", []):
        # Person bbox
        img = draw_bbox(img, person["bbox"], "Person", color_key="person")

        # Shirt region
        if person.get("upper_body_bbox"):
            shirt = person.get("shirt", {})
            status = "[OK] Shirt" if shirt.get("detected") else "[FAIL] No Shirt"
            img = draw_bbox(img, person["upper_body_bbox"], status,
                           shirt.get("confidence"), "shirt")

        # Pant region
        if person.get("lower_body_bbox"):
            pant = person.get("pant", {})
            status = "[OK] Pant" if pant.get("detected") else "[FAIL] No Pant"
            img = draw_bbox(img, person["lower_body_bbox"], status,
                           pant.get("confidence"), "pant")

        # Tucked region
        if person.get("tucked_bbox"):
            tucked = person.get("tucked", {})
            status = "Tucked" if tucked.get("detected") else "Untucked"
            img = draw_bbox(img, person["tucked_bbox"], status,
                           tucked.get("confidence"), "tucked")

    # Draw ID card detection
    id_result = analysis.get("id_card_result", {})
    if id_result.get("bbox"):
        status = "[OK] ID Card" if id_result.get("detected") else "[FAIL] No ID"
        img = draw_bbox(img, id_result["bbox"], status,
                       id_result.get("confidence"), "id_card")


    return img
