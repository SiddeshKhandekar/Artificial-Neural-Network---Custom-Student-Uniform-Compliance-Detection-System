"""
============================================================
Professional Attire Scoring Engine
============================================================
Weighted scoring: Shirt(30%) + Pant(30%) + Tucked(20%) + ID(20%)
Confidence penalty applied when model confidence is low.
============================================================
"""

from app.config import SCORING_WEIGHTS, CONFIDENCE_PENALTY_THRESHOLD, SCORE_LABELS


def compute_attire_score(detections: dict) -> dict:
    """
    Compute the professional attire compliance score.
    
    Args:
        detections: {
            "shirt": {"detected": bool, "confidence": float},
            "pant": {"detected": bool, "confidence": float},
            "tucked": {"detected": bool, "confidence": float},
            "id_card": {"detected": bool, "confidence": float},
        }
    
    Returns:
        {
            "total_score": 85.5,
            "label": "Good",
            "components": {
                "shirt": {"score": 28.5, "max": 30, ...},
                ...
            },
            "violations": ["Missing ID Card"]
        }
    """
    components = {}
    violations = []
    total = 0.0

    for component, weight in SCORING_WEIGHTS.items():
        det = detections.get(component, {"detected": False, "confidence": 0.0})
        detected = det.get("detected", False)
        confidence = det.get("confidence", 0.0)
        method = det.get("method", "None")
        max_score = weight * 100

        if detected:
            base_score = max_score
            # Apply confidence penalty if model isn't sure
            if confidence < CONFIDENCE_PENALTY_THRESHOLD:
                penalty = confidence / CONFIDENCE_PENALTY_THRESHOLD
                base_score *= penalty
            score = round(base_score, 1)
        else:
            score = 0.0
            # Log violation
            violation_names = {
                "shirt": "Missing College Shirt",
                "pant": "Missing College Pant",
                "tucked": "Shirt Not Tucked In",
                "id_card": "Missing ID Card",
            }
            violations.append(violation_names.get(component, f"Missing {component}"))

        components[component] = {
            "score": score,
            "max_score": max_score,
            "detected": detected,
            "confidence": round(confidence, 3),
            "method": method,
        }
        total += score

    total = round(total, 1)

    # Determine label
    label = "Non-Compliant"
    for threshold in sorted(SCORE_LABELS.keys(), reverse=True):
        if total >= threshold:
            label = SCORE_LABELS[threshold]
            break

    return {
        "total_score": total,
        "label": label,
        "components": components,
        "violations": violations,
    }
