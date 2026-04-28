"""
============================================================
Image Analysis Route -- POST /api/analyze
============================================================
Accepts an uploaded image, runs the full AI pipeline
(face -> uniform -> ID card -> scoring), and returns results.
============================================================
"""

import traceback
import numpy as np
from fastapi import APIRouter, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse

from app.utils.image_utils import bytes_to_cv2, encode_image_base64
from app.utils.drawing import annotate_full_results
from app.pipeline.scoring import compute_attire_score
from app.database import log_violation, get_student, log_analysis

router = APIRouter()


def _sanitize(obj):
    """
    Recursively convert numpy types to native Python types.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@router.post("/analyze")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    manual_student_id: str = Form(default=None),
):
    """
    Analyze an uploaded image for uniform compliance.
    """
    contents = await file.read()
    image = bytes_to_cv2(contents)

    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file"}
        )

    face_pipe = request.app.state.face_pipeline
    uniform_pipe = request.app.state.uniform_pipeline
    id_card_pipe = request.app.state.id_card_pipeline

    # ── Step 1: Face Recognition ────────────────────────
    face_results = {"faces": []}
    try:
        face_results = face_pipe.detect_and_identify(image)
    except Exception as e:
        print(f"    [WARN] Face recognition error: {e}")
        traceback.print_exc()

    student_id = None
    student_name = "Unknown"
    face_confidence = 0.0

    if face_results["faces"]:
        top_face = face_results["faces"][0]
        student_id = top_face["student_id"]
        student_name = top_face["name"]
        face_confidence = top_face["confidence"]

    if manual_student_id:
        student_id = manual_student_id
        # Learning from Correction: Update student embedding in DB
        if face_results["faces"]:
            try:
                emb = face_results["faces"][0].get("embedding")
                if emb:
                    update_student(manual_student_id, embedding=emb)
                    face_pipe.reload() # Immediate recognition for next frame
                    print(f"    [AI Learning] Enrolled embedding for {manual_student_id}")
            except Exception as e:
                print(f"    [WARN] Failed to auto-enroll: {e}")

        student_info = get_student(manual_student_id)
        if student_info:
            student_name = student_info["name"]
        face_confidence = 1.0
        if face_results["faces"]:
            face_results["faces"][0]["student_id"] = student_id
            face_results["faces"][0]["name"] = student_name
            face_results["faces"][0]["confidence"] = 1.0
            face_results["faces"][0]["method"] = "Administrative Entry"

    # ── Step 2: Uniform Detection ───────────────────────
    uniform_results = {"persons": []}
    try:
        uniform_results = uniform_pipe.detect(image, face_results)
    except Exception as e:
        print(f"    [WARN] Uniform detection error: {e}")
        traceback.print_exc()

    shirt_det = {"detected": False, "confidence": 0.0, "method": "None"}
    pant_det = {"detected": False, "confidence": 0.0, "method": "None"}
    tucked_det = {"detected": False, "confidence": 0.0, "method": "None"}
    person_bbox = None

    if uniform_results["persons"]:
        person = uniform_results["persons"][0]
        shirt_det = person.get("shirt", shirt_det)
        pant_det = person.get("pant", pant_det)
        tucked_det = person.get("tucked", tucked_det)
        person_bbox = person.get("bbox")

    # ── Step 3: ID Card Detection ───────────────────────
    id_card_result = {"detected": False, "confidence": 0.0, "method": "None", "bbox": None}
    try:
        id_card_result = id_card_pipe.detect(image, person_bbox)
    except Exception as e:
        print(f"    [WARN] ID card detection error: {e}")
        traceback.print_exc()

    # ── Step 4: Compute Score ───────────────────────────
    score = compute_attire_score({
        "shirt": shirt_det,
        "pant": pant_det,
        "tucked": tucked_det,
        "id_card": id_card_result,
    })

    # ── Step 5: Consolidate Process Logs ────────────────
    analysis_log = face_results.get("analysis_log", [])
    analysis_log.append({
        "step": "Attire Analysis",
        "algorithm": "YOLOv8 Neural Network",
        "reason": f"Scanning lower body and torso to identify Shirt ({shirt_det['confidence']:.0%}), Pant ({pant_det['confidence']:.0%}), and Tuck-in status."
    })
    analysis_log.append({
        "step": "Identity Verification",
        "algorithm": "White-Card Heuristics",
        "reason": "Searching upper torso for high-contrast rectangular markers to verify presence of Institutional ID Card."
    })

    # ── Step 6: Annotate Image ──────────────────────────
    analysis = {
        "face_results": face_results,
        "uniform_results": uniform_results,
        "id_card_result": id_card_result,
        "score": score,
    }
    annotated = annotate_full_results(image, analysis)
    annotated_b64 = encode_image_base64(annotated)

    response = {
        "student": {
            "student_id": student_id or "UNKNOWN",
            "name": student_name,
            "confidence": round(float(face_confidence), 3),
            "method": face_results["faces"][0]["method"] if face_results["faces"] else "None",
        },
        "score": score,
        "detections": {
            "face": face_results,
            "uniform": uniform_results,
            "id_card": id_card_result,
        },
        "analysis_log": analysis_log,
        "violations_logged": [],
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
    }
    return _sanitize(response)


from pydantic import BaseModel
from typing import List

class SaveAnalysisRequest(BaseModel):
    student_id: str
    total_score: float
    violations: List[str]
    face_confidence: float

class EnrollRequest(BaseModel):
    student_id: str
    embedding: List[float]

@router.post("/enroll-unknown")
async def enroll_unknown(request: Request, req: EnrollRequest):
    """Enroll an unknown face with a manual ID (Incremental Learning)."""
    try:
        update_student(req.student_id, embedding=req.embedding)
        request.app.state.face_pipeline.reload()
        return {"status": "success", "message": f"Successfully enrolled {req.student_id}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/save-analysis")
async def save_analysis(req: SaveAnalysisRequest):
    """Manually save analysis results."""
    logged_violations = []
    for violation_type in req.violations:
        v = log_violation(
            student_id=req.student_id,
            violation_type=violation_type,
            confidence=req.face_confidence,
            details={"score": req.total_score, "method": "manual-save"}
        )
        logged_violations.append(v)
    log_analysis(student_id=req.student_id, total_score=req.total_score, issues_found=req.violations)
    return {"status": "success", "saved_violations": len(logged_violations)}
