"""
============================================================
WebSocket Camera Stream -- /ws/stream
============================================================
Receives webcam frames from the frontend via WebSocket,
processes each frame through the AI pipeline, and sends
back annotated frames + detection JSON.
============================================================
"""

import asyncio
import json
import time
import traceback
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import STREAM_TARGET_FPS
from app.utils.image_utils import decode_base64_image, encode_image_base64
from app.utils.drawing import annotate_full_results
from app.pipeline.scoring import compute_attire_score

router = APIRouter()

def _sanitize(obj):
    """
    Recursively convert numpy types to native Python types
    so json.dumps can serialize them.
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


@router.websocket("/ws/stream")
async def camera_stream(websocket: WebSocket):
    """
    Bidirectional WebSocket for live camera analysis.
    
    Client sends: base64-encoded JPEG frame
    Server responds: JSON with annotated frame + detections
    
    Throttled to ~STREAM_TARGET_FPS for performance.
    """
    await websocket.accept()
    print("  [WebSocket] Client connected")

    # Get pipelines from app state
    app = websocket.app
    face_pipe = app.state.face_pipeline
    uniform_pipe = app.state.uniform_pipeline
    id_card_pipe = app.state.id_card_pipeline

    frame_interval = 1.0 / STREAM_TARGET_FPS
    last_process_time = 0

    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()

            # Throttle processing
            now = time.time()
            if now - last_process_time < frame_interval:
                continue
            last_process_time = now

            try:
                # Decode image
                image = decode_base64_image(data)
                if image is None:
                    continue

                # Run pipeline in thread pool (CPU-bound work)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, _process_frame, image, face_pipe, uniform_pipe, id_card_pipe
                )

                # Send back results
                await websocket.send_text(json.dumps(result))

            except Exception as e:
                print(f"  [WebSocket] Frame processing error: {e}")
                traceback.print_exc()
                continue

    except WebSocketDisconnect:
        print("  [WebSocket] Client disconnected")
    except Exception as e:
        print(f"  [WebSocket] Connection error: {e}")


def _process_frame(image, face_pipe, uniform_pipe, id_card_pipe) -> dict:
    """Process a single frame through all pipelines. Runs in thread pool."""

    # Face recognition
    face_results = {"faces": []}
    try:
        face_results = face_pipe.detect_and_identify(image)
    except Exception as e:
        print(f"    [WARN] Stream face recognition error: {e}")

    # Uniform detection
    uniform_results = {"persons": []}
    try:
        uniform_results = uniform_pipe.detect(image)
    except Exception as e:
        print(f"    [WARN] Stream uniform detection error: {e}")

    # ID card detection
    person_bbox = None
    if uniform_results["persons"]:
        person_bbox = uniform_results["persons"][0].get("bbox")
        
    id_card_result = {"detected": False, "confidence": 0.0, "method": "None", "bbox": None}
    try:
        id_card_result = id_card_pipe.detect(image, person_bbox)
    except Exception as e:
        print(f"    [WARN] Stream ID card detection error: {e}")

    # Scoring
    shirt_det = {"detected": False, "confidence": 0.0, "method": "None"}
    pant_det = {"detected": False, "confidence": 0.0, "method": "None"}
    tucked_det = {"detected": False, "confidence": 0.0, "method": "None"}

    if uniform_results["persons"]:
        p = uniform_results["persons"][0]
        shirt_det = p.get("shirt", shirt_det)
        pant_det = p.get("pant", pant_det)
        tucked_det = p.get("tucked", tucked_det)

    score = compute_attire_score({
        "shirt": shirt_det,
        "pant": pant_det,
        "tucked": tucked_det,
        "id_card": id_card_result,
    })

    # Annotate image
    analysis = {
        "face_results": face_results,
        "uniform_results": uniform_results,
        "id_card_result": id_card_result,
        "score": score,
    }
    annotated = annotate_full_results(image, analysis)
    annotated_b64 = encode_image_base64(annotated)

    # Student info
    student_id = "UNKNOWN"
    student_name = "Unknown"
    face_confidence = 0.0
    if face_results["faces"]:
        f = face_results["faces"][0]
        student_id = f["student_id"]
        student_name = f["name"]
        face_confidence = f["confidence"]

    # Consolidate Logs
    analysis_log = face_results.get("analysis_log", [])
    analysis_log.append({
        "step": "Attire Scan",
        "algorithm": "YOLOv8 Real-time",
        "reason": f"Active neural scan for Shirt ({shirt_det['confidence']:.0%}) and Pant ({pant_det['confidence']:.0%}) consistency."
    })
    analysis_log.append({
        "step": "ID Verification",
        "algorithm": "Heuristic Alignment",
        "reason": "Tracking rectangular ID geometry relative to detected upper torso bounds."
    })

    response = {
        "student": {
            "student_id": student_id,
            "name": student_name,
            "confidence": round(float(face_confidence), 3),
        },
        "score": score,
        "detections": {
            "face_count": len(face_results["faces"]),
            "persons_count": len(uniform_results["persons"]),
            "id_card": id_card_result.get("detected", False),
        },
        "analysis_log": analysis_log,
        "annotated_frame": f"data:image/jpeg;base64,{annotated_b64}",
    }
    return _sanitize(response)
