"""
============================================================
Students CRUD Routes -- /api/students
============================================================
"""

import os
import cv2
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from app.database import (
    create_student, get_student, get_all_students,
    update_student, delete_student
)
from app.config import FACES_DIR
from app.utils.image_utils import bytes_to_cv2

router = APIRouter()


@router.get("/students")
async def list_students():
    """List all registered students."""
    students = get_all_students()
    # Don't send raw embeddings to frontend (too large)
    for s in students:
        s["has_embedding"] = len(s.get("embedding", [])) > 0
        s.pop("embedding", None)
    return {"students": students, "total": len(students)}


@router.get("/students/{student_id}")
async def get_student_detail(student_id: str):
    """Get details of a specific student."""
    student = get_student(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    student["has_embedding"] = len(student.get("embedding", [])) > 0
    student.pop("embedding", None)
    return student


@router.post("/students")
async def create_new_student(
    student_id: str = Form(...),
    name: str = Form(...),
    department: str = Form(default=""),
    year: int = Form(default=3),
):
    """Register a new student."""
    existing = get_student(student_id)
    if existing:
        raise HTTPException(status_code=409, detail="Student ID already exists")

    # Create face directory
    face_dir = FACES_DIR / student_id
    face_dir.mkdir(parents=True, exist_ok=True)

    student = create_student(student_id, name, department, year)
    student.pop("embedding", None)
    return {"message": "Student created", "student": student}


@router.put("/students/{student_id}")
async def update_existing_student(
    student_id: str,
    name: Optional[str] = Form(default=None),
    department: Optional[str] = Form(default=None),
    year: Optional[int] = Form(default=None),
):
    """Update student information."""
    existing = get_student(student_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Student not found")

    kwargs = {}
    if name is not None:
        kwargs["name"] = name
    if department is not None:
        kwargs["department"] = department
    if year is not None:
        kwargs["year"] = year

    student = update_student(student_id, **kwargs)
    if student:
        student.pop("embedding", None)
    return {"message": "Student updated", "student": student}


@router.delete("/students/{student_id}")
async def remove_student(student_id: str):
    """Delete a student and their data."""
    if not get_student(student_id):
        raise HTTPException(status_code=404, detail="Student not found")

    # Remove face photos
    face_dir = FACES_DIR / student_id
    if face_dir.exists():
        import shutil
        shutil.rmtree(face_dir, ignore_errors=True)

    delete_student(student_id)
    return {"message": f"Student {student_id} deleted"}


@router.post("/students/{student_id}/enroll")
async def enroll_student_face(
    student_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
):
    """
    Enroll face photos for a student.
    Upload 3-5 clear face photos. The system will:
    1. Save photos to the student's face directory
    2. Extract FaceNet embeddings
    3. Train/retrain the MLP classifier
    """
    student = get_student(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    if len(files) < 1:
        raise HTTPException(status_code=400, detail="Upload at least 1 face photo")

    # Save face images
    face_dir = FACES_DIR / student_id
    face_dir.mkdir(parents=True, exist_ok=True)

    face_images = []
    for i, file in enumerate(files):
        contents = await file.read()
        image = bytes_to_cv2(contents)
        if image is not None:
            save_path = face_dir / f"face_{i+1}.jpg"
            cv2.imwrite(str(save_path), image)
            face_images.append(image)

    if not face_images:
        raise HTTPException(status_code=400, detail="No valid images found")

    # Enroll via face pipeline
    face_pipe = request.app.state.face_pipeline
    success = face_pipe.enroll_faces(student_id, face_images)

    if success:
        return {
            "message": f"Enrolled {len(face_images)} face photos for {student_id}",
            "photos_saved": len(face_images)
        }
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Face enrollment failed -- no faces detected in photos"}
        )
