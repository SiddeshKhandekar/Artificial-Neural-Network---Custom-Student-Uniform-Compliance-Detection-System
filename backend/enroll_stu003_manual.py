import cv2
import os
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.pipeline.face_recognition import FaceRecognitionPipeline
from app.config import FACES_DIR

def enroll_stu003():
    print("Loading pipeline...")
    pipe = FaceRecognitionPipeline()
    pipe.initialize()
    
    student_id = "STU003"
    face_dir = FACES_DIR / student_id
    
    if not face_dir.exists():
        print(f"Directory {face_dir} does not exist!")
        return
        
    image_paths = list(face_dir.glob("*.jpeg")) + list(face_dir.glob("*.jpg"))
    if not image_paths:
        print(f"No images found in {face_dir}")
        return
        
    print(f"Found {len(image_paths)} images. Loading...")
    face_images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            face_images.append(img)
            
    print(f"Loaded {len(face_images)} valid images. Enrolling...")
    success = pipe.enroll_faces(student_id, face_images)
    
    if success:
        print("Successfully enrolled STU003!")
    else:
        print("Enrollment failed.")

if __name__ == "__main__":
    enroll_stu003()
