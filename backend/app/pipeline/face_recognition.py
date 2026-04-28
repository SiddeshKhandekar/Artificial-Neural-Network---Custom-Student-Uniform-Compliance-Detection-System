"""
============================================================
Face Recognition Pipeline -- Primary + 2 Fallbacks
============================================================
Primary:   FaceNet embeddings -> Custom MLP classification
Fallback1: OpenCV Haar Cascade -> LBPH Face Recognizer
Fallback2: Returns UNKNOWN -> UI shows Manual Override
============================================================
"""

import cv2
import numpy as np
import torch
import json
from pathlib import Path

from app.config import (
    DEVICE, FACES_DIR, FACE_CONFIDENCE_THRESHOLD,
    FACE_FALLBACK_THRESHOLD, NUM_STUDENTS, MODELS_DIR,
    FACE_EMBEDDING_DIM
)
from app.models.mlp_classifier import FaceClassifierMLP, MLPTrainer
from app.database import get_all_students, update_student


class FaceRecognitionPipeline:
    """
    Multi-strategy face recognition with graceful degradation.
    """

    def __init__(self):
        self.facenet_model = None     # FaceNet for embeddings
        self.mtcnn = None             # MTCNN face detector
        self.mlp_model = None         # Custom MLP classifier
        self.student_map = {}         # index -> student_id mapping
        self.haar_cascade = None      # Fallback face detector
        self.lbph_recognizer = None   # Fallback face recognizer
        self.COSINE_SIMILARITY_THRESHOLD = 0.38
        self._initialized = False

    def reload(self):
        """Force reload student map from database."""
        self._initialized = False
        self.initialize()

    def initialize(self):
        """Load all models. Called once at startup."""
        if self._initialized:
            return

        print("  [Face Recognition] Initializing...")

        # ── Primary: FaceNet + MTCNN ────────────────────
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1

            # MTCNN: Multi-task Cascaded CNN for face detection
            self.mtcnn = MTCNN(
                image_size=160,
                margin=32,           # Increased margin for better feature context
                keep_all=True,       # Detect multiple faces
                device=DEVICE,
                post_process=True,   # Normalize pixel values
                min_face_size=40,    # Ignore tiny background faces
                thresholds=[0.6, 0.7, 0.7] # Optimized P/R/O-Net thresholds
            )

            # InceptionResnetV1: Pre-trained on VGGFace2 dataset
            # Outputs 512-dimensional face embedding vector
            # Similar faces produce similar (close) embeddings
            self.facenet_model = InceptionResnetV1(
                pretrained='vggface2'
            ).eval().to(DEVICE)

            print("    [OK] FaceNet + MTCNN loaded")
        except Exception as e:
            print(f"    [FAIL] FaceNet failed to load: {e}")

        # ── Load Custom MLP ─────────────────────────────
        self.mlp_model = MLPTrainer.load("face_mlp.pt")
        if self.mlp_model:
            print("    [OK] Custom MLP loaded")
        else:
            print("    [SKIP] No trained MLP found (run enrollment first)")

        # ── Build student mapping ───────────────────────
        self._build_student_map()

        # ── Fallback 1: Haar Cascade + LBPH ─────────────
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)

            self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=8, grid_y=8
            )
            lbph_path = MODELS_DIR / "lbph_model.yml"
            if lbph_path.exists():
                self.lbph_recognizer.read(str(lbph_path))
                print("    [OK] LBPH recognizer loaded")
            else:
                print("    [SKIP] No LBPH model found (will train on enrollment)")
        except Exception as e:
            print(f"    [FAIL] LBPH setup failed: {e}")

        self._initialized = True
        print("  [Face Recognition] Ready\n")

    def _build_student_map(self):
        """Map class indices to student IDs and cache their enrolled embeddings."""
        students = get_all_students()
        self.student_map = {}
        self.embedding_map = {} # class_idx -> torch.Tensor
        
        for i, s in enumerate(students):
            sid = s["student_id"]
            self.student_map[i] = sid
            if s["embedding"] and len(s["embedding"]) == FACE_EMBEDDING_DIM:
                self.embedding_map[i] = torch.tensor(s["embedding"], device=DEVICE)

    def _calculate_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute Cosine Similarity between two embeddings."""
        return torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()

    def detect_and_identify(self, image: np.ndarray) -> dict:
        """
        Main entry point: detect faces and identify students.
        
        Returns:
            {
                "faces": [...],
                "analysis_log": [...]
            }
        """
        self.initialize()
        results = {"faces": [], "analysis_log": []}

        # ── Primary Strategy ──────────────────────────
        results["analysis_log"].append({
            "step": "Primary Detection",
            "algorithm": "Neural Embedding Analysis (FaceNet)",
            "reason": "Capturing 512-dimensional facial embeddings using MTCNN alignment to find high-precision structural matches."
        })

        if self.facenet_model and self.mtcnn:
            try:
                primary_results = self._primary_detect(image)
                for face in primary_results:
                    if face["confidence"] >= FACE_CONFIDENCE_THRESHOLD:
                        results["analysis_log"].append({
                            "step": "Identification Success",
                            "algorithm": "Dual-Layer Neural Verification",
                            "reason": f"High-confidence match ({face['confidence']:.1%}) confirmed via Neural MLP Classification and Spatial Embedding Similarity."
                        })
                        results["faces"].append(face)
                    else:
                        # ── Fallback 1: LBPH ────────────
                        results["analysis_log"].append({
                            "step": "Fallback Strategy 1",
                            "algorithm": "Local Binary Patterns (LBPH)",
                            "reason": "Neural confidence below threshold. Invoking texture-based pattern matching (LBPH) to verify identity."
                        })
                        fallback = self._fallback_lbph(image, face["bbox"])
                        if fallback and fallback["confidence"] >= FACE_FALLBACK_THRESHOLD:
                            results["analysis_log"].append({
                                "step": "Identification Success",
                                "algorithm": "Texture Pattern (LBPH)",
                                "reason": f"Verification successful using pixel-neighborhood histogram analysis."
                            })
                            results["faces"].append(fallback)
                        else:
                            # ── Fallback 2: UNKNOWN ─────
                            results["analysis_log"].append({
                                "step": "Final Assessment",
                                "algorithm": "Exhaustive Pipeline Scan",
                                "reason": "No high-confidence matches found across deep learning or pattern matching models. Flagging for administrative review."
                            })
                            face["student_id"] = "UNKNOWN"
                            face["name"] = "Unknown"
                            face["method"] = "Exhaustive Pipeline Scan"
                            results["faces"].append(face)
                return results
            except Exception as e:
                results["analysis_log"].append({
                    "step": "System Error",
                    "algorithm": "None",
                    "reason": f"Critical failure in primary pipeline: {str(e)}"
                })

        # ── If primary completely fails, try Haar + LBPH ─
        if self.haar_cascade is not None:
            results["analysis_log"].append({
                "step": "Emergency Fallback",
                "algorithm": "Haar Cascades",
                "reason": "Primary AI models unavailable. Falling back to legacy OpenCV Haar Wavelet detection."
            })
            try:
                faces = self._haar_detect(image)
                for bbox in faces:
                    fallback = self._fallback_lbph(image, bbox)
                    if fallback:
                        results["faces"].append(fallback)
                    else:
                        results["faces"].append({
                            "bbox": bbox,
                            "student_id": "UNKNOWN",
                            "name": "Unknown",
                            "confidence": 0.0,
                            "method": "Exhaustive Pipeline Scan"
                        })
            except Exception as e:
                print(f"    Haar fallback error: {e}")

        return results

    def _primary_detect(self, image: np.ndarray) -> list:
        """FaceNet: detect faces, extract embeddings, classify with MLP + Spatial Verification."""
        from PIL import Image

        # Convert BGR (OpenCV) to RGB (FaceNet expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # MTCNN detects face bounding boxes and extracts aligned face crops
        boxes, probs = self.mtcnn.detect(pil_image)

        if boxes is None:
            return []

        # Get face crops as tensors (aligned and normalized)
        face_tensors = self.mtcnn(pil_image)

        if face_tensors is None:
            return []

        # Handle single face case
        if face_tensors.dim() == 3:
            face_tensors = face_tensors.unsqueeze(0)

        results = []
        for i, (box, face_tensor) in enumerate(zip(boxes, face_tensors)):
            bbox = [int(b) for b in box]

            # Extract 512-d embedding using FaceNet
            face_tensor = face_tensor.to(DEVICE)
            embedding = self.facenet_model(face_tensor.unsqueeze(0))
            embedding = embedding.squeeze(0)

            # Classify with custom MLP
            if self.mlp_model:
                class_idx, mlp_confidence = self.mlp_model.predict(embedding)
                student_id = self.student_map.get(class_idx, "UNKNOWN")
                
                # Spatial Verification (Cosine Similarity to Enrolled Template)
                # This prevents over-confident MLP predictions on random faces.
                similarity = 0.0
                enrolled_emb = self.embedding_map.get(class_idx)
                if enrolled_emb is not None:
                    similarity = self._calculate_similarity(embedding, enrolled_emb)

                # Final Confidence = Weighted score of neural and spatial data
                # We require a strict similarity threshold to prevent stranger misidentification.
                is_valid = similarity > self.COSINE_SIMILARITY_THRESHOLD 
                final_confidence = np.sqrt(mlp_confidence * max(0.1, similarity)) if is_valid else 0.0

                student = self._get_student_info(student_id) if is_valid else None
                results.append({
                    "bbox": bbox,
                    "student_id": student_id if is_valid else "UNKNOWN",
                    "name": student.get("name", "Unknown") if student else "Unknown",
                    "confidence": final_confidence,
                    "method": "Neural Embedding Analysis (FaceNet)" if is_valid else "Exhaustive Pipeline Scan",
                    "similarity_score": similarity,
                    "embedding": embedding.tolist() # Include raw embedding for enrollment
                })
            else:
                results.append({
                    "bbox": bbox,
                    "student_id": "UNKNOWN",
                    "name": "Unknown (MLP not trained)",
                    "confidence": 0.0,
                    "method": "Exhaustive Pipeline Scan"
                })

        return results

    def _haar_detect(self, image: np.ndarray) -> list:
        """Haar Cascade face detection (fallback detector)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        bboxes = []
        for (x, y, w, h) in faces:
            bboxes.append([x, y, x + w, y + h])
        return bboxes

    def _fallback_lbph(self, image: np.ndarray, bbox: list) -> dict | None:
        """LBPH Face Recognizer -- fallback classification."""
        if self.lbph_recognizer is None:
            return None

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            x1, y1, x2, y2 = bbox
            face_crop = gray[max(0,y1):y2, max(0,x1):x2]
            if face_crop.size == 0:
                return None

            face_crop = cv2.resize(face_crop, (100, 100))
            label, confidence_raw = self.lbph_recognizer.predict(face_crop)

            # LBPH confidence is distance (lower = better)
            # Convert to 0-1 scale (rough heuristic)
            confidence = max(0, 1 - confidence_raw / 200)
            student_id = self.student_map.get(label, "UNKNOWN")
            student = self._get_student_info(student_id)

            return {
                "bbox": bbox,
                "student_id": student_id,
                "name": student.get("name", "Unknown") if student else "Unknown",
                "confidence": confidence,
                "method": "Local Binary Patterns (LBPH)"
            }
        except Exception:
            return None

    def _get_student_info(self, student_id: str) -> dict | None:
        from app.database import get_student
        return get_student(student_id)

    def enroll_faces(self, student_id: str, face_images: list) -> bool:
        """
        Enroll a student: compute FaceNet embeddings, train MLP, train LBPH.
        
        Args:
            student_id: The student's ID string
            face_images: List of BGR numpy arrays (3-5 face photos)
        """
        if not self.facenet_model or not self.mtcnn:
            print("    Cannot enroll: FaceNet not loaded")
            return False

        from PIL import Image

        embeddings = []
        for img in face_images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            face_tensor = self.mtcnn(pil_img)
            if face_tensor is not None:
                if face_tensor.dim() == 4:
                    face_tensor = face_tensor[0]
                face_tensor = face_tensor.to(DEVICE)
                emb = self.facenet_model(face_tensor.unsqueeze(0))
                embeddings.append(emb.squeeze(0).detach().cpu().numpy().tolist())

        if not embeddings:
            return False

        # Store average embedding in database
        avg_emb = np.mean(embeddings, axis=0).tolist()
        update_student(student_id, embedding=avg_emb)

        # Retrain MLP with all enrolled students
        self._retrain_mlp()
        self._retrain_lbph()
        self._build_student_map()

        return True

    def _retrain_mlp(self):
        """Retrain MLP on all enrolled student embeddings."""
        students = get_all_students()
        all_embs, all_labels = [], []

        for i, student in enumerate(students):
            if student["embedding"]:
                emb = np.array(student["embedding"])
                if emb.ndim == 1 and len(emb) == 512:
                    # Augment: add small noise for more training samples
                    for _ in range(10):
                        noisy = emb + np.random.normal(0, 0.01, emb.shape)
                        all_embs.append(noisy)
                        all_labels.append(i)

        if not all_embs:
            return

        all_embs = np.array(all_embs, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)

        model = FaceClassifierMLP(num_classes=len(students))
        trainer = MLPTrainer(model)
        print("    Training MLP on enrolled faces...")
        trainer.train(all_embs, all_labels, epochs=100, batch_size=16)
        trainer.save("face_mlp.pt")
        self.mlp_model = model.eval()

    def _retrain_lbph(self):
        """Retrain LBPH on enrolled face images."""
        if self.lbph_recognizer is None:
            return

        faces, labels = [], []
        students = get_all_students()

        for i, student in enumerate(students):
            face_dir = FACES_DIR / student["student_id"]
            if face_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_path in face_dir.glob(ext):
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (100, 100))
                            faces.append(img)
                            labels.append(i)

        if faces:
            self.lbph_recognizer.train(faces, np.array(labels))
            self.lbph_recognizer.write(str(MODELS_DIR / "lbph_model.yml"))
            print("    [OK] LBPH retrained")
