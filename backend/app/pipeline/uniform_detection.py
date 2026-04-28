"""
============================================================
Uniform Detection Pipeline -- Multi-Strategy
============================================================
Primary:   YOLOv8 person detection -> Custom CNN classification
Fallback1: HSV color masking on body regions
Fallback2: 3D Color Histogram for tuck-in status
============================================================
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from app.config import (
    DEVICE, YOLO_MODEL_NAME, YOLO_CONFIDENCE, YOLO_PERSON_CLASS_ID,
    UNIFORM_SHIRT_HSV_LOW, UNIFORM_SHIRT_HSV_HIGH,
    UNIFORM_PANT_HSV_LOW, UNIFORM_PANT_HSV_HIGH,
    UNIFORM_CNN_CONFIDENCE_THRESHOLD, TUCKIN_OVERLAP_THRESHOLD_PX,
    UNIFORM_IMAGE_SIZE, MODELS_DIR
)
from app.models.cnn_uniform import UniformClassifierCNN, CNNTrainer


class UniformDetectionPipeline:
    """Multi-strategy uniform compliance detection."""

    def __init__(self):
        self.yolo_model = None
        self.cnn_model = None
        self.transform = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        print("  [Uniform Detection] Initializing...")

        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(YOLO_MODEL_NAME)
            print("    [OK] YOLOv8 loaded")
        except Exception as e:
            print(f"    [FAIL] YOLOv8 failed: {e}")

        self.cnn_model = CNNTrainer.load("uniform_cnn.pt")
        if self.cnn_model:
            print("    [OK] Uniform CNN loaded")
        else:
            print("    [SKIP] No trained CNN (using fallback methods)")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(UNIFORM_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._initialized = True
        print("  [Uniform Detection] Ready\n")

    def detect(self, image: np.ndarray, face_results: dict = None) -> dict:
        """Detect uniform compliance."""
        self.initialize()
        results = {"persons": []}

        # Step 1: Detect persons (with face-based fallback)
        person_bboxes = self._detect_persons(image, face_results)

        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            person_crop = image[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            h = y2 - y1
            # Adjust waistline slightly lower for better shirt/pant split
            waist_y = int(h * 0.54)
            upper_body = person_crop[0:waist_y, :]
            lower_body = person_crop[waist_y:, :]

            upper_bbox = [x1, y1, x2, y1 + waist_y]
            lower_bbox = [x1, y1 + waist_y, x2, y2]

            shirt_result = self._classify_region(upper_body, "shirt")
            pant_result = self._classify_region(lower_body, "pant")
            tucked_result = self._check_tuckin_cnn(person_crop)

            print(f"    [Uniform] CNN results: shirt={shirt_result['confidence']:.2f}, pant={pant_result['confidence']:.2f}, tucked={tucked_result['confidence']:.2f}")

            if shirt_result["confidence"] < UNIFORM_CNN_CONFIDENCE_THRESHOLD:
                shirt_result = self._hsv_check(upper_body, UNIFORM_SHIRT_HSV_LOW, UNIFORM_SHIRT_HSV_HIGH, "shirt")
                print(f"    [Uniform] HSV shirt: detected={shirt_result['detected']}, conf={shirt_result['confidence']:.2f}")

            if pant_result["confidence"] < UNIFORM_CNN_CONFIDENCE_THRESHOLD:
                pant_result = self._hsv_check(lower_body, UNIFORM_PANT_HSV_LOW, UNIFORM_PANT_HSV_HIGH, "pant")
                print(f"    [Uniform] HSV pant: detected={pant_result['detected']}, conf={pant_result['confidence']:.2f}")

            if tucked_result["confidence"] < UNIFORM_CNN_CONFIDENCE_THRESHOLD:
                tucked_result = self._tuckin_heuristic(person_crop, waist_y)
                print(f"    [Uniform] Dynamic heuristic: detected={tucked_result['detected']}, conf={tucked_result['confidence']}")

            # Use dynamic waist_y if available from heuristic
            waist_y_actual = tucked_result.get("waist_y", waist_y)
            
            upper_bbox = [x1, y1, x2, y1 + waist_y_actual]
            lower_bbox = [x1, y1 + waist_y_actual, x2, y2]
            tucked_bbox = [x1, y1 + waist_y_actual - 10, x2, y1 + waist_y_actual + 10]

            results["persons"].append({
                "bbox": bbox,
                "shirt": shirt_result,
                "pant": pant_result,
                "tucked": tucked_result,
                "upper_body_bbox": upper_bbox,
                "lower_body_bbox": lower_bbox,
                "tucked_bbox": tucked_bbox,
            })

        return results

    def _detect_persons(self, image: np.ndarray, face_results: dict = None) -> list:
        h, w = image.shape[:2]
        
        # 1. Primary: YOLO
        if self.yolo_model:
            try:
                results = self.yolo_model(image, conf=YOLO_CONFIDENCE, verbose=False)
                bboxes = []
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == YOLO_PERSON_CLASS_ID:
                            coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                            bboxes.append(coords)
                if bboxes:
                    return bboxes
            except Exception:
                pass
        
        # 2. Fallback: Estimate body from Face detection
        if face_results and face_results.get("faces"):
            bboxes = []
            for face in face_results["faces"]:
                fx1, fy1, fx2, fy2 = face["bbox"]
                fw = fx2 - fx1
                fh = fy2 - fy1
                
                # Estimate body: width is 4x face, height is 8x face
                px1 = max(0, fx1 - int(fw * 1.5))
                py1 = max(0, fy1 - int(fh * 0.5))
                px2 = min(w, fx2 + int(fw * 1.5))
                py2 = min(h, fy1 + int(fh * 7.0))
                bboxes.append([px1, py1, px2, py2])
            
            if bboxes:
                print("    [Uniform] YOLO failed, using face-based body estimation")
                return bboxes

        # 3. Final Fallback: Center of Image (Portrait)
        print("    [Uniform] All detection failed, using center-portrait fallback")
        return [[int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)]]

    def _classify_region(self, region: np.ndarray, target: str) -> dict:
        if self.cnn_model is None or region.size == 0:
            return {"detected": False, "confidence": 0.0, "method": "None"}
        try:
            rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(DEVICE)
            preds = self.cnn_model.predict(tensor.squeeze(0))
            conf = preds.get(target, 0.0)
            return {"detected": conf > 0.5, "confidence": conf, "method": "CNN"}
        except Exception:
            return {"detected": False, "confidence": 0.0, "method": "None"}

    def _check_tuckin_cnn(self, person_crop: np.ndarray) -> dict:
        if self.cnn_model is None or person_crop.size == 0:
            return {"detected": False, "confidence": 0.0, "method": "None"}
        try:
            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(DEVICE)
            preds = self.cnn_model.predict(tensor.squeeze(0))
            conf = preds.get("tucked", 0.0)
            return {"detected": conf > 0.5, "confidence": conf, "method": "CNN"}
        except Exception:
            return {"detected": False, "confidence": 0.0, "method": "None"}

    def _hsv_check(self, region: np.ndarray, hsv_low: tuple, hsv_high: tuple, label: str) -> dict:
        if region.size == 0:
            return {"detected": False, "confidence": 0.0, "method": "HSV"}
        try:
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            total_pixels = mask.shape[0] * mask.shape[1]
            matching_pixels = cv2.countNonZero(mask)
            ratio = matching_pixels / max(total_pixels, 1)
            detected = ratio > 0.30
            confidence = min(ratio * 2, 1.0)
            return {"detected": detected, "confidence": confidence, "method": "HSV"}
        except Exception:
            return {"detected": False, "confidence": 0.0, "method": "HSV"}

    def _tuckin_heuristic(self, person_crop: np.ndarray, initial_waist_y: int) -> dict:
        """
        Dynamically find the waistline and check for tuck-in status.
        """
        if person_crop.size == 0:
            return {"detected": False, "confidence": 0.0, "method": "DynamicSplit", "waist_y": initial_waist_y}
        
        try:
            h, w = person_crop.shape[:2]
            hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)

            # 1. Dynamically find the best waistline split
            # Scan the middle 40% of the body
            start_y = int(h * 0.35)
            end_y = int(h * 0.75)
            
            best_waist_y = initial_waist_y
            max_diff = -1
            
            # Sample dominant colors
            top_color = np.mean(hsv[int(h*0.1):int(h*0.3), int(w*0.3):int(w*0.7)], axis=(0, 1))
            bot_color = np.mean(hsv[int(h*0.7):int(h*0.9), int(w*0.3):int(w*0.7)], axis=(0, 1))

            for y in range(start_y, end_y, 5):
                # Compare row above and row below
                row_above = hsv[y-5:y, int(w*0.2):int(w*0.8)]
                row_below = hsv[y:y+5, int(w*0.2):int(w*0.8)]
                
                diff = np.linalg.norm(np.mean(row_above, axis=(0, 1)) - np.mean(row_below, axis=(0, 1)))
                if diff > max_diff:
                    max_diff = diff
                    best_waist_y = y

            print(f"    [Tucked] Dynamic waist found at {best_waist_y/h:.2f} (diff={max_diff:.1f})")
            
            # 2. Analyze the split for Tucked vs Untucked
            # Region 1: Shirt area (immediately above waist)
            # Region 2: Pant area (immediately below waist)
            shirt_region = hsv[max(0, best_waist_y-40):best_waist_y, int(w*0.2):int(w*0.8)]
            below_region = hsv[best_waist_y:min(h, best_waist_y+40), int(w*0.2):int(w*0.8)]
            
            if shirt_region.size == 0 or below_region.size == 0:
                 return {"detected": False, "confidence": 0.0, "method": "DynamicSplit", "waist_y": best_waist_y}

            # ── FEATURE 1: Enhanced Belt & Buckle Detection ──────
            # Search a wider area (40 pixels) around the waistline
            waist_zone = person_crop[max(0, best_waist_y-20):min(h, best_waist_y+20), int(w*0.2):int(w*0.8)]
            gray_waist = cv2.cvtColor(waist_zone, cv2.COLOR_BGR2GRAY)
            
            # Use adaptive thresholding to find the buckle (bright spot in dark zone)
            # Higher sensitivity for diverse lighting
            buckle_mask = cv2.threshold(gray_waist, 100, 255, cv2.THRESH_BINARY)[1]
            cnts, _ = cv2.findContours(buckle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            buckle_detected = False
            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                # Buckle characteristics: medium size, centered
                if 10 < bw < 80 and 8 < bh < 50:
                    center_dist = abs((bx + bw/2) - (waist_zone.shape[1]/2))
                    if center_dist < waist_zone.shape[1] * 0.3:
                        buckle_detected = True
                        break

            # Check for "Dark Strip" (Belt fabric)
            # Scan multiple rows in the waist zone for a dark horizontal band
            has_dark_belt = False
            for y_off in range(0, waist_zone.shape[0]-5, 5):
                row_v = np.mean(cv2.cvtColor(waist_zone[y_off:y_off+5, :], cv2.COLOR_BGR2HSV)[:, :, 2])
                if row_v < 65: # Dark belt detected
                    has_dark_belt = True
                    break
            
            belt_detected = buckle_detected or has_dark_belt
            
            # ── FEATURE 1.5: Color Similarity ───────────────────
            hist_shirt = cv2.calcHist([shirt_region], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist_below = cv2.calcHist([below_region], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist_shirt, hist_shirt, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_below, hist_below, 0, 1, cv2.NORM_MINMAX)
            similarity = cv2.compareHist(hist_shirt, hist_below, cv2.HISTCMP_CORREL)
            
            # ── FEATURE 2: Split Position ───────────────────────
            split_ratio = best_waist_y / h
            is_position_tucked = 0.35 < split_ratio < 0.65
            
            print(f"    [Tucked] Split={split_ratio:.2f}, Buckle={buckle_detected}, Belt={has_dark_belt}")

            # FINAL DECISION
            if buckle_detected:
                is_tucked = True
                confidence = 1.0 # Silver buckle is a definitive signal
            elif has_dark_belt and is_position_tucked:
                is_tucked = True
                confidence = 0.90
            elif split_ratio > 0.68: # Definitely too low
                is_tucked = False
                confidence = 0.95
            else:
                # Fallback to color similarity
                is_tucked = similarity < 0.65
                confidence = 0.85 if is_tucked else 0.70
            
            return {
                "detected": is_tucked,
                "confidence": round(float(confidence), 3),
                "method": "AdvancedBeltSensor",
                "waist_y": best_waist_y
            }
        except Exception as e:
            print(f"    [Tucked] Error: {e}")
            return {"detected": False, "confidence": 0.0, "method": "DynamicSplit", "waist_y": initial_waist_y}
