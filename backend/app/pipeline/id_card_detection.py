"""
============================================================
ID Card Detection Pipeline -- Multi-Strategy
============================================================
Strategy 1: YOLO object detection for card-like objects
Strategy 2: Lanyard / Strap detection via color segmentation
Strategy 3: Rectangular contour detection for card shapes
Strategy 4: Tesseract OCR on chest/torso region (optional)
============================================================
"""

import cv2
import numpy as np

from app.config import (
    YOLO_MODEL_NAME, YOLO_CONFIDENCE, ID_CARD_CONFIDENCE_THRESHOLD,
    ID_CARD_OCR_KEYWORDS
)


class IDCardDetectionPipeline:
    """Detect ID cards using multiple visual strategies."""

    def __init__(self):
        self.yolo_model = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        print("  [ID Card Detection] Initializing...")

        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(YOLO_MODEL_NAME)
            print("    [OK] YOLOv8 loaded for ID detection")
        except Exception as e:
            print(f"    [FAIL] YOLOv8 failed: {e}")

        self._initialized = True
        print("  [ID Card Detection] Ready\n")

    def detect(self, image: np.ndarray, person_bbox: list = None) -> dict:
        """
        Detect if an ID card is visible using multiple strategies.
        
        Returns:
            {
                "detected": True/False,
                "confidence": 0.75,
                "method": "YOLO" | "Lanyard" | "Contour" | "OCR" | "None",
                "bbox": [x1, y1, x2, y2] or None
            }
        """
        self.initialize()

        # ID card is typically on upper body (12-55% of person height)
        x_offset = 0
        y_offset = 0
        
        if person_bbox:
            x1, y1, x2, y2 = person_bbox
            h = y2 - y1
            w = x2 - x1
            
            # ── ENHANCEMENT: Strict Center-Chest Cropping ──
            # Only search the central 50% of the person's width to avoid arms/sleeves
            chest_x1 = x1 + int(w * 0.25)
            chest_x2 = x1 + int(w * 0.75)
            chest_y1 = y1 + int(h * 0.12)
            chest_y2 = y1 + int(h * 0.55) 
            
            chest_region = image[chest_y1:chest_y2, chest_x1:chest_x2]
            x_offset, y_offset = chest_x1, chest_y1
        else:
            h, w = image.shape[:2]
            chest_y1 = int(h*0.12)
            # Full width fallback but biased to center
            chest_region = image[chest_y1:int(h*0.55), int(w*0.25):int(w*0.75)]
            x_offset, y_offset = int(w*0.25), chest_y1

        # ── Strategy 1: YOLO Detection ──────────────────
        yolo_result = self._yolo_detect(image, person_bbox)
        if yolo_result["confidence"] >= ID_CARD_CONFIDENCE_THRESHOLD:
            print(f"    [ID Card] Detected via YOLO (conf={yolo_result['confidence']:.2f})")
            return yolo_result

        # ── Strategy 2: White Card Detection (PRIMARY) ──────
        white_card_result = self._detect_white_card(chest_region)
        if white_card_result["detected"]:
            print(f"    [ID Card] Detected via WhiteCard (conf={white_card_result['confidence']:.2f})")
            if white_card_result["bbox"]:
                wx1, wy1, wx2, wy2 = white_card_result["bbox"]
                white_card_result["bbox"] = [wx1 + x_offset, wy1 + y_offset, wx2 + x_offset, wy2 + y_offset]
            return white_card_result

        # ── Strategy 3: Lanyard / Strap Detection (SUPPORTING)
        lanyard_result = self._detect_lanyard(chest_region)
        if lanyard_result["detected"] and lanyard_result["confidence"] > 0.65:
            print(f"    [ID Card] Detected via Lanyard (conf={lanyard_result['confidence']:.2f})")
            if lanyard_result["bbox"]:
                lx1, ly1, lx2, ly2 = lanyard_result["bbox"]
                lanyard_result["bbox"] = [lx1 + x_offset, ly1 + y_offset, lx2 + x_offset, ly2 + y_offset]
            return lanyard_result

        # ── Strategy 3: Rectangle / Card Shape Detection ─
        contour_result = self._detect_card_contour(chest_region)
        if contour_result["detected"]:
            print(f"    [ID Card] Detected via Contour (conf={contour_result['confidence']:.2f})")
            if contour_result["bbox"]:
                cx1, cy1, cx2, cy2 = contour_result["bbox"]
                contour_result["bbox"] = [cx1 + x_offset, cy1 + y_offset, cx2 + x_offset, cy2 + y_offset]
            return contour_result

        # ── Strategy 4: Tesseract OCR (optional) ────────
        ocr_result = self._ocr_detect(chest_region)
        if ocr_result["detected"]:
            print(f"    [ID Card] Detected via OCR (conf={ocr_result['confidence']:.2f})")
            if ocr_result["bbox"]:
                ox1, oy1, ox2, oy2 = ocr_result["bbox"]
                ocr_result["bbox"] = [ox1 + x_offset, oy1 + y_offset, ox2 + x_offset, oy2 + y_offset]
            return ocr_result

        print("    [ID Card] Not detected by any method")
        return {
            "detected": False,
            "confidence": 0.0,
            "method": "None",
            "bbox": None
        }

    def _yolo_detect(self, image: np.ndarray, person_bbox: list = None) -> dict:
        """
        Use YOLOv8 to detect card-like objects.
        """
        if not self.yolo_model:
            return {"detected": False, "confidence": 0.0, "method": "YOLO", "bbox": None}

        try:
            results = self.yolo_model(image, conf=0.25, verbose=False)
            card_classes = {67, 73, 26}  # cell phone, book, handbag

            best_conf = 0.0
            best_bbox = None

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id in card_classes:
                        coords = box.xyxy[0].cpu().numpy().astype(int).tolist()

                        # If we have a person bbox, check if detection is in chest area
                        if person_bbox:
                            px1, py1, px2, py2 = person_bbox
                            ph = py2 - py1
                            chest_top = py1 + int(ph * 0.15)
                            chest_bot = py1 + int(ph * 0.55)
                            obj_cy = (coords[1] + coords[3]) // 2

                            if not (chest_top <= obj_cy <= chest_bot):
                                continue

                        if conf > best_conf:
                            best_conf = conf
                            best_bbox = coords

            return {
                "detected": best_conf >= ID_CARD_CONFIDENCE_THRESHOLD,
                "confidence": best_conf,
                "method": "YOLO",
                "bbox": best_bbox
            }
        except Exception:
            return {"detected": False, "confidence": 0.0, "method": "YOLO", "bbox": None}

    def _detect_white_card(self, chest_region: np.ndarray) -> dict:
        """
        Look for a white rectangular object (the actual card).
        Supports both Portrait and Landscape orientations.
        """
        if chest_region.size == 0:
            return {"detected": False, "confidence": 0.0, "bbox": None}
            
        h, w = chest_region.shape[:2]
        
        # ── Step 1: Pre-processing for edge-based detection ──
        gray = cv2.cvtColor(chest_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect white regions specifically
        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # ── Step 2: Find Rectangular Contours ──
        cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_card_conf = 0.0
        best_card_bbox = None
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 400: # Minimum size for an ID card
                continue
                
            x, y, cw, ch = cv2.boundingRect(c)
            aspect_ratio = cw / max(ch, 1)
            width_ratio = cw / w
            
            # ID cards: Portrait (0.6) or Landscape (1.6)
            is_card_shape = 0.5 < aspect_ratio < 2.2
            is_card_size = 0.05 < width_ratio < 0.45
            
            if is_card_shape and is_card_size:
                # ── FEATURE: Centrality Bias ──
                # ID cards are almost always centered on the chest
                # Penalty for being near the horizontal edges (sleeves/arms)
                center_dist = abs((x + cw/2) - (w/2))
                centrality_factor = 1.0 - (center_dist / (w/2)) ** 2
                
                # Extent = Ratio of contour area to bounding box area
                rect_area = cw * ch
                extent = float(area) / rect_area
                
                # Solidity = Ratio of contour area to its convex hull area
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / max(hull_area, 1)

                if extent > 0.65 and solidity > 0.8:
                    # Confidence is high if it's both rectangular, solid, and centered
                    conf = (0.70 + (solidity * 0.30)) * centrality_factor
                    if conf > best_card_conf:
                        best_card_conf = conf
                        best_card_bbox = [x, y, x+cw, y+ch]

        return {
            "detected": best_card_conf > 0.5,
            "confidence": round(best_card_conf, 3),
            "method": "WhiteCardSensor (Portrait/Landscape)",
            "bbox": best_card_bbox
        }

    def _detect_lanyard(self, chest_region: np.ndarray) -> dict:
        """
        Color-Agnostic Lanyard Detection.
        Finds vertical strap structures based on geometry/edges.
        """
        if chest_region.size == 0:
            return {"detected": False, "confidence": 0.0, "bbox": None}
            
        try:
            h, w = chest_region.shape[:2]
            gray = cv2.cvtColor(chest_region, cv2.COLOR_BGR2GRAY)
            
            # Use Canny to find strap edges
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            strap_votes = []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = ch / max(cw, 1)
                
                # Lanyards are long, thin, vertical/slanted strips
                if aspect_ratio > 3.0 and ch > h * 0.2:
                    # Check for symmetry (straps usually come in pairs)
                    center_x = x + cw/2
                    strap_votes.append({
                        "bbox": [x, y, x+cw, y+ch],
                        "center_x": center_x,
                        "height": ch
                    })

            if len(strap_votes) >= 1:
                # If we find at least one vertical strap-like object
                best_strap = max(strap_votes, key=lambda x: x["height"])
                conf = 0.70 if len(strap_votes) > 1 else 0.50 # Higher conf if pair detected
                
                return {
                    "detected": True,
                    "confidence": conf,
                    "method": "Structural-Lanyard",
                    "bbox": best_strap["bbox"]
                }

            return {"detected": False, "confidence": 0.0, "method": "None", "bbox": None}

        except Exception:
            return {"detected": False, "confidence": 0.0, "method": "None", "bbox": None}

        except Exception as e:
            print(f"    [ID Card] Lanyard detection error: {e}")
            return {"detected": False, "confidence": 0.0, "method": "Lanyard", "bbox": None}

    def _detect_card_contour(self, chest_region: np.ndarray) -> dict:
        """
        Detect card-shaped rectangles in the chest region.
        """
        if chest_region is None or chest_region.size == 0:
            return {"detected": False, "confidence": 0.0, "method": "Contour", "bbox": None}

        try:
            h, w = chest_region.shape[:2]
            gray = cv2.cvtColor(chest_region, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

                if len(approx) == 4:
                    x, y, cw, ch = cv2.boundingRect(approx)
                    aspect = max(cw, ch) / max(min(cw, ch), 1)
                    area_ratio = (cw * ch) / (w * h)

                    if 1.2 <= aspect <= 2.5 and 0.02 < area_ratio < 0.30:
                        confidence = min(0.5 + area_ratio * 3, 0.9)
                        return {
                            "detected": True,
                            "confidence": round(confidence, 3),
                            "method": "Contour",
                            "bbox": [x, y, x + cw, y + ch]
                        }

            return {"detected": False, "confidence": 0.0, "method": "Contour", "bbox": None}
        except Exception as e:
            print(f"    [ID Card] Contour detection error: {e}")
            return {"detected": False, "confidence": 0.0, "method": "Contour", "bbox": None}

    def _ocr_detect(self, chest_region: np.ndarray) -> dict:
        """
        OCR Fallback.
        """
        if chest_region is None or chest_region.size == 0:
            return {"detected": False, "confidence": 0.0, "method": "OCR", "bbox": None}

        try:
            import pytesseract
            gray = cv2.cvtColor(chest_region, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            denoised = cv2.fastNlMeansDenoising(thresh, h=10)
            text = pytesseract.image_to_string(denoised, config='--psm 6')
            text_lower = text.lower().strip()

            if not text_lower:
                return {"detected": False, "confidence": 0.0, "method": "OCR", "bbox": None}

            matches = sum(1 for kw in ID_CARD_OCR_KEYWORDS if kw in text_lower)
            if matches > 0:
                confidence = min(matches / 3.0, 1.0)
                return {
                    "detected": True,
                    "confidence": confidence,
                    "method": "OCR",
                    "bbox": None,
                    "ocr_text": text_lower[:100]
                }

            return {"detected": False, "confidence": 0.0, "method": "OCR", "bbox": None}
        except Exception:
            return {"detected": False, "confidence": 0.0, "method": "OCR", "bbox": None}
