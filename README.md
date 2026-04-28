# 🎓 Student Uniform Compliance Detection System

**ANN Course Project — 6th Semester**

An AI-powered web application that analyzes images (upload or live webcam) to detect student uniform compliance, recognize faces, and log rule violations.

## 🏗️ Architecture

```
Frontend (React + Vite :5173)
    ↕ REST API + WebSocket
Backend (FastAPI :8000)
    ↕ AI Pipeline
┌──────────────────────────────────────┐
│  Face Recognition  │ FaceNet + MLP   │
│  Uniform Detection │ YOLO + CNN      │
│  ID Card Detection │ YOLO + OCR      │
│  Scoring Engine    │ Weighted Score   │
└──────────────────────────────────────┘
    ↕
SQLite Database
```

## 🧠 ANN Concepts Highlighted

| Concept | Where It's Used |
|---------|----------------|
| **MLP (Multilayer Perceptron)** | `models/mlp_classifier.py` — classifies face embeddings |
| **CNN (Convolutional Neural Network)** | `models/cnn_uniform.py` — classifies uniform components |
| **Forward Propagation** | Every `forward()` method with detailed comments |
| **Backpropagation** | Training loops with `loss.backward()` |
| **ReLU Activation** | Used in both MLP and CNN |
| **Dropout Regularization** | Prevents overfitting on small datasets |
| **Batch Normalization** | Stabilizes training |
| **Softmax / Sigmoid** | Classification output layers |
| **CrossEntropy / BCE Loss** | Loss functions with explanations |
| **Adam Optimizer** | Adaptive learning rate |

---

## 📋 Prerequisites

### 1. Python 3.10+
```bash
python --version
```

### 2. Node.js 18+ & npm
Download from: https://nodejs.org/

### 3. Tesseract OCR (for ID card fallback)
```bash
# Windows (using Chocolatey)
choco install tesseract

# OR download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR
```

### 4. NVIDIA GPU (optional, for faster inference)
- Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- PyTorch will auto-detect CUDA

---

## 🚀 Setup & Run

### Backend

```bash
# 1. Navigate to backend
cd backend

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database with demo students
python init_db.py

# 5. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
# 1. Navigate to frontend
cd frontend

# 2. Install dependencies
npm install

# 3. Start dev server
npm run dev
```

### Access
- **Frontend**: http://localhost:5173
- **Backend API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

---

## 👤 Enrolling Student Faces

Before face recognition works, you need to enroll face photos:

### Option 1: API
```bash
# Upload face photos for STU001
curl -X POST http://localhost:8000/api/students/STU001/enroll \
  -F "files=@face1.jpg" \
  -F "files=@face2.jpg" \
  -F "files=@face3.jpg"
```

### Option 2: File System
1. Place 3-5 clear face photos in: `backend/data/student_faces/STU001/`
2. Restart the server (MLP retraining triggers automatically)

---

## 🔍 AI Pipeline — Fallback Strategy

### Face Recognition
| Priority | Method | Trigger |
|----------|--------|---------|
| Primary | FaceNet → Custom MLP | Always tried first |
| Fallback 1 | Haar Cascade → LBPH | Confidence < 70% |
| Fallback 2 | Manual Override (UI) | Both fail |

### Uniform Detection
| Priority | Method | Trigger |
|----------|--------|---------|
| Primary | YOLOv8 → Custom CNN | Always tried first |
| Fallback 1 | HSV Color Masking | CNN confidence < 60% |
| Fallback 2 | Edge-based Heuristic | Tuck-in uncertain |

### ID Card Detection
| Priority | Method | Trigger |
|----------|--------|---------|
| Primary | YOLOv8 Object Detection | Always tried first |
| Fallback | Tesseract OCR | YOLO confidence < 50% |

---

## 📊 Scoring Algorithm

| Component | Weight |
|-----------|--------|
| Correct Shirt | 30% |
| Correct Pant | 30% |
| Tucked In | 20% |
| ID Card Present | 20% |

**Confidence Penalty**: If model confidence < 80%, score is proportionally reduced.

**Labels**: ≥95% = "Best Professional Attire" | ≥80% = "Good" | ≥60% = "Needs Improvement" | <60% = "Non-Compliant"

---

## 📁 Project Structure

```
ANN/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry
│   │   ├── config.py            # Settings
│   │   ├── database.py          # SQLite ORM
│   │   ├── models/
│   │   │   ├── mlp_classifier.py    # Custom MLP ★
│   │   │   └── cnn_uniform.py      # Custom CNN ★
│   │   ├── pipeline/
│   │   │   ├── face_recognition.py  # Face detection
│   │   │   ├── uniform_detection.py # Uniform analysis
│   │   │   ├── id_card_detection.py # ID card check
│   │   │   └── scoring.py          # Score calculator
│   │   ├── routes/
│   │   │   ├── analyze.py       # Image analysis API
│   │   │   ├── stream.py        # WebSocket streaming
│   │   │   ├── students.py      # Student CRUD
│   │   │   └── violations.py    # Violation logs
│   │   └── utils/
│   │       ├── image_utils.py   # Image helpers
│   │       └── drawing.py       # Bounding box drawing
│   ├── data/                    # Models & face photos
│   ├── init_db.py               # DB seeder
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css            # Design system
│   │   ├── components/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── CameraFeed.jsx
│   │   │   ├── ImageUpload.jsx
│   │   │   ├── Scorecard.jsx
│   │   │   ├── IdentificationPanel.jsx
│   │   │   ├── ManualOverride.jsx
│   │   │   └── ViolationLog.jsx
│   │   └── utils/api.js
│   ├── package.json
│   └── vite.config.js
└── README.md
```

★ = Contains detailed ANN curriculum comments

---

## 🎓 Demo Students

| ID | Name | Department |
|----|------|-----------|
| STU001 | Aarav Sharma | Computer Science |
| STU002 | Priya Patel | Information Technology |
| STU003 | Rahul Verma | Computer Science |
| STU004 | Sneha Gupta | Electronics |
| STU005 | Vikram Singh | Mechanical Engineering |
