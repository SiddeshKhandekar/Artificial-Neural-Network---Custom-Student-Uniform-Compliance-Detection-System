"""
============================================================
FastAPI Application Entry Point
============================================================
Initializes the API server, loads AI models at startup,
and mounts all route modules.
============================================================
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, DATA_DIR
from app.database import init_database
from app.routes import analyze, stream, students, violations, calibrate


# ── Global Pipeline Instances ───────────────────────────────
from app.pipeline.face_recognition import FaceRecognitionPipeline
from app.pipeline.uniform_detection import UniformDetectionPipeline
from app.pipeline.id_card_detection import IDCardDetectionPipeline

face_pipeline = FaceRecognitionPipeline()
uniform_pipeline = UniformDetectionPipeline()
id_card_pipeline = IDCardDetectionPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB and load AI models. Shutdown: cleanup."""
    print("\n" + "=" * 60)
    print("  Student Uniform Compliance Detection System")
    print("  Starting up...")
    print("=" * 60 + "\n")

    # Initialize database
    init_database()
    print("  [OK] Database initialized\n")

    # Load AI models (runs in thread to avoid blocking)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, face_pipeline.initialize)
    await loop.run_in_executor(None, uniform_pipeline.initialize)
    await loop.run_in_executor(None, id_card_pipeline.initialize)

    # Store pipelines in app state for access from routes
    app.state.face_pipeline = face_pipeline
    app.state.uniform_pipeline = uniform_pipeline
    app.state.id_card_pipeline = id_card_pipeline

    print("=" * 60)
    print("  [OK] All systems ready!")
    print("  -> API docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")

    yield  # App is running

    # Shutdown
    print("\n  Shutting down...")


# ── Create FastAPI App ──────────────────────────────────────
app = FastAPI(
    title="Student Uniform Compliance System",
    description="AI-powered uniform detection with face recognition, "
                "uniform classification, and ID card detection.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Routes ────────────────────────────────────────────
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(stream.router, tags=["Streaming"])
app.include_router(students.router, prefix="/api", tags=["Students"])
app.include_router(violations.router, prefix="/api", tags=["Violations"])
app.include_router(calibrate.router, prefix="/api", tags=["Calibration"])

# ── Static Files (for serving student face photos) ──────────
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


# ── Health Check ────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Student Uniform Compliance System",
        "models": {
            "face_recognition": face_pipeline._initialized,
            "uniform_detection": uniform_pipeline._initialized,
            "id_card_detection": id_card_pipeline._initialized,
        }
    }