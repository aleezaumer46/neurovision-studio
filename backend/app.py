"""
================================================================================
NeuroVision Studio Enterprise - Async FastAPI Telemetry & Analytics Server
================================================================================
"""

import json
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

try:
    from .database import engine, Base, get_db
    from .models import InferenceLog
except ImportError:
    from database import engine, Base, get_db
    from models import InferenceLog

# Auto-create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="NeuroVision Studio Enterprise Telemetry API",
    version="2.0.0",
    description="High-throughput telemetry ingestion and aggregated edge analytics for client-side CV inference."
)

# Enable CORS for local Vite dev server and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class LogInferenceRequest(BaseModel):
    model_name: str = Field(..., example="MobileNetV2 (ImageNet)")
    module: str = Field("classification", example="classification")
    input_dimensions: Optional[str] = Field(None, example="640 × 480 px")
    top_label: Optional[str] = Field(None, example="Automobile / Car")
    top_class: Optional[str] = None
    confidence: float = Field(0, example=96.4)
    inference_time_ms: Optional[float] = Field(None, example=18.5)
    latency_ms: Optional[float] = None
    timestamp: Optional[datetime] = None
    fps: Optional[int] = Field(0, example=54)
    hardware_backend: Optional[str] = Field("webgl", example="webgl")
    is_ood: Optional[bool] = Field(False, example=False)
    extra_metadata: Optional[dict] = None

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "NeuroVision Enterprise Telemetry Server",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "SQLite / SQLAlchemy"
    }

@app.post("/api/log-inference")
def log_inference(payload: LogInferenceRequest, db: Session = Depends(get_db)):
    """
    Ingests an edge inference event from the client-side browser runtime.
    """
    top_label = payload.top_label or payload.top_class
    latency = payload.inference_time_ms if payload.inference_time_ms is not None else payload.latency_ms
    if top_label is None or latency is None:
        raise HTTPException(status_code=422, detail="top_label and inference_time_ms are required")

    record = InferenceLog(
        model_name=payload.model_name,
        module=payload.module,
        input_dimensions=payload.input_dimensions,
        top_class=top_label,
        confidence=payload.confidence,
        latency_ms=latency,
        timestamp=(payload.timestamp.replace(tzinfo=None) if payload.timestamp else datetime.now(timezone.utc).replace(tzinfo=None)),
        fps=payload.fps,
        hardware_backend=payload.hardware_backend,
        is_ood=payload.is_ood,
        extra_metadata=json.dumps(payload.extra_metadata) if payload.extra_metadata else None
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"status": "recorded", "id": record.id}

@app.get("/api/analytics")
def get_analytics(db: Session = Depends(get_db)):
    """
    Delivers aggregated edge inference telemetry for the frontend analytics panel:
    - Total Inferences
    - Average, Min, and Max Latency
    - Top 10 Detected Classes Breakdown
    - Inference Distribution by Model & Module
    - Hardware Backend Utilization
    """
    total_logs = db.query(InferenceLog).count()

    if total_logs == 0:
        return {
            "total_inferences": 0,
            "average_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "top_classes": [],
            "latency_trends": [],
            "module_distribution": {},
            "model_distribution": {},
            "backend_distribution": {},
            "recent_events": []
        }

    # Aggregate stats
    avg_latency = db.query(func.avg(InferenceLog.latency_ms)).scalar() or 0.0
    min_latency = db.query(func.min(InferenceLog.latency_ms)).scalar() or 0.0
    max_latency = db.query(func.max(InferenceLog.latency_ms)).scalar() or 0.0

    # Top detected classes
    class_counts = db.query(
        InferenceLog.top_class,
        func.count(InferenceLog.id).label("count")
    ).group_by(InferenceLog.top_class).order_by(desc("count")).limit(10).all()

    top_classes = [{"class_name": c[0], "count": c[1]} for c in class_counts]

    # Module breakdown
    module_rows = db.query(
        InferenceLog.module,
        func.count(InferenceLog.id)
    ).group_by(InferenceLog.module).all()
    module_distribution = {m[0]: m[1] for m in module_rows}

    # Model breakdown
    model_rows = db.query(
        InferenceLog.model_name,
        func.count(InferenceLog.id)
    ).group_by(InferenceLog.model_name).all()
    model_distribution = {m[0]: m[1] for m in model_rows}

    # Backend breakdown
    backend_rows = db.query(
        InferenceLog.hardware_backend,
        func.count(InferenceLog.id)
    ).group_by(InferenceLog.hardware_backend).all()
    backend_distribution = {b[0]: b[1] for b in backend_rows}

    # Recent 10 events
    recent = db.query(InferenceLog).order_by(desc(InferenceLog.id)).limit(10).all()
    recent_events = [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "module": r.module,
            "model": r.model_name,
            "top_class": r.top_class,
            "confidence": round(r.confidence, 1),
            "latency_ms": round(r.latency_ms, 1),
            "backend": r.hardware_backend,
            "is_ood": r.is_ood
        }
        for r in recent
    ]

    # Latency trends over the last 20 inferences
    trend_rows = db.query(InferenceLog.id, InferenceLog.latency_ms, InferenceLog.model_name).order_by(desc(InferenceLog.id)).limit(20).all()
    trend_rows.reverse()
    latency_trends = [
        {"point": idx + 1, "latency_ms": round(row[1], 1), "model": row[2]}
        for idx, row in enumerate(trend_rows)
    ]

    return {
        "total_inferences": total_logs,
        "average_latency_ms": round(float(avg_latency), 2),
        "min_latency_ms": round(float(min_latency), 2),
        "max_latency_ms": round(float(max_latency), 2),
        "top_classes": top_classes,
        "latency_trends": latency_trends,
        "module_distribution": module_distribution,
        "model_distribution": model_distribution,
        "backend_distribution": backend_distribution,
        "recent_events": recent_events
    }

@app.delete("/api/clear-logs")
def clear_logs(db: Session = Depends(get_db)):
    count = db.query(InferenceLog).delete()
    db.commit()
    return {"status": "cleared", "deleted_count": count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
