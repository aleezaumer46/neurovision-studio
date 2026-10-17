"""
================================================================================
NeuroVision Studio Enterprise - Telemetry Data Models
================================================================================
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
try:
    from .database import Base
except ImportError:
    from database import Base

class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), index=True)
    model_name = Column(String(100), index=True)
    module = Column(String(50), index=True, default="classification")
    input_dimensions = Column(String(50), nullable=True)
    top_class = Column(String(100), index=True)
    confidence = Column(Float)
    latency_ms = Column(Float)
    fps = Column(Integer, default=0)
    hardware_backend = Column(String(50), default="webgl")
    is_ood = Column(Boolean, default=False)
    extra_metadata = Column(Text, nullable=True)
