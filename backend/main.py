"""
================================================================================
NeuroVision Studio Enterprise - FastAPI Server Entrypoint (main.py)
================================================================================
"""

try:
    from .app import app
except ImportError:
    from app import app

if __name__ == "__main__":
    import uvicorn
    app_path = "backend.main:app" if __package__ else "main:app"
    uvicorn.run(app_path, host="127.0.0.1", port=8000, reload=True)
