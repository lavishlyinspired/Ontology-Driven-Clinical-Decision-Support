#!/usr/bin/env python3
"""
Simple Backend Startup Script for LCA API
Starts FastAPI server without heavy dependencies
"""

import sys
import os
from pathlib import Path

# Add backend to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "backend"))

print("\n" + "=" * 80)
print("LUNG CANCER ASSISTANT - BACKEND API SERVER")
print("=" * 80)
print("\nStarting FastAPI server...")
print("  URL: http://localhost:8000")
print("  Docs: http://localhost:8000/docs")
print("  ReDoc: http://localhost:8000/redoc")
print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    import uvicorn
    
    # Start server with auto-reload
    uvicorn.run(
        "backend.src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
