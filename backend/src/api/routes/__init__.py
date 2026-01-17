"""
API Routes Module
Provides modular route organization for the LCA REST API
"""

from .patients import router as patients_router
from .treatments import router as treatments_router
from .guidelines import router as guidelines_router

__all__ = [
    "patients_router",
    "treatments_router", 
    "guidelines_router"
]
