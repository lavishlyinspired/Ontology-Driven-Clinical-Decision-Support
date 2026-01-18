"""
API Routes Module
Provides modular route organization for the LCA REST API
"""

from .patients import router as patients_router
from .treatments import router as treatments_router
from .guidelines import router as guidelines_router
from .analytics import router as analytics_router
from .audit import router as audit_router
from .biomarkers import router as biomarkers_router

__all__ = [
    "patients_router",
    "treatments_router", 
    "guidelines_router",
    "analytics_router",
    "audit_router",
    "biomarkers_router"
]

