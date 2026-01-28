"""
API Routes Module
Provides modular route organization for the LCA REST API
"""

from .patients import router as patients_router
from .patient_routes import router as patient_crud_router
from .treatments import router as treatments_router
from .guidelines import router as guidelines_router
from .analytics import router as analytics_router
from .analytics_detail import router as analytics_detail_router
from .audit import router as audit_router
from .audit_detail import router as audit_detail_router
from .biomarkers import router as biomarkers_router
from .patient_similarity import router as patient_similarity_router
from .biomarker_detail import router as biomarker_detail_router
from .counterfactual import router as counterfactual_router
from .export import router as export_router
from .system import router as system_router
from .digital_twin_api import router as digital_twin_router
from .chat import router as chat_router

__all__ = [
    "patients_router",
    "patient_crud_router",
    "treatments_router", 
    "guidelines_router",
    "analytics_router",
    "analytics_detail_router",
    "audit_router",
    "audit_detail_router",
    "biomarkers_router",
    "patient_similarity_router",
    "biomarker_detail_router",
    "counterfactual_router",
    "export_router",
    "system_router",
    "digital_twin_router",
    "chat_router"
]

