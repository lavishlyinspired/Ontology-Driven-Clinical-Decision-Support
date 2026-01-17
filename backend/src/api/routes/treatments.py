"""
Treatment Routes - API endpoints for treatment operations
Implements treatment-related endpoints from LCA_Complete_Implementation_Plan
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

router = APIRouter(prefix="/treatments", tags=["Treatments"])


# ==================== Response Models ====================

class TreatmentTypeResponse(BaseModel):
    """Response model for treatment type"""
    treatment_type: str
    description: str
    typical_intent: str
    snomed_code: Optional[str] = None


class TreatmentStatisticsResponse(BaseModel):
    """Response model for treatment statistics"""
    treatment_type: str
    patient_count: int
    avg_survival_days: Optional[float]
    median_survival_days: Optional[float]
    outcome_types: List[str]
    avg_age: Optional[float]
    stages_treated: List[str]


class TreatmentRecommendationResponse(BaseModel):
    """Response model for treatment recommendation"""
    treatment_type: str
    rule_id: str
    rule_source: str
    evidence_level: str
    treatment_intent: str
    survival_benefit: Optional[str]
    contraindications: List[str]
    priority: int
    confidence_score: float


# ==================== Endpoints ====================

# Available treatment types based on LUCADA ontology
TREATMENT_TYPES = [
    {
        "treatment_type": "Surgery",
        "description": "Surgical resection of lung tumor (lobectomy, pneumonectomy, wedge resection)",
        "typical_intent": "Curative",
        "snomed_code": "387713003"
    },
    {
        "treatment_type": "Chemotherapy",
        "description": "Systemic chemotherapy with platinum-based regimens",
        "typical_intent": "Curative/Palliative",
        "snomed_code": "367336001"
    },
    {
        "treatment_type": "Radiotherapy",
        "description": "External beam radiotherapy or SABR",
        "typical_intent": "Curative/Palliative",
        "snomed_code": "108290001"
    },
    {
        "treatment_type": "Chemoradiotherapy",
        "description": "Concurrent chemotherapy and radiotherapy",
        "typical_intent": "Curative",
        "snomed_code": "703423002"
    },
    {
        "treatment_type": "Immunotherapy",
        "description": "PD-1/PD-L1 checkpoint inhibitors (pembrolizumab, nivolumab)",
        "typical_intent": "Palliative",
        "snomed_code": "76334006"
    },
    {
        "treatment_type": "TargetedTherapy",
        "description": "Molecular targeted therapy (EGFR TKIs, ALK inhibitors)",
        "typical_intent": "Palliative",
        "snomed_code": "416608005"
    },
    {
        "treatment_type": "PalliativeCare",
        "description": "Symptom management and supportive care",
        "typical_intent": "Palliative",
        "snomed_code": "103735009"
    },
    {
        "treatment_type": "ActiveMonitoring",
        "description": "Watchful waiting with regular surveillance",
        "typical_intent": "Observation",
        "snomed_code": "373787003"
    },
    {
        "treatment_type": "Brachytherapy",
        "description": "Internal radiation therapy",
        "typical_intent": "Palliative",
        "snomed_code": "152198000"
    }
]


@router.get("/", response_model=List[TreatmentTypeResponse])
async def list_treatment_types():
    """
    List all available treatment types.
    
    Returns treatment types from the LUCADA ontology with SNOMED-CT codes.
    """
    return [TreatmentTypeResponse(**t) for t in TREATMENT_TYPES]


@router.get("/{treatment_type}", response_model=TreatmentTypeResponse)
async def get_treatment_type(treatment_type: str):
    """
    Get details for a specific treatment type.
    """
    for t in TREATMENT_TYPES:
        if t["treatment_type"].lower() == treatment_type.lower():
            return TreatmentTypeResponse(**t)
    
    raise HTTPException(status_code=404, detail=f"Treatment type '{treatment_type}' not found")


@router.get("/{treatment_type}/statistics", response_model=TreatmentStatisticsResponse)
async def get_treatment_statistics(treatment_type: str):
    """
    Get outcome statistics for a specific treatment type.
    
    Returns aggregated statistics including:
    - Patient count
    - Average and median survival
    - Outcome types
    - Typical patient demographics
    """
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # First check if treatment type exists
        valid_types = [t["treatment_type"].lower() for t in TREATMENT_TYPES]
        if treatment_type.lower() not in valid_types:
            raise HTTPException(status_code=404, detail=f"Treatment type '{treatment_type}' not found")
        
        # Get statistics from Neo4j
        if hasattr(lca_service, 'graph_db') and lca_service.graph_db:
            stats = lca_service.graph_db.get_treatment_statistics(treatment_type)
            
            return TreatmentStatisticsResponse(
                treatment_type=treatment_type,
                patient_count=stats.get("patient_count", 0),
                avg_survival_days=stats.get("avg_survival_days"),
                median_survival_days=stats.get("median_survival_days"),
                outcome_types=stats.get("outcome_types", []),
                avg_age=stats.get("avg_age"),
                stages_treated=stats.get("stages_treated", [])
            )
        
        # Return empty stats if Neo4j not available
        return TreatmentStatisticsResponse(
            treatment_type=treatment_type,
            patient_count=0,
            avg_survival_days=None,
            median_survival_days=None,
            outcome_types=[],
            avg_age=None,
            stages_treated=[]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")


@router.post("/recommend")
async def recommend_treatments(
    tnm_stage: str = Query(..., description="TNM stage"),
    histology_type: str = Query(..., description="Histology type"),
    performance_status: int = Query(..., ge=0, le=4, description="WHO Performance Status"),
    biomarkers: Optional[List[str]] = Query(default=None, description="Biomarker status")
):
    """
    Get treatment recommendations for a patient profile.
    
    Uses the NICE guideline rules to determine applicable treatments.
    This is a quick lookup without creating a patient record.
    """
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Build patient data for classification
        patient_data = {
            "tnm_stage": tnm_stage,
            "histology_type": histology_type,
            "performance_status": performance_status,
            "biomarkers": biomarkers or []
        }
        
        # Get recommendations from rule engine
        if hasattr(lca_service, 'rule_engine'):
            recommendations = lca_service.rule_engine.classify_patient(patient_data)
            
            return {
                "patient_profile": patient_data,
                "recommendations": [
                    TreatmentRecommendationResponse(
                        treatment_type=r.get("recommended_treatment", "Unknown"),
                        rule_id=r.get("rule_id", ""),
                        rule_source=r.get("source", "NICE Guidelines"),
                        evidence_level=r.get("evidence_level", "Grade C"),
                        treatment_intent=r.get("treatment_intent", "Unknown"),
                        survival_benefit=r.get("survival_benefit"),
                        contraindications=r.get("contraindications", []),
                        priority=r.get("priority", 0),
                        confidence_score=0.85
                    ).model_dump()
                    for r in recommendations
                ],
                "total_recommendations": len(recommendations)
            }
        
        return {"patient_profile": patient_data, "recommendations": [], "total_recommendations": 0}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@router.get("/outcomes/summary")
async def get_outcomes_summary():
    """
    Get summary of treatment outcomes across all treatment types.
    
    Provides an overview of treatment effectiveness in the patient cohort.
    """
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        summaries = []
        
        if hasattr(lca_service, 'graph_db') and lca_service.graph_db:
            for treatment in TREATMENT_TYPES:
                stats = lca_service.graph_db.get_treatment_statistics(treatment["treatment_type"])
                if stats.get("patient_count", 0) > 0:
                    summaries.append({
                        "treatment_type": treatment["treatment_type"],
                        "patient_count": stats.get("patient_count", 0),
                        "avg_survival_days": stats.get("avg_survival_days"),
                        "typical_intent": treatment["typical_intent"]
                    })
        
        return {
            "treatment_summaries": summaries,
            "total_treatments_tracked": len(summaries)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve outcomes summary: {str(e)}")
