"""
Laboratory API Router
Provides LOINC-based lab result interpretation endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from src.services.loinc_service import get_loinc_service
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Request/Response Models
class LabInterpretationRequest(BaseModel):
    loinc_code: str = Field(..., description="LOINC code")
    value: float = Field(..., description="Lab value")
    units: str = Field(..., description="Units")
    patient_context: Optional[Dict[str, Any]] = Field(None, description="Patient context (sex, age, etc.)")


class BatchLabInterpretationRequest(BaseModel):
    patient_id: str
    lab_results: List[Dict[str, Any]]


# Endpoints
@router.post("/interpret")
async def interpret_lab_result(request: LabInterpretationRequest):
    """Interpret a single lab result with clinical context"""
    try:
        service = get_loinc_service()
        result = service.interpret_lab_result(
            loinc_code=request.loinc_code,
            value=request.value,
            unit=request.units,
            patient_context=request.patient_context or {}
        )

        return {
            "status": "success",
            "interpretation": {
                "loinc_code": result.loinc_code,
                "value": result.value,
                "unit": result.unit,
                "interpretation": result.interpretation,
                "reference_range": result.reference_range,
                "clinical_significance": result.clinical_significance,
                "recommendations": result.recommendations
            }
        }

    except Exception as e:
        logger.error(f"Lab interpretation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/panels/{panel_type}")
async def get_lab_panel(panel_type: str):
    """Get a predefined lab panel"""
    try:
        service = get_loinc_service()
        panel = service.get_lung_cancer_panel(panel_type)

        return {
            "status": "success",
            "panel_type": panel_type,
            "panel": panel
        }

    except Exception as e:
        logger.error(f"Lab panel error: {e}")
        raise HTTPException(status_code=404, detail=f"Panel '{panel_type}' not found")


@router.post("/batch-interpret")
async def batch_interpret_labs(request: BatchLabInterpretationRequest):
    """Interpret multiple lab results for a patient"""
    try:
        service = get_loinc_service()

        interpretations = []
        critical_flags = []

        for lab in request.lab_results:
            result = service.interpret_lab_result(
                loinc_code=lab.get("loinc_code"),
                value=lab.get("value"),
                unit=lab.get("units", "")
            )

            interpretation = {
                "loinc_code": result.loinc_code,
                "interpretation": result.interpretation,
                "clinical_significance": result.clinical_significance
            }
            interpretations.append(interpretation)

            if result.interpretation in ["critical_low", "critical_high"]:
                critical_flags.append(interpretation)

        return {
            "status": "success",
            "patient_id": request.patient_id,
            "interpretations": interpretations,
            "critical_flags": critical_flags
        }

    except Exception as e:
        logger.error(f"Batch interpretation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patients/{patient_id}/labs")
async def get_patient_labs(
    patient_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    loinc_code: Optional[str] = Query(None)
):
    """Get lab result history for a patient"""
    # Placeholder - would query from database
    return {
        "status": "success",
        "patient_id": patient_id,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "loinc_code": loinc_code
        },
        "labs": []  # Would be populated from DB
    }


@router.get("/search")
async def search_loinc(
    q: str = Query(..., min_length=2),
    category: Optional[str] = Query("all"),
    max_results: int = Query(20)
):
    """Search LOINC codes by query"""
    try:
        service = get_loinc_service()
        results = service.search_loinc(
            query=q,
            category=category if category != "all" else None,
            limit=max_results
        )

        return {
            "status": "success",
            "query": q,
            "count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"LOINC search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
