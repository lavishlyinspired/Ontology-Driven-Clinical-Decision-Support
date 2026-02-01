"""
Monitoring API Router
Provides lab-drug monitoring protocol endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from src.services.lab_drug_service import get_lab_drug_service
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Request/Response Models
class DoseAssessmentRequest(BaseModel):
    drug_name: str
    lab_results: Dict[str, Dict[str, Any]] = Field(..., description="LOINC code -> {value, units}")


class MonitoringProtocolRequest(BaseModel):
    regimen: str
    start_date: str


# Endpoints
@router.get("/protocols/{regimen}")
async def get_monitoring_protocol(regimen: str):
    """Get lab monitoring protocol for a regimen"""
    try:
        service = get_lab_drug_service()
        protocol = service.get_monitoring_protocol(regimen)

        return {
            "status": "success",
            "regimen": regimen,
            "protocol": protocol
        }

    except Exception as e:
        logger.error(f"Protocol error: {e}")
        raise HTTPException(status_code=404, detail=f"Protocol for '{regimen}' not found")


@router.post("/assess-dose")
async def assess_dose_adjustment(request: DoseAssessmentRequest):
    """Assess if dose adjustment needed based on labs"""
    try:
        service = get_lab_drug_service()
        assessment = service.assess_dose_for_labs(
            drug_name=request.drug_name,
            lab_results=request.lab_results
        )

        return {
            "status": "success",
            "drug_name": request.drug_name,
            "assessment": assessment
        }

    except Exception as e:
        logger.error(f"Dose assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict-effects/{drug_name}")
async def predict_lab_effects(drug_name: str):
    """Get expected lab changes for a drug"""
    try:
        service = get_lab_drug_service()
        effects = service.check_drug_lab_effects(drug_name)

        return {
            "status": "success",
            "drug_name": drug_name,
            "effects": effects.get("lab_effects", [])
        }

    except Exception as e:
        logger.error(f"Effects prediction error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/patients/{patient_id}/monitoring-protocol")
async def get_patient_protocol(patient_id: str):
    """Get active monitoring protocol for patient"""
    # Placeholder - would query from database
    return {
        "status": "success",
        "patient_id": patient_id,
        "protocol": None  # Would be populated from DB
    }


@router.post("/patients/{patient_id}/monitoring-protocol")
async def create_patient_protocol(
    patient_id: str,
    request: MonitoringProtocolRequest
):
    """Create monitoring protocol for patient"""
    try:
        service = get_lab_drug_service()
        protocol = service.get_monitoring_protocol(request.regimen)

        # Placeholder - would store in database
        return {
            "status": "success",
            "patient_id": patient_id,
            "protocol": {
                **protocol,
                "start_date": request.start_date,
                "status": "active"
            }
        }

    except Exception as e:
        logger.error(f"Protocol creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
