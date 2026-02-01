"""
Clinical Trials API Router
Provides ClinicalTrials.gov integration endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from src.services.clinical_trials_service import get_clinical_trials_service
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Request/Response Models
class PatientTrialMatchRequest(BaseModel):
    patient_id: Optional[str] = None
    patient_data: Optional[Dict[str, Any]] = None
    max_trials: int = 10


# Endpoints
@router.get("/search")
async def search_clinical_trials(
    condition: str = Query("lung cancer"),
    intervention: Optional[str] = Query(None),
    phase: Optional[List[str]] = Query(None),
    status: str = Query("RECRUITING"),
    max_results: int = Query(50)
):
    """Search clinical trials from ClinicalTrials.gov"""
    try:
        service = get_clinical_trials_service()
        results = await service.search_trials(
            condition=condition,
            intervention=intervention,
            phases=phase,
            status=status,
            max_results=max_results
        )

        return {
            "status": "success",
            "count": len(results.get("trials", [])),
            "trials": results.get("trials", [])
        }

    except Exception as e:
        logger.error(f"Trial search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/match-patient")
async def match_patient_to_trials(request: PatientTrialMatchRequest):
    """Match a patient to eligible clinical trials"""
    try:
        service = get_clinical_trials_service()

        # Get patient data
        patient_data = request.patient_data
        if not patient_data and request.patient_id:
            # Would fetch from database
            patient_data = {}

        if not patient_data:
            raise HTTPException(status_code=400, detail="Patient data required")

        results = await service.match_patient_to_trials(
            patient_data=patient_data,
            max_trials=request.max_trials
        )

        return {
            "status": "success",
            "trials": results.get("trials", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trial matching error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{nct_id}")
async def get_trial_details(nct_id: str):
    """Get full trial details with FHIR ResearchStudy"""
    try:
        service = get_clinical_trials_service()
        trial = await service.get_trial_by_nct_id(nct_id)

        if not trial:
            raise HTTPException(status_code=404, detail=f"Trial {nct_id} not found")

        return {
            "status": "success",
            "trial": trial
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trial details error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{nct_id}/store")
async def store_trial_in_neo4j(nct_id: str):
    """Store trial in Neo4j with SNOMED/LUCADA mapping"""
    try:
        service = get_clinical_trials_service()
        result = await service.store_trial_in_neo4j(nct_id)

        return {
            "status": "success",
            "nct_id": nct_id,
            "stored": result.get("stored", False)
        }

    except Exception as e:
        logger.error(f"Trial storage error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patients/{patient_id}/eligible-trials")
async def get_patient_eligible_trials(patient_id: str):
    """Get eligible trials for patient from Neo4j"""
    # Placeholder - would query from Neo4j
    return {
        "status": "success",
        "patient_id": patient_id,
        "trials": []  # Would be populated from Neo4j
    }
