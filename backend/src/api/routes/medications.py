"""
Medications API Router
Provides RxNorm-based drug formulary and interaction endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from src.services.rxnorm_service import get_rxnorm_service
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Request/Response Models
class DrugInteractionRequest(BaseModel):
    drug_list: List[str] = Field(..., min_items=2, description="List of drug names")


class AddMedicationRequest(BaseModel):
    drug_name: str
    dose: str
    route: str
    frequency: str
    start_date: str


# Endpoints
@router.get("/search")
async def search_drugs(
    q: str = Query(..., min_length=2),
    drug_class: Optional[str] = Query("all")
):
    """Search drug formulary by name"""
    try:
        service = get_rxnorm_service()
        results = service.search_drug(
            query=q,
            drug_class=drug_class if drug_class != "all" else None
        )

        return {
            "status": "success",
            "query": q,
            "results": results
        }

    except Exception as e:
        logger.error(f"Drug search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interactions")
async def check_interactions(request: DrugInteractionRequest):
    """Check for drug-drug interactions"""
    try:
        service = get_rxnorm_service()
        results = service.check_drug_interactions(request.drug_list)

        return {
            "status": "success",
            "drug_count": len(request.drug_list),
            "interactions": results.get("interactions", [])
        }

    except Exception as e:
        logger.error(f"Interaction check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{rxcui}/alternatives")
async def get_alternatives(rxcui: str):
    """Get therapeutic alternatives within same class"""
    try:
        service = get_rxnorm_service()
        results = service.get_therapeutic_alternatives(rxcui)

        return {
            "status": "success",
            "drug": rxcui,
            "alternatives": results.get("alternatives", [])
        }

    except Exception as e:
        logger.error(f"Alternatives error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{rxcui}/details")
async def get_drug_details(rxcui: str):
    """Get full drug information"""
    try:
        service = get_rxnorm_service()
        details = service.get_drug_details(rxcui)

        if not details:
            raise HTTPException(status_code=404, detail=f"Drug {rxcui} not found")

        return {
            "status": "success",
            "drug": details
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drug details error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patients/{patient_id}/medications")
async def get_patient_medications(
    patient_id: str,
    active_only: bool = Query(True)
):
    """Get current and historical medication list"""
    # Placeholder - would query from database
    return {
        "status": "success",
        "patient_id": patient_id,
        "active_only": active_only,
        "medications": []  # Would be populated from DB
    }


@router.post("/patients/{patient_id}/medications")
async def add_patient_medication(
    patient_id: str,
    request: AddMedicationRequest
):
    """Add new medication with DDI check"""
    try:
        # Placeholder - would check existing meds and perform DDI check
        return {
            "status": "success",
            "patient_id": patient_id,
            "medication": request.dict(),
            "ddi_warnings": []  # Would be populated from DDI check
        }

    except Exception as e:
        logger.error(f"Add medication error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
