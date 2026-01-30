"""
Digital Twin API Endpoints

REST API for interacting with patient digital twins
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# Centralized logging
from ...logging_config import get_logger, log_execution

logger = get_logger(__name__)

from ...digital_twin import (
    DigitalTwinEngine,
    UpdateType,
    TwinState
)

router = APIRouter(prefix="/api/v1/digital-twin", tags=["Digital Twin"])

# In-memory twin registry (in production, use database)
_active_twins: Dict[str, DigitalTwinEngine] = {}


# ========================================
# REQUEST/RESPONSE MODELS
# ========================================

class TwinInitRequest(BaseModel):
    """Request to initialize a new digital twin"""
    patient_id: str = Field(..., description="Unique patient identifier")
    patient_data: Dict[str, Any] = Field(..., description="Complete patient clinical data")


class TwinUpdateRequest(BaseModel):
    """Request to update a digital twin"""
    update_type: str = Field(..., description="Type of update (lab_result, imaging, etc.)")
    data: Dict[str, Any] = Field(..., description="Update data")
    timestamp: Optional[str] = Field(None, description="Timestamp of update (ISO format)")


class TwinStateResponse(BaseModel):
    """Response with current twin state"""
    twin_id: str
    patient_id: str
    state: str
    created_at: str
    last_updated: str
    clinical_state: Dict[str, Any]
    active_alerts: List[Dict[str, Any]]
    predictions: Dict[str, Any]
    context_graph: Dict[str, Any]


class AlertResponse(BaseModel):
    """Individual alert"""
    alert_id: str
    severity: str
    category: str
    message: str
    timestamp: str
    confidence: float
    recommended_actions: List[str]


# ========================================
# ENDPOINTS
# ========================================

@router.post("/initialize", response_model=Dict[str, Any])
async def initialize_digital_twin(request: TwinInitRequest):
    """
    Initialize a new digital twin for a patient
    
    This creates the twin, runs baseline analysis, and sets up monitoring.
    
    Example:
        POST /api/v1/digital-twin/initialize
        {
            "patient_id": "P12345",
            "patient_data": {
                "age": 68,
                "stage": "IIIA",
                "histology": "Adenocarcinoma",
                "biomarkers": {"EGFR": "Ex19del"}
            }
        }
    """
    patient_id = request.patient_id
    
    # Check if twin already exists
    if patient_id in _active_twins:
        raise HTTPException(
            status_code=400,
            detail=f"Digital twin already exists for patient {patient_id}"
        )
    
    try:
        # Create twin
        twin = DigitalTwinEngine(patient_id=patient_id)
        
        # Initialize
        result = await twin.initialize(request.patient_data)
        
        # Store in registry
        _active_twins[patient_id] = twin
        
        return {
            "success": True,
            "message": f"Digital twin initialized for patient {patient_id}",
            **result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Twin initialization failed: {str(e)}")


@router.post("/{patient_id}/update", response_model=Dict[str, Any])
async def update_digital_twin(patient_id: str, request: TwinUpdateRequest):
    """
    Update digital twin with new clinical information
    
    This triggers cascading updates through the context graph and may generate alerts.
    
    Example:
        POST /api/v1/digital-twin/P12345/update
        {
            "update_type": "lab_result",
            "data": {
                "test": "EGFR mutation",
                "result": "T790M detected"
            }
        }
    """
    # Get twin
    twin = _active_twins.get(patient_id)
    if not twin:
        raise HTTPException(
            status_code=404,
            detail=f"Digital twin not found for patient {patient_id}. Initialize first."
        )
    
    try:
        # Parse timestamp if provided
        timestamp = None
        if request.timestamp:
            timestamp = datetime.fromisoformat(request.timestamp)
        
        # Update twin
        result = await twin.update({
            "type": request.update_type,
            "data": request.data,
            "timestamp": timestamp
        })
        
        return {
            "success": True,
            "message": f"Digital twin updated for patient {patient_id}",
            **result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Twin update failed: {str(e)}")


@router.get("/{patient_id}/state", response_model=TwinStateResponse)
async def get_twin_state(patient_id: str):
    """
    Get current state of digital twin
    
    Returns complete twin state including clinical data, alerts, and predictions.
    """
    twin = _active_twins.get(patient_id)
    if not twin:
        raise HTTPException(
            status_code=404,
            detail=f"Digital twin not found for patient {patient_id}"
        )
    
    state = twin.get_current_state()
    return state


@router.get("/{patient_id}/alerts", response_model=List[AlertResponse])
async def get_twin_alerts(patient_id: str, active_only: bool = True):
    """
    Get alerts for digital twin
    
    Query params:
        active_only: If true, return only active/unresolved alerts
    """
    twin = _active_twins.get(patient_id)
    if not twin:
        raise HTTPException(
            status_code=404,
            detail=f"Digital twin not found for patient {patient_id}"
        )
    
    alerts = twin.active_alerts
    
    return [
        {
            "alert_id": alert.alert_id,
            "severity": alert.severity,
            "category": alert.category,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "confidence": alert.confidence,
            "recommended_actions": alert.recommended_actions
        }
        for alert in alerts
    ]


@router.get("/{patient_id}/predictions", response_model=Dict[str, Any])
async def get_twin_predictions(patient_id: str):
    """
    Get trajectory predictions for digital twin
    
    Returns predicted disease pathways with probabilities and PFS estimates.
    """
    twin = _active_twins.get(patient_id)
    if not twin:
        raise HTTPException(
            status_code=404,
            detail=f"Digital twin not found for patient {patient_id}"
        )
    
    predictions = await twin.predict_trajectories()
    
    return {
        "patient_id": patient_id,
        "generated_at": datetime.now().isoformat(),
        **predictions
    }


@router.get("/{patient_id}/reasoning/{recommendation_id}", response_model=List[Dict[str, Any]])
async def get_reasoning_chain(patient_id: str, recommendation_id: str):
    """
    Get complete reasoning chain for a recommendation
    
    Shows audit trail: data → agents → reasoning → recommendation
    """
    twin = _active_twins.get(patient_id)
    if not twin:
        raise HTTPException(
            status_code=404,
            detail=f"Digital twin not found for patient {patient_id}"
        )
    
    chain = twin.get_reasoning_chain(recommendation_id)
    
    return chain


@router.get("/{patient_id}/export", response_model=Dict[str, Any])
async def export_twin(patient_id: str):
    """
    Export complete digital twin state
    
    Returns serializable representation for backup or transfer.
    """
    twin = _active_twins.get(patient_id)
    if not twin:
        raise HTTPException(
            status_code=404,
            detail=f"Digital twin not found for patient {patient_id}"
        )
    
    export_data = twin.export_twin()
    
    return export_data


@router.delete("/{patient_id}")
async def delete_twin(patient_id: str):
    """
    Delete/deactivate a digital twin
    
    Removes twin from active registry (in production, would archive to database).
    """
    if patient_id not in _active_twins:
        raise HTTPException(
            status_code=404,
            detail=f"Digital twin not found for patient {patient_id}"
        )
    
    twin = _active_twins[patient_id]
    
    # Export for archival (in production, save to database)
    export_data = twin.export_twin()
    
    # Remove from active registry
    del _active_twins[patient_id]
    
    return {
        "success": True,
        "message": f"Digital twin for patient {patient_id} deactivated",
        "archived_data_available": True,
        "snapshots_archived": len(export_data.get("snapshots", []))
    }


@router.get("/active", response_model=List[Dict[str, str]])
async def list_active_twins():
    """
    List all active digital twins
    
    Returns basic info for all twins currently in memory.
    """
    twins = []
    for patient_id, twin in _active_twins.items():
        twins.append({
            "patient_id": patient_id,
            "twin_id": twin.twin_id,
            "state": twin.state.value,
            "created_at": twin.created_at.isoformat(),
            "last_updated": twin.last_updated.isoformat(),
            "active_alerts": len(twin.active_alerts)
        })
    
    return twins


# ========================================
# HEALTH CHECK
# ========================================

@router.get("/health")
async def digital_twin_health():
    """Health check for digital twin service"""
    return {
        "status": "healthy",
        "service": "digital_twin_engine",
        "active_twins": len(_active_twins),
        "timestamp": datetime.now().isoformat()
    }
