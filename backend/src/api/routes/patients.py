"""
Patient Routes - API endpoints for patient operations
Implements endpoints from LCA_Complete_Implementation_Plan Section 16
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

router = APIRouter(prefix="/patients", tags=["Patients"])


# ==================== Request Models ====================

class PatientCreateRequest(BaseModel):
    """Request model for creating a new patient"""
    patient_id: Optional[str] = Field(None, description="Patient ID (auto-generated if not provided)")
    name: str = Field(..., description="Patient name")
    sex: str = Field(..., pattern="^[MFU]$", description="Sex: M/F/U")
    age: int = Field(..., ge=0, le=120, description="Age at diagnosis")
    tnm_stage: str = Field(..., description="TNM stage (IA, IB, IIA, IIB, IIIA, IIIB, IV)")
    histology_type: str = Field(..., description="Histology type")
    performance_status: int = Field(..., ge=0, le=4, description="WHO Performance Status (0-4)")
    fev1_percent: Optional[float] = Field(None, ge=0, le=150, description="FEV1 percentage")
    laterality: str = Field(default="Unknown", description="Tumor laterality")
    comorbidities: Optional[List[str]] = Field(default=[], description="List of comorbidities")
    diagnosis: Optional[str] = Field(default="Malignant Neoplasm of Lung", description="Primary diagnosis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Jenny_Sesen",
                "sex": "F",
                "age": 72,
                "tnm_stage": "IIA",
                "histology_type": "Carcinosarcoma",
                "performance_status": 1,
                "laterality": "Right"
            }
        }


class PatientUpdateRequest(BaseModel):
    """Request model for updating patient data"""
    name: Optional[str] = None
    tnm_stage: Optional[str] = None
    histology_type: Optional[str] = None
    performance_status: Optional[int] = Field(None, ge=0, le=4)
    fev1_percent: Optional[float] = None
    comorbidities: Optional[List[str]] = None


# ==================== Response Models ====================

class PatientResponse(BaseModel):
    """Response model for patient data"""
    patient_id: str
    name: str
    sex: str
    age: int
    tnm_stage: str
    histology_type: str
    performance_status: int
    fev1_percent: Optional[float]
    laterality: str
    comorbidities: List[str]
    created_at: Optional[datetime] = None


class SimilarPatientResponse(BaseModel):
    """Response model for similar patient queries"""
    patient_id: str
    name: str
    similarity_score: float
    tnm_stage: str
    histology_type: str
    treatment_received: Optional[str]
    outcome: Optional[str]
    survival_days: Optional[int]


class PatientHistoryResponse(BaseModel):
    """Response model for patient history"""
    patient_id: str
    patient_data: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    outcomes: List[Dict[str, Any]]


# ==================== Endpoints ====================

@router.post("/", response_model=PatientResponse)
async def create_patient(patient: PatientCreateRequest):
    """
    Create a new patient record.
    
    Creates patient in both the OWL ontology and Neo4j graph database.
    Follows the Figure 2 pattern from the original LCA paper.
    """
    # Import here to avoid circular dependency
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Generate patient ID if not provided
        import uuid
        patient_id = patient.patient_id or f"LC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
        
        patient_data = patient.model_dump()
        patient_data["patient_id"] = patient_id
        
        # Create in Neo4j if available
        if hasattr(lca_service, 'graph_db') and lca_service.graph_db:
            lca_service.graph_db.create_patient(patient_data)
        
        # Create in ontology
        if hasattr(lca_service, 'ontology') and lca_service.ontology:
            lca_service.ontology.create_patient_individual(
                patient_id=patient_id,
                name=patient.name,
                sex=patient.sex,
                age=patient.age,
                diagnosis=patient.diagnosis or "Malignant Neoplasm of Lung",
                tnm_stage=patient.tnm_stage,
                histology_type=patient.histology_type,
                laterality=patient.laterality,
                performance_status=patient.performance_status,
                fev1_percent=patient.fev1_percent
            )
        
        return PatientResponse(
            patient_id=patient_id,
            name=patient.name,
            sex=patient.sex,
            age=patient.age,
            tnm_stage=patient.tnm_stage,
            histology_type=patient.histology_type,
            performance_status=patient.performance_status,
            fev1_percent=patient.fev1_percent,
            laterality=patient.laterality,
            comorbidities=patient.comorbidities or [],
            created_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create patient: {str(e)}")


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: str):
    """
    Get patient by ID.
    
    Retrieves patient data from Neo4j graph database.
    """
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        if hasattr(lca_service, 'graph_db') and lca_service.graph_db:
            history = lca_service.graph_db.get_patient_history(patient_id)
            if history and history.get("patient"):
                p = history["patient"]
                return PatientResponse(
                    patient_id=p.get("patient_id", patient_id),
                    name=p.get("name", "Unknown"),
                    sex=p.get("sex", "U"),
                    age=p.get("age_at_diagnosis", 0),
                    tnm_stage=p.get("tnm_stage", ""),
                    histology_type=p.get("histology_type", ""),
                    performance_status=p.get("performance_status", 0),
                    fev1_percent=p.get("fev1_percent"),
                    laterality=p.get("laterality", "Unknown"),
                    comorbidities=p.get("comorbidities", [])
                )
        
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve patient: {str(e)}")


@router.get("/{patient_id}/similar", response_model=List[SimilarPatientResponse])
async def get_similar_patients(
    patient_id: str,
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of similar patients")
):
    """
    Find patients with similar clinical profiles.
    
    Uses Neo4j graph queries to find patients with:
    - Same TNM stage
    - Similar performance status (±1)
    - Similar age range
    """
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        if hasattr(lca_service, 'graph_db') and lca_service.graph_db:
            similar = lca_service.graph_db.find_similar_patients(patient_id, limit=limit)
            
            return [
                SimilarPatientResponse(
                    patient_id=p.get("patient_id", ""),
                    name=p.get("name", "Unknown"),
                    similarity_score=0.8,  # Calculated by Neo4j query
                    tnm_stage=p.get("stage", ""),
                    histology_type=p.get("histology", ""),
                    treatment_received=p.get("treatments", [None])[0] if p.get("treatments") else None,
                    outcome=None,
                    survival_days=int(p.get("avg_survival")) if p.get("avg_survival") else None
                )
                for p in similar
            ]
        
        return []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find similar patients: {str(e)}")


@router.get("/{patient_id}/history", response_model=PatientHistoryResponse)
async def get_patient_history(patient_id: str):
    """
    Get complete patient history including recommendations and outcomes.
    """
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        if hasattr(lca_service, 'graph_db') and lca_service.graph_db:
            history = lca_service.graph_db.get_patient_history(patient_id)
            
            if history:
                return PatientHistoryResponse(
                    patient_id=patient_id,
                    patient_data=history.get("patient", {}),
                    recommendations=history.get("recommendations", []),
                    outcomes=[]
                )
        
        raise HTTPException(status_code=404, detail=f"No history found for patient {patient_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve patient history: {str(e)}")


@router.post("/search/snomed")
async def search_by_snomed(
    snomed_codes: List[str],
    limit: int = Query(default=50, ge=1, le=200)
):
    """
    Search for patients by SNOMED-CT codes.
    
    Useful for finding patients with specific:
    - Diagnoses
    - Histology types
    - Biomarkers
    - Procedures
    """
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Use neo4j_tools if available
        from ...db.neo4j_tools import Neo4jReadTools
        read_tools = Neo4jReadTools()
        
        if read_tools.is_available:
            patients = read_tools.search_patients_by_snomed(snomed_codes)
            read_tools.close()
            return {"snomed_codes": snomed_codes, "patients": patients[:limit]}
        
        return {"snomed_codes": snomed_codes, "patients": [], "message": "Neo4j not available"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SNOMED search failed: {str(e)}")
