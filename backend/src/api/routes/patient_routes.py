"""
Patient Management API Routes
Provides CRUD operations for patient data with Neo4j persistence
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import os

from ...db.neo4j_tools import Neo4jTools

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/patients", tags=["Patients"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class Demographics(BaseModel):
    """Patient demographic information"""
    age: int = Field(..., ge=0, le=120)
    sex: str = Field(..., pattern="^(M|F|Other)$")
    ethnicity: Optional[str] = None


class ClinicalData(BaseModel):
    """Clinical patient data"""
    tnm_stage: str
    histology_type: str
    performance_status: int = Field(..., ge=0, le=4)
    fev1_percent: Optional[float] = None
    laterality: Optional[str] = None
    diagnosis: Optional[str] = None


class Biomarkers(BaseModel):
    """Biomarker test results"""
    egfr_mutation: Optional[str] = None
    egfr_mutation_type: Optional[str] = None
    alk_rearrangement: Optional[bool] = None
    ros1_rearrangement: Optional[bool] = None
    braf_mutation: Optional[str] = None
    met_exon14: Optional[bool] = None
    ret_rearrangement: Optional[bool] = None
    kras_mutation: Optional[str] = None
    pdl1_tps: Optional[float] = None
    tmb_score: Optional[float] = None


class PatientCreate(BaseModel):
    """Request model for creating a new patient"""
    patient_id: str
    name: Optional[str] = None
    demographics: Demographics
    clinical_data: ClinicalData
    biomarkers: Optional[Biomarkers] = None
    comorbidities: List[str] = []


class PatientUpdate(BaseModel):
    """Request model for updating patient data"""
    demographics: Optional[Demographics] = None
    clinical_data: Optional[ClinicalData] = None
    biomarkers: Optional[Biomarkers] = None
    comorbidities: Optional[List[str]] = None


class PatientResponse(BaseModel):
    """Response model for patient data"""
    patient_id: str
    name: Optional[str] = None
    demographics: Demographics
    clinical_data: ClinicalData
    biomarkers: Optional[Biomarkers] = None
    comorbidities: List[str] = []
    created_at: str
    updated_at: str
    neo4j_node_id: Optional[int] = None


class PatientListResponse(BaseModel):
    """Response model for listing patients"""
    total: int
    patients: List[PatientResponse]
    page: int
    page_size: int


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_neo4j_tools() -> Neo4jTools:
    """Initialize Neo4j connection"""
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        raise HTTPException(
            status_code=503,
            detail="Neo4j not configured - set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD"
        )
    
    return Neo4jTools(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)


# ============================================================================
# CRUD ENDPOINTS
# ============================================================================

@router.post("", response_model=PatientResponse, status_code=201)
async def create_patient(patient: PatientCreate):
    """
    Create a new patient record in Neo4j.
    
    Args:
        patient: Patient data including demographics, clinical data, and biomarkers
    
    Returns:
        Created patient with Neo4j node ID and timestamps
    """
    try:
        neo4j = get_neo4j_tools()
        
        # Create patient node
        query = """
        CREATE (p:Patient {
            patient_id: $patient_id,
            name: $name,
            age: $age,
            sex: $sex,
            ethnicity: $ethnicity,
            tnm_stage: $tnm_stage,
            histology_type: $histology_type,
            performance_status: $performance_status,
            fev1_percent: $fev1_percent,
            laterality: $laterality,
            diagnosis: $diagnosis,
            comorbidities: $comorbidities,
            created_at: datetime(),
            updated_at: datetime()
        })
        RETURN id(p) as node_id, p
        """
        
        params = {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "age": patient.demographics.age,
            "sex": patient.demographics.sex,
            "ethnicity": patient.demographics.ethnicity,
            "tnm_stage": patient.clinical_data.tnm_stage,
            "histology_type": patient.clinical_data.histology_type,
            "performance_status": patient.clinical_data.performance_status,
            "fev1_percent": patient.clinical_data.fev1_percent,
            "laterality": patient.clinical_data.laterality,
            "diagnosis": patient.clinical_data.diagnosis,
            "comorbidities": patient.comorbidities
        }
        
        result = neo4j.execute_query(query, params)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create patient")
        
        node = result[0]
        
        # Add biomarkers if provided
        if patient.biomarkers:
            biomarker_query = """
            MATCH (p:Patient {patient_id: $patient_id})
            CREATE (b:Biomarker {
                egfr_mutation: $egfr_mutation,
                alk_rearrangement: $alk_rearrangement,
                pdl1_tps: $pdl1_tps,
                created_at: datetime()
            })
            CREATE (p)-[:HAS_BIOMARKER]->(b)
            """
            
            neo4j.execute_query(biomarker_query, {
                "patient_id": patient.patient_id,
                "egfr_mutation": patient.biomarkers.egfr_mutation,
                "alk_rearrangement": patient.biomarkers.alk_rearrangement,
                "pdl1_tps": patient.biomarkers.pdl1_tps
            })
        
        neo4j.close()
        
        return PatientResponse(
            patient_id=patient.patient_id,
            name=patient.name,
            demographics=patient.demographics,
            clinical_data=patient.clinical_data,
            biomarkers=patient.biomarkers,
            comorbidities=patient.comorbidities,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            neo4j_node_id=node.get("node_id")
        )
        
    except Exception as e:
        logger.error(f"Failed to create patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: str):
    """
    Retrieve a patient by ID.
    
    Args:
        patient_id: Unique patient identifier
    
    Returns:
        Patient data with all clinical information
    """
    try:
        neo4j = get_neo4j_tools()
        
        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        OPTIONAL MATCH (p)-[:HAS_BIOMARKER]->(b:Biomarker)
        RETURN id(p) as node_id, p, b
        """
        
        result = neo4j.execute_query(query, {"patient_id": patient_id})
        neo4j.close()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        node = result[0]
        p = node.get("p")
        b = node.get("b")
        
        return PatientResponse(
            patient_id=p["patient_id"],
            name=p.get("name"),
            demographics=Demographics(
                age=p["age"],
                sex=p["sex"],
                ethnicity=p.get("ethnicity")
            ),
            clinical_data=ClinicalData(
                tnm_stage=p["tnm_stage"],
                histology_type=p["histology_type"],
                performance_status=p["performance_status"],
                fev1_percent=p.get("fev1_percent"),
                laterality=p.get("laterality"),
                diagnosis=p.get("diagnosis")
            ),
            biomarkers=Biomarkers(
                egfr_mutation=b.get("egfr_mutation") if b else None,
                alk_rearrangement=b.get("alk_rearrangement") if b else None,
                pdl1_tps=b.get("pdl1_tps") if b else None
            ) if b else None,
            comorbidities=p.get("comorbidities", []),
            created_at=p.get("created_at", datetime.now()).isoformat(),
            updated_at=p.get("updated_at", datetime.now()).isoformat(),
            neo4j_node_id=node.get("node_id")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(patient_id: str, updates: PatientUpdate):
    """
    Update an existing patient record.
    
    Args:
        patient_id: Unique patient identifier
        updates: Fields to update
    
    Returns:
        Updated patient data
    """
    try:
        neo4j = get_neo4j_tools()
        
        # Build dynamic update query
        set_clauses = ["p.updated_at = datetime()"]
        params = {"patient_id": patient_id}
        
        if updates.demographics:
            set_clauses.append("p.age = $age")
            set_clauses.append("p.sex = $sex")
            params["age"] = updates.demographics.age
            params["sex"] = updates.demographics.sex
            if updates.demographics.ethnicity:
                set_clauses.append("p.ethnicity = $ethnicity")
                params["ethnicity"] = updates.demographics.ethnicity
        
        if updates.clinical_data:
            set_clauses.append("p.tnm_stage = $tnm_stage")
            set_clauses.append("p.performance_status = $performance_status")
            params["tnm_stage"] = updates.clinical_data.tnm_stage
            params["performance_status"] = updates.clinical_data.performance_status
        
        if updates.comorbidities is not None:
            set_clauses.append("p.comorbidities = $comorbidities")
            params["comorbidities"] = updates.comorbidities
        
        query = f"""
        MATCH (p:Patient {{patient_id: $patient_id}})
        SET {', '.join(set_clauses)}
        RETURN id(p) as node_id, p
        """
        
        result = neo4j.execute_query(query, params)
        neo4j.close()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        # Fetch updated patient
        return await get_patient(patient_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{patient_id}", status_code=204)
async def delete_patient(patient_id: str):
    """
    Delete a patient record.
    
    Args:
        patient_id: Unique patient identifier
    """
    try:
        neo4j = get_neo4j_tools()
        
        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        DETACH DELETE p
        """
        
        neo4j.execute_query(query, {"patient_id": patient_id})
        neo4j.close()
        
    except Exception as e:
        logger.error(f"Failed to delete patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=PatientListResponse)
async def list_patients(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    stage: Optional[str] = None
):
    """
    List all patients with pagination and optional filtering.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of patients per page
        stage: Optional TNM stage filter
    
    Returns:
        Paginated list of patients
    """
    try:
        neo4j = get_neo4j_tools()
        
        # Build query with optional filter
        where_clause = ""
        params: Dict[str, Any] = {}
        
        if stage:
            where_clause = "WHERE p.tnm_stage = $stage"
            params["stage"] = stage
        
        # Count total
        count_query = f"""
        MATCH (p:Patient)
        {where_clause}
        RETURN count(p) as total
        """
        
        count_result = neo4j.execute_query(count_query, params)
        total = count_result[0]["total"] if count_result else 0
        
        # Get paginated results
        skip = (page - 1) * page_size
        params["skip"] = skip
        params["limit"] = page_size
        
        query = f"""
        MATCH (p:Patient)
        {where_clause}
        RETURN id(p) as node_id, p
        ORDER BY p.created_at DESC
        SKIP $skip
        LIMIT $limit
        """
        
        result = neo4j.execute_query(query, params)
        neo4j.close()
        
        patients = [
            PatientResponse(
                patient_id=node["p"]["patient_id"],
                name=node["p"].get("name"),
                demographics=Demographics(
                    age=node["p"]["age"],
                    sex=node["p"]["sex"],
                    ethnicity=node["p"].get("ethnicity")
                ),
                clinical_data=ClinicalData(
                    tnm_stage=node["p"]["tnm_stage"],
                    histology_type=node["p"]["histology_type"],
                    performance_status=node["p"]["performance_status"],
                    fev1_percent=node["p"].get("fev1_percent"),
                    laterality=node["p"].get("laterality"),
                    diagnosis=node["p"].get("diagnosis")
                ),
                comorbidities=node["p"].get("comorbidities", []),
                created_at=node["p"].get("created_at", datetime.now()).isoformat(),
                updated_at=node["p"].get("updated_at", datetime.now()).isoformat(),
                neo4j_node_id=node.get("node_id")
            )
            for node in result
        ]
        
        return PatientListResponse(
            total=total,
            patients=patients,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to list patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))
