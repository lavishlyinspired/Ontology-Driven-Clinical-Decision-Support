"""
FastAPI REST API for Lung Cancer Assistant
Provides RESTful endpoints for clinical decision support

6-Agent Architecture per final.md specification:
1. IngestionAgent: Validates and normalizes raw patient data
2. SemanticMappingAgent: Maps clinical concepts to SNOMED-CT codes
3. ClassificationAgent: Applies LUCADA ontology and NICE guidelines
4. ConflictResolutionAgent: Resolves conflicting recommendations
5. PersistenceAgent: THE ONLY AGENT THAT WRITES TO NEO4J
6. ExplanationAgent: Generates MDT summaries
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.lca_service import LungCancerAssistantService, TreatmentRecommendation
from src.agents.lca_workflow import LCAWorkflow, analyze_patient as workflow_analyze
from src.db.models import DecisionSupportResponse as WorkflowResponse

# Import route modules
from src.api.routes import (
    patients_router, 
    treatments_router, 
    guidelines_router,
    analytics_router,
    audit_router,
    biomarkers_router
)

# Initialize FastAPI
app = FastAPI(
    title="Lung Cancer Assistant API",
    description="Ontology-driven clinical decision support for lung cancer treatment",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include modular routes
app.include_router(patients_router, prefix="/api/v1")
app.include_router(treatments_router, prefix="/api/v1")
app.include_router(guidelines_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")
app.include_router(audit_router, prefix="/api/v1")
app.include_router(biomarkers_router, prefix="/api/v1")

# Global service instance
lca_service: Optional[LungCancerAssistantService] = None


# ==================== Pydantic Models ====================

class PatientInput(BaseModel):
    """Patient input for analysis"""
    patient_id: Optional[str] = None
    name: str = Field(..., description="Patient name")
    sex: str = Field(..., pattern="^[MFU]$", description="Sex: M/F/U")
    age: int = Field(..., ge=0, le=120, description="Age at diagnosis")
    tnm_stage: str = Field(..., description="TNM stage (IA, IB, IIA, IIB, IIIA, IIIB, IV)")
    histology_type: str = Field(..., description="Histology type")
    performance_status: int = Field(..., ge=0, le=4, description="WHO Performance Status (0-4)")
    fev1_percent: Optional[float] = Field(None, description="FEV1 percentage")
    laterality: str = Field(default="Unknown", description="Tumor laterality")
    comorbidities: Optional[List[str]] = Field(default=[], description="List of comorbidities")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "sex": "M",
                "age": 68,
                "tnm_stage": "IIIA",
                "histology_type": "Adenocarcinoma",
                "performance_status": 1,
                "fev1_percent": 65.0,
                "laterality": "Right",
                "comorbidities": ["COPD", "Hypertension"]
            }
        }


class TreatmentRecommendationResponse(BaseModel):
    """Treatment recommendation response"""
    treatment_type: str
    rule_id: str
    rule_source: str
    evidence_level: str
    treatment_intent: str
    survival_benefit: Optional[str]
    contraindications: List[str]
    priority: int
    confidence_score: float


class DecisionSupportResponse(BaseModel):
    """Complete decision support response"""
    patient_id: str
    timestamp: datetime
    patient_scenarios: List[str]
    recommendations: List[TreatmentRecommendationResponse]
    mdt_summary: str
    similar_patients_count: int
    semantic_guidelines_count: int


class GuidelineRuleResponse(BaseModel):
    """Guideline rule information"""
    rule_id: str
    name: str
    source: str
    description: str
    recommended_treatment: str
    treatment_intent: str
    evidence_level: str
    survival_benefit: Optional[str]


class SystemStatsResponse(BaseModel):
    """System statistics"""
    ontology: Dict[str, Any]
    guidelines: Dict[str, Any]
    neo4j: Dict[str, Any]
    vector_store: Dict[str, Any]


# ==================== Lifecycle Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize LCA service on startup"""
    global lca_service

    print("=" * 80)
    print("Starting Lung Cancer Assistant API")
    print("=" * 80)

    lca_service = LungCancerAssistantService(
        use_neo4j=False,  # Set to True if Neo4j is running
        use_vector_store=True
    )

    print("✓ LCA Service initialized")
    print("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global lca_service

    if lca_service:
        lca_service.close()
        print("✓ LCA Service shut down")


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Lung Cancer Assistant API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "analyze_patient": "POST /api/v1/patients/analyze",
            "list_guidelines": "GET /api/v1/guidelines",
            "system_stats": "GET /api/v1/system/stats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "operational"
    }


@app.post("/api/v1/patients/analyze", response_model=DecisionSupportResponse)
async def analyze_patient(
    patient: PatientInput,
    use_ai_workflow: bool = True,
    background_tasks: BackgroundTasks = None
):
    """
    Analyze a patient and generate treatment recommendations.

    Args:
        patient: Patient clinical data
        use_ai_workflow: Whether to run AI agent workflow (takes ~20s)

    Returns:
        Complete decision support with recommendations and MDT summary
    """
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Process patient
        result = await lca_service.process_patient(
            patient.model_dump(),
            use_ai_workflow=use_ai_workflow
        )

        # Convert to response format
        recommendations_response = [
            TreatmentRecommendationResponse(
                treatment_type=rec.treatment_type,
                rule_id=rec.rule_id,
                rule_source=rec.rule_source,
                evidence_level=rec.evidence_level,
                treatment_intent=rec.treatment_intent,
                survival_benefit=rec.survival_benefit,
                contraindications=rec.contraindications,
                priority=rec.priority,
                confidence_score=rec.confidence_score
            )
            for rec in result.recommendations
        ]

        return DecisionSupportResponse(
            patient_id=result.patient_id,
            timestamp=result.timestamp,
            patient_scenarios=result.patient_scenarios,
            recommendations=recommendations_response,
            mdt_summary=result.mdt_summary,
            similar_patients_count=len(result.similar_patients),
            semantic_guidelines_count=len(result.semantic_guidelines)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Patient processing failed: {str(e)}"
        )


@app.get("/api/v1/guidelines", response_model=List[GuidelineRuleResponse])
async def list_guidelines():
    """
    List all available clinical guideline rules.

    Returns:
        List of all NICE guideline rules
    """
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        rules = lca_service.rule_engine.get_all_rules()

        return [
            GuidelineRuleResponse(
                rule_id=rule.rule_id,
                name=rule.name,
                source=rule.source,
                description=rule.description,
                recommended_treatment=rule.recommended_treatment,
                treatment_intent=rule.treatment_intent,
                evidence_level=rule.evidence_level,
                survival_benefit=rule.survival_benefit
            )
            for rule in rules
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve guidelines: {str(e)}"
        )


@app.get("/api/v1/guidelines/{rule_id}", response_model=GuidelineRuleResponse)
async def get_guideline(rule_id: str):
    """
    Get a specific guideline rule by ID.

    Args:
        rule_id: Guideline rule ID (e.g., "R1", "R2")

    Returns:
        Guideline rule details
    """
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        rule = lca_service.rule_engine.get_rule_by_id(rule_id)

        if rule is None:
            raise HTTPException(status_code=404, detail=f"Guideline {rule_id} not found")

        return GuidelineRuleResponse(
            rule_id=rule.rule_id,
            name=rule.name,
            source=rule.source,
            description=rule.description,
            recommended_treatment=rule.recommended_treatment,
            treatment_intent=rule.treatment_intent,
            evidence_level=rule.evidence_level,
            survival_benefit=rule.survival_benefit
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve guideline: {str(e)}"
        )


@app.get("/api/v1/system/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """
    Get system statistics and configuration.

    Returns:
        System statistics including ontology, guidelines, databases
    """
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        stats = lca_service.get_system_stats()
        return SystemStatsResponse(**stats)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stats: {str(e)}"
        )


@app.post("/api/v1/guidelines/search")
async def search_guidelines(query: str, limit: int = 5):
    """
    Semantic search for guidelines using vector store.

    Args:
        query: Natural language query
        limit: Maximum number of results

    Returns:
        Matching guidelines with similarity scores
    """
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if lca_service.vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not available")

    try:
        results = lca_service.vector_store.search_guidelines(query, n_results=limit)

        return {
            "query": query,
            "results": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# ==================== NEW V2 API: 6-Agent Workflow ====================

# Global workflow instance
lca_workflow: Optional[LCAWorkflow] = None


class PatientInputV2(BaseModel):
    """Patient input for V2 analysis (6-agent workflow)"""
    patient_id: Optional[str] = Field(None, description="Patient ID (auto-generated if not provided)")
    sex: Optional[str] = Field(None, pattern="^(Male|Female|Unknown|M|F|U)$", description="Sex")
    age_at_diagnosis: Optional[int] = Field(None, ge=0, le=120, description="Age at diagnosis")
    tnm_stage: str = Field(..., description="TNM stage (IA, IB, IIA, IIB, IIIA, IIIB, IV)")
    histology_type: str = Field(..., description="Histology type")
    performance_status: Optional[int] = Field(None, ge=0, le=4, description="ECOG Performance Status (0-4)")
    laterality: Optional[str] = Field(None, description="Tumor laterality (Left, Right, Bilateral)")
    diagnosis: Optional[str] = Field(None, description="Primary diagnosis")
    comorbidities: Optional[List[str]] = Field(default=[], description="List of comorbidities")

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "LC-2024-001",
                "sex": "Male",
                "age_at_diagnosis": 68,
                "tnm_stage": "IIIA",
                "histology_type": "Adenocarcinoma",
                "performance_status": 1,
                "laterality": "Right",
                "diagnosis": "Malignant Neoplasm of Lung"
            }
        }


class WorkflowResponseV2(BaseModel):
    """Response from 6-agent workflow"""
    patient_id: Optional[str]
    success: bool
    workflow_status: str
    agent_chain: List[str]
    
    # Classification
    scenario: Optional[str]
    scenario_confidence: Optional[float]
    recommendations: List[Dict[str, Any]]
    reasoning_chain: List[str]
    
    # SNOMED mappings
    snomed_mappings: Dict[str, Optional[str]]
    mapping_confidence: float
    
    # Persistence
    inference_id: Optional[str]
    persisted: bool
    
    # MDT summary
    mdt_summary: Optional[str]
    key_considerations: List[str]
    discussion_points: List[str]
    
    # Metadata
    processing_time_seconds: float
    errors: List[str]
    guideline_refs: List[str]


@app.post("/api/v2/patients/analyze", response_model=WorkflowResponseV2)
async def analyze_patient_v2(
    patient: PatientInputV2,
    persist_to_neo4j: bool = False
):
    """
    Analyze a patient using the new 6-agent workflow.
    
    This endpoint uses the production-ready 6-agent architecture:
    1. IngestionAgent: Validates and normalizes raw patient data
    2. SemanticMappingAgent: Maps clinical concepts to SNOMED-CT codes
    3. ClassificationAgent: Applies LUCADA ontology and NICE guidelines
    4. ConflictResolutionAgent: Resolves conflicting recommendations
    5. PersistenceAgent: Saves to Neo4j (if enabled)
    6. ExplanationAgent: Generates MDT summaries
    
    CRITICAL: Only PersistenceAgent writes to Neo4j.

    Args:
        patient: Patient clinical data
        persist_to_neo4j: Whether to save results to Neo4j

    Returns:
        Complete decision support with SNOMED mappings and MDT summary
    """
    global lca_workflow
    
    try:
        # Initialize workflow if needed
        if lca_workflow is None:
            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USER") 
            neo4j_password = os.getenv("NEO4J_PASSWORD")
            
            lca_workflow = LCAWorkflow(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                persist_results=persist_to_neo4j
            )
        
        # Run workflow
        result = lca_workflow.run(patient.model_dump())
        
        return WorkflowResponseV2(
            patient_id=result.patient_id,
            success=result.success,
            workflow_status=result.workflow_status,
            agent_chain=result.agent_chain,
            scenario=result.scenario,
            scenario_confidence=result.scenario_confidence,
            recommendations=result.recommendations,
            reasoning_chain=result.reasoning_chain,
            snomed_mappings=result.snomed_mappings,
            mapping_confidence=result.mapping_confidence,
            inference_id=result.inference_id,
            persisted=result.persisted,
            mdt_summary=result.mdt_summary,
            key_considerations=result.key_considerations,
            discussion_points=result.discussion_points,
            processing_time_seconds=result.processing_time_seconds,
            errors=result.errors,
            guideline_refs=result.guideline_refs
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow processing failed: {str(e)}"
        )


@app.get("/api/v2/workflow/info")
async def get_workflow_info():
    """
    Get information about the 6-agent workflow.
    
    Returns:
        Workflow architecture and agent descriptions
    """
    return {
        "version": "2.0.0",
        "architecture": "6-Agent LangGraph Workflow",
        "principle": "Neo4j as a tool, not a brain",
        "agents": [
            {
                "name": "IngestionAgent",
                "order": 1,
                "role": "Validates and normalizes raw patient data",
                "neo4j_access": "READ-ONLY"
            },
            {
                "name": "SemanticMappingAgent",
                "order": 2,
                "role": "Maps clinical concepts to SNOMED-CT codes",
                "neo4j_access": "READ-ONLY"
            },
            {
                "name": "ClassificationAgent",
                "order": 3,
                "role": "Applies LUCADA ontology and NICE guidelines",
                "neo4j_access": "READ-ONLY"
            },
            {
                "name": "ConflictResolutionAgent",
                "order": 4,
                "role": "Resolves conflicting recommendations",
                "neo4j_access": "READ-ONLY"
            },
            {
                "name": "PersistenceAgent",
                "order": 5,
                "role": "THE ONLY AGENT THAT WRITES TO NEO4J",
                "neo4j_access": "WRITE"
            },
            {
                "name": "ExplanationAgent",
                "order": 6,
                "role": "Generates MDT summaries",
                "neo4j_access": "READ-ONLY"
            }
        ],
        "data_flow": "Input → Ingestion → SemanticMapping → Classification → ConflictResolution → Persistence → Explanation → Output"
    }


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 80)
    print("LUNG CANCER ASSISTANT - REST API")
    print("=" * 80)
    print("\nStarting server...")
    print("  URL: http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("  ReDoc: http://localhost:8000/redoc")
    print("\n" + "=" * 80)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
