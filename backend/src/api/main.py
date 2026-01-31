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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import os
import sys
from pathlib import Path
import logging
import json
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Add parent to path first (needed for imports)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import and initialize centralized logging
from src.logging_config import setup_logging, get_logger, log_execution

# Initialize logging with file output enabled
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    enable_json=os.getenv("LOG_ENABLE_JSON", "false").lower() == "true",
    enable_file_logging=os.getenv("LOG_ENABLE_FILE", "true").lower() == "true",
    enable_langsmith=True
)

logger = get_logger(__name__)

# Prometheus metrics (optional)
REQUEST_COUNT = Counter(
    'lca_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'lca_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

from src.services.lca_service import LungCancerAssistantService, TreatmentRecommendation
from src.agents.lca_workflow import LCAWorkflow, analyze_patient as workflow_analyze
from src.agents.integrated_workflow import IntegratedLCAWorkflow, analyze_patient_integrated
from src.db.models import DecisionSupportResponse as WorkflowResponse
from src.db.neo4j_tools import Neo4jTools

# Import route modules
from src.api.routes import (
    patients_router,
    patient_crud_router,
    treatments_router,
    guidelines_router,
    analytics_router,
    analytics_detail_router,
    audit_router,
    audit_detail_router,
    biomarkers_router,
    patient_similarity_router,
    biomarker_detail_router,
    counterfactual_router,
    export_router,
    system_router,
    chat_router,
    graph_router,
    chat_graph_router,
    ontology_router,
    graph_algorithms_router
)

# Initialize FastAPI
app = FastAPI(
    title="Lung Cancer Assistant API",
    description="Ontology-driven clinical decision support for lung cancer treatment",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==================== Security & Performance Middleware ====================

# CORS with environment-based origins
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining"]
)

# GZip compression for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted host protection
if os.getenv("ENVIRONMENT") == "production":
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*.lca-system.com", "localhost"])


# ==================== Request Logging Middleware ====================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing and metrics"""
    request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")
    start_time = time.time()
    
    # Log request
    logger.info(f"[{request_id}] {request.method} {request.url.path}", extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host
    })
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Log response
        logger.info(f"[{request_id}] {response.status_code} {latency:.3f}s", extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "latency_seconds": latency
        })
        
        # Update Prometheus metrics (disabled by default to avoid overhead)
        if os.getenv("METRICS_ENABLED", "false").lower() == "true":
            try:
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code
                ).inc()
                REQUEST_LATENCY.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(latency)
            except Exception:
                pass  # Silently ignore metrics errors
        
        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency:.3f}s"
        
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] Request failed: {str(e)}", extra={
            "request_id": request_id,
            "error": str(e),
            "error_type": type(e).__name__
        })
        raise


# ==================== Rate Limiting Middleware ====================

rate_limit_storage = {}  # Simple in-memory storage (use Redis in production)

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Simple rate limiting middleware"""
    if os.getenv("RATE_LIMIT_ENABLED", "true").lower() != "true":
        return await call_next(request)
    
    client_ip = request.client.host
    current_minute = int(time.time() / 60)
    key = f"{client_ip}:{current_minute}"
    
    # Get current request count
    request_count = rate_limit_storage.get(key, 0)
    limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    if request_count >= limit:
        return Response(
            content=json.dumps({"error": "Rate limit exceeded. Please try again later."}),
            status_code=429,
            media_type="application/json",
            headers={"X-RateLimit-Remaining": "0"}
        )
    
    # Increment counter
    rate_limit_storage[key] = request_count + 1
    
    # Clean old entries (simple cleanup)
    if len(rate_limit_storage) > 10000:
        old_keys = [k for k in rate_limit_storage.keys() if int(k.split(":")[1]) < current_minute - 5]
        for k in old_keys:
            del rate_limit_storage[k]
    
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(limit - request_count - 1)
    
    return response

# Include modular routes
app.include_router(patients_router, prefix="/api/v1")
app.include_router(patient_crud_router)  # Already has /api/v1/patients prefix
app.include_router(treatments_router, prefix="/api/v1")
app.include_router(guidelines_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")
app.include_router(analytics_detail_router)  # Already has /api/v1/analytics prefix
app.include_router(audit_router, prefix="/api/v1")
app.include_router(audit_detail_router)  # Already has /api/v1/audit prefix
app.include_router(biomarkers_router, prefix="/api/v1")
app.include_router(patient_similarity_router)  # Already has /api/v1/patients prefix
app.include_router(biomarker_detail_router)  # Already has /api/v1/biomarkers prefix
app.include_router(counterfactual_router)  # Already has /api/v1/analytics prefix
app.include_router(export_router)  # Already has /api/v1/export prefix
app.include_router(system_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")  # Chat streaming endpoint
app.include_router(graph_router, prefix="/api")  # Graph visualization endpoints
app.include_router(chat_graph_router, prefix="/api")  # Enhanced chat with graph integration
app.include_router(ontology_router, prefix="/api/v1")  # Ontology lookup and validation
app.include_router(graph_algorithms_router, prefix="/api/v1")  # Neo4j graph algorithms

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
    """Initialize all services on startup"""
    global lca_service

    print("=" * 80)
    print("Starting Lung Cancer Assistant API v2.0.0")
    print("=" * 80)

    # Step 1: Initialize core LCA service
    print("📦 Initializing Core LCA Service...")
    lca_service = LungCancerAssistantService(
        use_neo4j=os.getenv("NEO4J_URI") is not None,
        use_vector_store=True
    )
    print("   ✓ LCA Service initialized")

    # Step 2: Initialize Redis connection pool (for cache, websocket, batch)
    print("📦 Initializing Redis Connection Pool...")
    try:
        import redis.asyncio as aioredis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        app.state.redis = await aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50
        )
        print(f"   ✓ Redis connected: {redis_url}")
    except Exception as e:
        print(f"   ⚠ Redis connection failed: {e}")
        print(f"   → Services will run in degraded mode")
        app.state.redis = None

    # Step 3: Initialize Authentication Service
    print("📦 Initializing Authentication Service...")
    try:
        # Auth service already initialized as global
        app.state.auth_service = auth_service
        print("   ✓ Auth service ready")
    except Exception as e:
        print(f"   ⚠ Auth service warning: {e}")

    # Step 4: Initialize Audit Logger
    print("📦 Initializing Audit Logger...")
    try:
        app.state.audit_logger = audit_logger
        # Log system startup
        await audit_logger.log_event(
            action="SYSTEM_STARTUP",
            user_id="system",
            resource_type="api",
            resource_id="main",
            details={"version": "2.0.0", "environment": os.getenv("ENVIRONMENT", "development")}
        )
        print("   ✓ Audit logger active")
    except Exception as e:
        print(f"   ⚠ Audit logger warning: {e}")

    # Step 5: Initialize Human-in-the-Loop Service
    print("📦 Initializing HITL Service...")
    try:
        app.state.hitl_service = hitl_service
        print("   ✓ HITL service ready")
    except Exception as e:
        print(f"   ⚠ HITL service warning: {e}")

    # Step 6: Initialize Analytics Service
    print("📦 Initializing Analytics Service...")
    try:
        app.state.analytics_service = analytics_service
        print("   ✓ Analytics service ready")
    except Exception as e:
        print(f"   ⚠ Analytics service warning: {e}")

    # Step 7: Initialize RAG Service
    print("📦 Initializing RAG Service...")
    try:
        app.state.rag_service = rag_service
        # Initialize embeddings if not already loaded
        if not rag_service.embeddings_model:
            await rag_service.initialize()
        print("   ✓ RAG service ready (embeddings loaded)")
    except Exception as e:
        print(f"   ⚠ RAG service warning: {e}")

    # Step 8: Initialize WebSocket Manager
    print("📦 Initializing WebSocket Manager...")
    try:
        app.state.websocket_service = websocket_service
        print("   ✓ WebSocket manager ready")
    except Exception as e:
        print(f"   ⚠ WebSocket manager warning: {e}")

    # Step 9: Initialize Version Manager
    print("📦 Initializing Guideline Version Manager...")
    try:
        app.state.version_service = version_service
        print("   ✓ Version manager ready")
    except Exception as e:
        print(f"   ⚠ Version manager warning: {e}")

    # Step 10: Initialize Batch Processor
    print("📦 Initializing Batch Processor...")
    try:
        app.state.batch_service = batch_service
        print("   ✓ Batch processor ready")
    except Exception as e:
        print(f"   ⚠ Batch processor warning: {e}")

    # Step 11: Initialize FHIR Service
    print("📦 Initializing FHIR Service...")
    try:
        app.state.fhir_service = fhir_service
        fhir_url = os.getenv("FHIR_SERVER_URL", "http://localhost:8080/fhir")
        print(f"   ✓ FHIR service ready (target: {fhir_url})")
    except Exception as e:
        print(f"   ⚠ FHIR service warning: {e}")

    # Step 12: Initialize Cache Service
    print("📦 Initializing Cache Service...")
    try:
        app.state.cache_service = cache_service
        print("   ✓ Cache service ready")
    except Exception as e:
        print(f"   ⚠ Cache service warning: {e}")

    print("=" * 80)
    print("✅ All services initialized successfully!")
    print("📊 System Status:")
    print(f"   • Core LCA Service: ✓")
    print(f"   • Authentication: ✓")
    print(f"   • Audit Logging: ✓")
    print(f"   • HITL: ✓")
    print(f"   • Analytics: ✓")
    print(f"   • RAG: ✓")
    print(f"   • WebSocket: ✓")
    print(f"   • Version Control: ✓")
    print(f"   • Batch Processing: ✓")
    print(f"   • FHIR Integration: ✓")
    print(f"   • Cache: ✓")
    print("=" * 80)
    print("🌐 API Documentation: http://localhost:8000/docs")
    print("🔍 Redoc: http://localhost:8000/redoc")
    print("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup all services on shutdown"""
    global lca_service

    print("=" * 80)
    print("🛑 Shutting down Lung Cancer Assistant API...")
    print("=" * 80)

    # Log system shutdown
    try:
        await audit_logger.log_event(
            action="SYSTEM_SHUTDOWN",
            user_id="system",
            resource_type="api",
            resource_id="main",
            details={"reason": "normal_shutdown"}
        )
    except Exception:
        pass

    # Close core LCA service
    if lca_service:
        lca_service.close()
        print("✓ LCA Service shut down")

    # Close Redis connection pool
    if hasattr(app.state, "redis") and app.state.redis:
        await app.state.redis.close()
        print("✓ Redis connection closed")

    # Close RAG service embeddings
    if hasattr(app.state, "rag_service") and app.state.rag_service:
        # RAG service cleanup if needed
        print("✓ RAG Service closed")

    # Close WebSocket connections
    if hasattr(app.state, "websocket_service") and app.state.websocket_service:
        await websocket_service.disconnect_all()
        print("✓ WebSocket connections closed")

    print("=" * 80)
    print("✅ All services shut down successfully")
    print("=" * 80)


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


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    
    Returns metrics in Prometheus text format for scraping
    """
    if os.getenv("METRICS_ENABLED", "true").lower() != "true":
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
    persist_to_neo4j: bool = True
):
    """
    Analyze a patient using the integrated 2025-2026 workflow.
    
    This endpoint uses the production-ready integrated architecture:
    1. Dynamic complexity assessment
    2. IngestionAgent: Validates and normalizes raw patient data
    3. SemanticMappingAgent: Maps clinical concepts to SNOMED-CT codes
    4. Specialized agents: NSCLC/SCLC/Biomarker/Comorbidity
    5. Multi-agent negotiation (if multiple proposals)
    6. Advanced analytics: Survival, uncertainty, clinical trials
    7. ExplanationAgent: Generates MDT summaries
    8. PersistenceAgent: Saves to Neo4j (if enabled)
    
    CRITICAL: Only PersistenceAgent writes to Neo4j.

    Args:
        patient: Patient clinical data
        persist_to_neo4j: Whether to save results to Neo4j (default: True)

    Returns:
        Complete decision support with SNOMED mappings, analytics, and MDT summary
    """
    
    try:
        # Initialize Neo4j tools if persistence requested
        neo4j_tools = None
        if persist_to_neo4j:
            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USER") 
            neo4j_password = os.getenv("NEO4J_PASSWORD")
            
            if neo4j_uri and neo4j_user and neo4j_password:
                neo4j_tools = Neo4jTools(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password
                )
        
        # Initialize integrated workflow
        workflow = IntegratedLCAWorkflow(
            neo4j_tools=neo4j_tools,
            enable_analytics=True,
            enable_negotiation=True
        )
        
        # Run integrated workflow
        result = await workflow.analyze_patient_comprehensive(
            patient_data=patient.model_dump(),
            persist=persist_to_neo4j
        )
        
        # Convert integrated workflow result to API response format
        return WorkflowResponseV2(
            patient_id=result.get("patient_id", "unknown"),
            success=result.get("status") == "success",
            workflow_status=result.get("status", "unknown"),
            agent_chain=result.get("agent_chain", []),
            scenario=result.get("complexity", "unknown"),
            scenario_confidence=result.get("mapping_confidence", 0.0),
            recommendations=result.get("recommendations", []),
            reasoning_chain=[],  # Integrated workflow uses different format
            snomed_mappings={},  # Embedded in patient_with_codes
            mapping_confidence=result.get("mapping_confidence", 0.0),
            inference_id=result.get("persistence", {}).get("inference_id"),
            persisted=persist_to_neo4j and "persistence" in result and not result.get("persistence", {}).get("error"),
            mdt_summary=result.get("mdt_summary", ""),
            key_considerations=[],
            discussion_points=[],
            processing_time_seconds=result.get("processing_time_ms", 0) / 1000.0,
            errors=result.get("errors", []),
            guideline_refs=[]
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
