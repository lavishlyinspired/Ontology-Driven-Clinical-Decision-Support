"""
Guidelines Routes - API endpoints for clinical guideline operations
Implements guideline-related endpoints from LCA_Complete_Implementation_Plan
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

router = APIRouter(prefix="/guidelines", tags=["Guidelines"])


# ==================== Response Models ====================

class GuidelineRuleResponse(BaseModel):
    """Response model for guideline rule"""
    rule_id: str
    name: str
    source: str
    description: str
    recommended_treatment: str
    treatment_intent: str
    evidence_level: str
    contraindications: List[str]
    survival_benefit: Optional[str]


class GuidelineOutcomeResponse(BaseModel):
    """Response model for guideline outcomes"""
    guideline_id: str
    guideline_name: Optional[str]
    treatment_type: Optional[str]
    patient_count: int
    avg_survival_days: Optional[float]
    outcome_statuses: List[str]


class GuidelineMatchResponse(BaseModel):
    """Response model for guideline matching result"""
    rule_id: str
    rule_name: str
    matches: bool
    confidence: float
    reason: str


# ==================== Endpoints ====================

@router.get("/", response_model=List[GuidelineRuleResponse])
async def list_guidelines():
    """
    List all available clinical guideline rules.
    
    Returns NICE Lung Cancer Guidelines and contemporary precision medicine rules.
    """
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        if hasattr(lca_service, 'rule_engine'):
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
                    contraindications=rule.contraindications,
                    survival_benefit=rule.survival_benefit
                )
                for rule in rules
            ]
        
        return []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve guidelines: {str(e)}")


@router.get("/{rule_id}", response_model=GuidelineRuleResponse)
async def get_guideline(rule_id: str):
    """
    Get a specific guideline rule by ID.
    
    Args:
        rule_id: Guideline rule ID (e.g., "R1", "R2", ...)
    """
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        if hasattr(lca_service, 'rule_engine'):
            rule = lca_service.rule_engine.get_rule_by_id(rule_id)
            
            if rule:
                return GuidelineRuleResponse(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    source=rule.source,
                    description=rule.description,
                    recommended_treatment=rule.recommended_treatment,
                    treatment_intent=rule.treatment_intent,
                    evidence_level=rule.evidence_level,
                    contraindications=rule.contraindications,
                    survival_benefit=rule.survival_benefit
                )
        
        raise HTTPException(status_code=404, detail=f"Guideline {rule_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve guideline: {str(e)}")


@router.get("/{rule_id}/outcomes", response_model=GuidelineOutcomeResponse)
async def get_guideline_outcomes(rule_id: str):
    """
    Get outcome statistics for patients treated according to a guideline.
    
    Returns aggregated outcome data including:
    - Patient count
    - Average survival
    - Outcome status distribution
    """
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Verify guideline exists
        if hasattr(lca_service, 'rule_engine'):
            rule = lca_service.rule_engine.get_rule_by_id(rule_id)
            if not rule:
                raise HTTPException(status_code=404, detail=f"Guideline {rule_id} not found")
        
        # Get outcomes from Neo4j
        from ...db.neo4j_tools import Neo4jReadTools
        read_tools = Neo4jReadTools()
        
        if read_tools.is_available:
            outcomes = read_tools.get_guideline_outcomes(rule_id)
            read_tools.close()
            
            return GuidelineOutcomeResponse(
                guideline_id=outcomes.get("guideline_id", rule_id),
                guideline_name=outcomes.get("guideline_name"),
                treatment_type=outcomes.get("treatment_type"),
                patient_count=outcomes.get("patient_count", 0),
                avg_survival_days=outcomes.get("avg_survival_days"),
                outcome_statuses=outcomes.get("outcome_statuses", [])
            )
        
        # Return empty outcome if Neo4j not available
        return GuidelineOutcomeResponse(
            guideline_id=rule_id,
            guideline_name=None,
            treatment_type=None,
            patient_count=0,
            avg_survival_days=None,
            outcome_statuses=[]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve outcomes: {str(e)}")


@router.post("/match")
async def match_guidelines(
    tnm_stage: str = Query(..., description="TNM stage"),
    histology_type: str = Query(..., description="Histology type"),
    performance_status: int = Query(..., ge=0, le=4, description="WHO Performance Status"),
    biomarkers: Optional[List[str]] = Query(default=None, description="Biomarker status list")
):
    """
    Match patient profile against all guideline rules.
    
    Returns which guidelines apply and why.
    """
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        patient_data = {
            "tnm_stage": tnm_stage,
            "histology_type": histology_type,
            "performance_status": performance_status,
            "biomarkers": biomarkers or []
        }
        
        matches = []
        
        if hasattr(lca_service, 'rule_engine'):
            all_rules = lca_service.rule_engine.get_all_rules()
            applicable = lca_service.rule_engine.classify_patient(patient_data)
            applicable_ids = [r.get("rule_id") for r in applicable]
            
            for rule in all_rules:
                is_match = rule.rule_id in applicable_ids
                matches.append(GuidelineMatchResponse(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    matches=is_match,
                    confidence=0.95 if is_match else 0.0,
                    reason=f"Matches criteria" if is_match else "Does not match criteria"
                ).model_dump())
        
        return {
            "patient_profile": patient_data,
            "matches": matches,
            "applicable_count": len([m for m in matches if m["matches"]])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")


@router.post("/search")
async def search_guidelines(
    query: str = Query(..., description="Natural language search query"),
    limit: int = Query(default=5, ge=1, le=20, description="Maximum results")
):
    """
    Semantic search for guidelines using vector store.
    
    Uses embeddings to find guidelines matching natural language queries.
    """
    from ..main import lca_service
    
    if lca_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        if hasattr(lca_service, 'vector_store') and lca_service.vector_store:
            results = lca_service.vector_store.search_guidelines(query, n_results=limit)
            
            return {
                "query": query,
                "results": results,
                "count": len(results)
            }
        
        return {"query": query, "results": [], "count": 0, "message": "Vector store not available"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/sources/summary")
async def get_guideline_sources():
    """
    Get summary of guideline sources.
    
    Returns information about the clinical guidelines used in the system.
    """
    return {
        "sources": [
            {
                "name": "NICE Lung Cancer 2011",
                "code": "CG121",
                "description": "National Institute for Health and Care Excellence guidelines for lung cancer diagnosis and management",
                "last_updated": "2011",
                "rules": ["R1", "R2", "R3", "R4", "R5", "R6"]
            },
            {
                "name": "Contemporary Immunotherapy Guidelines 2025",
                "description": "Modern immunotherapy guidelines for advanced NSCLC",
                "last_updated": "2025",
                "rules": ["R7"]
            },
            {
                "name": "ESMO/ASCO Precision Medicine Guidelines 2025",
                "description": "Molecular targeted therapy guidelines for biomarker-positive NSCLC",
                "last_updated": "2025",
                "rules": ["R8", "R9", "R10"]
            }
        ],
        "total_rules": 10,
        "evidence_levels": {
            "Grade A": "High-quality randomized controlled trials, meta-analyses",
            "Grade B": "Well-designed clinical studies",
            "Grade C": "Expert opinion, case series"
        }
    }
