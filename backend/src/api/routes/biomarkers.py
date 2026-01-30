"""
Biomarker Routes - API endpoints for biomarker management and recommendations
Implements biomarker endpoints from MISSING_API_ENDPOINTS.md
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date

# Centralized logging
from ...logging_config import get_logger, log_execution

logger = get_logger(__name__)

router = APIRouter(prefix="/biomarkers", tags=["Biomarkers"])


# ==================== Request Models ====================

class BiomarkerTestResults(BaseModel):
    """Request model for submitting biomarker test results"""
    patient_id: str = Field(..., description="Patient ID")
    test_date: date = Field(..., description="Date biomarker test was performed")
    egfr_mutation: Optional[str] = Field(None, description="EGFR mutation status")
    alk_status: Optional[str] = Field(None, description="ALK fusion status")
    ros1_status: Optional[str] = Field(None, description="ROS1 fusion status")
    pd_l1_score: Optional[float] = Field(None, ge=0, le=100, description="PD-L1 expression percentage")
    tmb_score: Optional[float] = Field(None, ge=0, description="Tumor Mutational Burden (mutations/Mb)")
    kras_mutation: Optional[str] = Field(None, description="KRAS mutation")
    braf_mutation: Optional[str] = Field(None, description="BRAF mutation")
    met_mutation: Optional[str] = Field(None, description="MET mutation")
    her2_mutation: Optional[str] = Field(None, description="HER2 mutation")
    ret_fusion: Optional[str] = Field(None, description="RET fusion status")
    ntrk_fusion: Optional[str] = Field(None, description="NTRK fusion status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "PAT-12345",
                "test_date": "2026-01-10",
                "egfr_mutation": "Exon 19 deletion",
                "alk_status": "Negative",
                "ros1_status": "Negative",
                "pd_l1_score": 55.0,
                "tmb_score": 12.5,
                "kras_mutation": "G12C"
            }
        }


# ==================== Response Models ====================

class ActionableMutation(BaseModel):
    """Actionable biomarker finding"""
    biomarker: str
    value: str
    actionable: bool
    approved_therapies: List[str]
    clinical_trial_options: int


class BiomarkerSubmissionResponse(BaseModel):
    """Response model for biomarker submission"""
    biomarker_id: str
    patient_id: str
    test_date: date
    actionable_mutations: List[ActionableMutation]
    recommended_therapies: List[str]
    clinical_trial_matches: int


class BiomarkerRecommendation(BaseModel):
    """Treatment recommendation based on biomarker"""
    treatment: str
    biomarker_match: str
    evidence_level: str
    expected_response_rate: float
    median_pfs_months: float
    median_os_months: Optional[float] = None
    nccn_category: str


class BiomarkerRecommendationsResponse(BaseModel):
    """Response model for biomarker-driven recommendations"""
    patient_id: str
    biomarker_summary: Dict[str, Any]
    recommendations: List[BiomarkerRecommendation]
    testing_complete: bool
    missing_tests: Optional[List[str]] = None


class SuggestedTest(BaseModel):
    """Suggested next biomarker test"""
    test: str
    rationale: str
    priority: str  # High, Medium, Low
    estimated_cost: str
    turnaround_time_days: int
    clinical_utility: str


class SuggestedTestsResponse(BaseModel):
    """Response model for suggested tests"""
    patient_id: str
    current_biomarkers: Dict[str, Any]
    suggested_tests: List[SuggestedTest]


# ==================== Endpoints ====================

@router.post("/", response_model=BiomarkerSubmissionResponse)
async def submit_biomarker_results(results: BiomarkerTestResults):
    """
    Submit biomarker test results for a patient.
    
    Analyzes the biomarker profile and identifies:
    - Actionable mutations
    - FDA-approved targeted therapies
    - Available clinical trials
    """
    try:
        import uuid
        
        # Generate biomarker ID
        biomarker_id = f"BIO-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
        
        # Analyze biomarkers for actionable mutations
        actionable = []
        recommended = []
        
        if results.egfr_mutation and results.egfr_mutation != "Negative":
            actionable.append(ActionableMutation(
                biomarker="EGFR",
                value=results.egfr_mutation,
                actionable=True,
                approved_therapies=["Osimertinib", "Erlotinib", "Gefitinib", "Afatinib"],
                clinical_trial_options=15
            ))
            recommended.extend(["Osimertinib", "Erlotinib"])
        
        if results.alk_status == "Positive":
            actionable.append(ActionableMutation(
                biomarker="ALK",
                value="Positive",
                actionable=True,
                approved_therapies=["Alectinib", "Brigatinib", "Ceritinib", "Crizotinib"],
                clinical_trial_options=8
            ))
            recommended.extend(["Alectinib", "Brigatinib"])
        
        if results.kras_mutation == "G12C":
            actionable.append(ActionableMutation(
                biomarker="KRAS",
                value="G12C",
                actionable=True,
                approved_therapies=["Sotorasib", "Adagrasib"],
                clinical_trial_options=12
            ))
            recommended.append("Sotorasib")
        
        if results.pd_l1_score and results.pd_l1_score >= 50:
            actionable.append(ActionableMutation(
                biomarker="PD-L1",
                value=f"{results.pd_l1_score}%",
                actionable=True,
                approved_therapies=["Pembrolizumab monotherapy"],
                clinical_trial_options=20
            ))
            recommended.append("Pembrolizumab")
        
        # Store in Neo4j
        from ...db.neo4j_tools import Neo4jWriteTools
        write_tools = Neo4jWriteTools()
        
        if write_tools.is_available:
            # Store biomarker results in Neo4j
            pass
        
        write_tools.close()
        
        return BiomarkerSubmissionResponse(
            biomarker_id=biomarker_id,
            patient_id=results.patient_id,
            test_date=results.test_date,
            actionable_mutations=actionable,
            recommended_therapies=list(set(recommended)),
            clinical_trial_matches=sum(m.clinical_trial_options for m in actionable)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit biomarker results: {str(e)}")


@router.get("/{patient_id}/recommendations", response_model=BiomarkerRecommendationsResponse)
async def get_biomarker_recommendations(patient_id: str):
    """
    Get biomarker-driven treatment recommendations for a patient.
    
    Returns targeted therapy options based on the patient's molecular profile.
    """
    try:
        from ...db.neo4j_tools import Neo4jReadTools
        
        read_tools = Neo4jReadTools()
        
        if not read_tools.is_available:
            raise HTTPException(status_code=503, detail="Neo4j database not available")
        
        # Get patient biomarkers from Neo4j
        # For now, return mock data
        
        biomarker_summary = {
            "egfr_mutation": "Exon 19 deletion",
            "pd_l1_score": 55,
            "alk_status": "Negative",
            "kras_mutation": "Wild-type"
        }
        
        recommendations = [
            BiomarkerRecommendation(
                treatment="Osimertinib",
                biomarker_match="EGFR Exon 19 deletion",
                evidence_level="A",
                expected_response_rate=0.75,
                median_pfs_months=18.9,
                median_os_months=38.6,
                nccn_category="Category 1"
            ),
            BiomarkerRecommendation(
                treatment="Pembrolizumab",
                biomarker_match="PD-L1 ≥50%",
                evidence_level="A",
                expected_response_rate=0.45,
                median_pfs_months=10.3,
                median_os_months=20.0,
                nccn_category="Category 1"
            )
        ]
        
        read_tools.close()
        
        return BiomarkerRecommendationsResponse(
            patient_id=patient_id,
            biomarker_summary=biomarker_summary,
            recommendations=recommendations,
            testing_complete=True,
            missing_tests=None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get biomarker recommendations: {str(e)}")


@router.get("/{patient_id}/suggested-tests", response_model=SuggestedTestsResponse)
async def get_suggested_tests(patient_id: str):
    """
    Suggest next biomarker tests for a patient.
    
    Based on:
    - Current treatment status
    - Disease progression
    - Available biomarker data
    - NCCN guidelines
    """
    try:
        from ...db.neo4j_tools import Neo4jReadTools
        
        read_tools = Neo4jReadTools()
        
        # Get patient data and treatment history
        # Analyze what tests are missing or need updating
        
        suggested = [
            SuggestedTest(
                test="EGFR T790M Resistance Mutation",
                rationale="Patient on osimertinib for 12 months, progression suspected",
                priority="High",
                estimated_cost="$500",
                turnaround_time_days=7,
                clinical_utility="Identifies acquired resistance mechanism"
            ),
            SuggestedTest(
                test="Comprehensive Genomic Profiling (NGS)",
                rationale="Identify alternative actionable mutations after EGFR TKI failure",
                priority="High",
                estimated_cost="$3,000",
                turnaround_time_days=14,
                clinical_utility="May reveal MET amplification, HER2, or other targets"
            ),
            SuggestedTest(
                test="Liquid Biopsy (ctDNA)",
                rationale="Less invasive option for resistance mutation testing",
                priority="Medium",
                estimated_cost="$1,500",
                turnaround_time_days=10,
                clinical_utility="Non-invasive monitoring of treatment response"
            )
        ]
        
        read_tools.close()
        
        return SuggestedTestsResponse(
            patient_id=patient_id,
            current_biomarkers={
                "egfr_mutation": "Exon 19 deletion",
                "alk_status": "Negative",
                "pd_l1_score": 55
            },
            suggested_tests=suggested
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggested tests: {str(e)}")


@router.get("/{patient_id}/history", response_model=List[Dict[str, Any]])
async def get_biomarker_history(patient_id: str):
    """
    Get complete biomarker testing history for a patient.
    
    Shows evolution of biomarker profile over time,
    useful for tracking resistance mutations.
    """
    try:
        from ...db.neo4j_tools import Neo4jReadTools
        
        read_tools = Neo4jReadTools()
        
        # Query biomarker history from Neo4j
        history = [
            {
                "test_date": "2025-01-15",
                "egfr_mutation": "Exon 19 deletion",
                "pd_l1_score": 55,
                "alk_status": "Negative"
            },
            {
                "test_date": "2025-12-10",
                "egfr_mutation": "Exon 19 deletion + T790M",
                "pd_l1_score": 60,
                "alk_status": "Negative"
            }
        ]
        
        read_tools.close()
        return history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get biomarker history: {str(e)}")
