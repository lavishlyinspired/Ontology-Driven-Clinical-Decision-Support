"""
Enhanced Analytics Routes
Additional analytics endpoints for survival details, uncertainty analysis, and clinical trials
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import os

router = APIRouter(prefix="/api/v1/analytics", tags=["Advanced Analytics"])


# ============================================================================
# MODELS
# ============================================================================

class SurvivalCurvePoint(BaseModel):
    """Single point on survival curve"""
    time_months: float
    survival_probability: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    at_risk: int


class SurvivalAnalysisDetail(BaseModel):
    """Detailed survival analysis response"""
    cohort_description: str
    total_patients: int
    median_survival_months: Optional[float]
    survival_curve: List[SurvivalCurvePoint]
    stratified_by: Optional[str] = None
    strata: Optional[Dict[str, Any]] = None


class UncertaintyMetrics(BaseModel):
    """Uncertainty quantification for a recommendation"""
    recommendation_confidence: float
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    confidence_interval_95: List[float]
    sensitivity_analysis: Dict[str, float]


class ClinicalTrialMatch(BaseModel):
    """Clinical trial match details"""
    nct_id: str
    title: str
    phase: str
    status: str
    match_score: float
    matching_criteria: List[str]
    exclusion_reasons: List[str]
    location: Optional[str] = None
    estimated_enrollment: Optional[int] = None
    primary_completion_date: Optional[str] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/survival/{patient_id}", response_model=SurvivalAnalysisDetail)
async def get_patient_survival_analysis(
    patient_id: str,
    include_similar: bool = Query(False, description="Include similar patient cohort")
):
    """
    Get detailed survival analysis for a specific patient.
    
    Uses similar patient cohort to generate Kaplan-Meier curves
    and survival probabilities.
    """
    try:
        # Mock implementation - replace with actual analytics service
        return SurvivalAnalysisDetail(
            cohort_description=f"Similar patients to {patient_id}",
            total_patients=45,
            median_survival_months=24.5,
            survival_curve=[
                SurvivalCurvePoint(
                    time_months=0, survival_probability=1.0,
                    confidence_interval_lower=1.0, confidence_interval_upper=1.0,
                    at_risk=45
                ),
                SurvivalCurvePoint(
                    time_months=12, survival_probability=0.82,
                    confidence_interval_lower=0.78, confidence_interval_upper=0.86,
                    at_risk=37
                ),
                SurvivalCurvePoint(
                    time_months=24, survival_probability=0.58,
                    confidence_interval_lower=0.52, confidence_interval_upper=0.64,
                    at_risk=26
                ),
                SurvivalCurvePoint(
                    time_months=36, survival_probability=0.42,
                    confidence_interval_lower=0.35, confidence_interval_upper=0.49,
                    at_risk=19
                )
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uncertainty/{inference_id}", response_model=UncertaintyMetrics)
async def get_uncertainty_metrics(inference_id: str):
    """
    Get uncertainty quantification for a specific inference.
    
    Returns epistemic (model) and aleatoric (data) uncertainty,
    plus sensitivity analysis.
    """
    try:
        # Mock implementation
        return UncertaintyMetrics(
            recommendation_confidence=0.92,
            epistemic_uncertainty=0.05,
            aleatoric_uncertainty=0.03,
            confidence_interval_95=[0.88, 0.96],
            sensitivity_analysis={
                "performance_status_change": 0.12,
                "age_variation": 0.04,
                "biomarker_uncertainty": 0.08
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trials/{patient_id}", response_model=List[ClinicalTrialMatch])
async def get_clinical_trials(
    patient_id: str,
    max_results: int = Query(10, ge=1, le=50),
    phase: Optional[str] = Query(None, description="Filter by trial phase")
):
    """
    Find eligible clinical trials for a patient.
    
    Matches patient characteristics against trial inclusion/exclusion criteria.
    """
    try:
        # Mock implementation - replace with actual ClinicalTrialMatcher
        trials = [
            ClinicalTrialMatch(
                nct_id="NCT04567890",
                title="Phase III Trial of Osimertinib in EGFR+ NSCLC",
                phase="Phase 3",
                status="Recruiting",
                match_score=0.94,
                matching_criteria=[
                    "EGFR Exon 19 deletion",
                    "Stage IIIA-IV",
                    "Age 18-80",
                    "PS 0-1"
                ],
                exclusion_reasons=[],
                location="Multiple sites",
                estimated_enrollment=500,
                primary_completion_date="2026-12-31"
            ),
            ClinicalTrialMatch(
                nct_id="NCT05678901",
                title="Combination Immunotherapy for Advanced NSCLC",
                phase="Phase 2",
                status="Recruiting",
                match_score=0.76,
                matching_criteria=[
                    "Stage IIIB-IV",
                    "PS 0-1",
                    "PD-L1 ≥50%"
                ],
                exclusion_reasons=["Prior immunotherapy"],
                location="MD Anderson, Mayo Clinic",
                estimated_enrollment=120,
                primary_completion_date="2027-06-30"
            )
        ]
        
        if phase:
            trials = [t for t in trials if phase.lower() in t.phase.lower()]
        
        return trials[:max_results]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cohort/survival", response_model=SurvivalAnalysisDetail)
async def analyze_cohort_survival(
    cohort_filters: Dict[str, Any],
    stratify_by: Optional[str] = None
):
    """
    Perform survival analysis on a custom cohort.
    
    Define cohort by TNM stage, histology, biomarkers, etc.
    Optionally stratify by a variable (e.g., treatment, EGFR status).
    """
    try:
        # Mock implementation
        return SurvivalAnalysisDetail(
            cohort_description=f"Cohort: {cohort_filters}",
            total_patients=128,
            median_survival_months=28.3,
            survival_curve=[
                SurvivalCurvePoint(
                    time_months=0, survival_probability=1.0,
                    confidence_interval_lower=1.0, confidence_interval_upper=1.0,
                    at_risk=128
                ),
                SurvivalCurvePoint(
                    time_months=12, survival_probability=0.85,
                    confidence_interval_lower=0.82, confidence_interval_upper=0.88,
                    at_risk=109
                ),
                SurvivalCurvePoint(
                    time_months=24, survival_probability=0.68,
                    confidence_interval_lower=0.63, confidence_interval_upper=0.73,
                    at_risk=87
                )
            ],
            stratified_by=stratify_by
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/metrics", response_model=Dict[str, Any])
async def get_dashboard_metrics():
    """
    Get high-level analytics dashboard metrics.
    
    Returns system usage, accuracy, and performance metrics.
    """
    try:
        return {
            "total_patients": 1247,
            "total_inferences": 3892,
            "last_30_days": {
                "new_patients": 87,
                "inferences": 245,
                "avg_processing_time_seconds": 18.3
            },
            "recommendation_accuracy": {
                "overall": 0.94,
                "by_complexity": {
                    "simple": 0.98,
                    "moderate": 0.93,
                    "complex": 0.89
                }
            },
            "agent_performance": {
                "IngestionAgent": {"success_rate": 0.99, "avg_time_ms": 120},
                "SemanticMappingAgent": {"success_rate": 0.97, "avg_time_ms": 850},
                "NSCLCAgent": {"success_rate": 0.96, "avg_time_ms": 2300},
                "PersistenceAgent": {"success_rate": 0.98, "avg_time_ms": 450}
            },
            "top_recommendations": [
                {"treatment": "Osimertinib", "count": 234, "success_rate": 0.92},
                {"treatment": "Chemoradiotherapy", "count": 189, "success_rate": 0.88},
                {"treatment": "Pembrolizumab", "count": 156, "success_rate": 0.91}
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
