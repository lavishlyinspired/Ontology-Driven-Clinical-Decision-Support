"""
Human-in-the-Loop (HITL) API Routes
Handles review queue, case approvals, and clinical oversight
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

from src.services.hitl_service import hitl_service, ReviewStatus
from src.services.audit_service import audit_logger
from src.api.routes.auth import get_current_active_user

router = APIRouter(prefix="/hitl", tags=["Human-in-the-Loop"])


# ==================== Pydantic Models ====================

class CaseSubmission(BaseModel):
    """Submit a case for human review"""
    patient_id: str
    case_type: str = Field(..., description="Type of case (recommendation, classification, etc.)")
    recommendations: dict
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., description="Reason requiring review")
    priority: int = Field(default=2, ge=1, le=3, description="Priority (1=high, 2=medium, 3=low)")
    metadata: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "P12345",
                "case_type": "treatment_recommendation",
                "recommendations": {
                    "primary": "Concurrent chemoradiotherapy",
                    "alternative": "Sequential chemotherapy"
                },
                "confidence_score": 0.72,
                "reason": "Conflicting guidelines for stage IIIA with COPD",
                "priority": 1,
                "metadata": {"tnm_stage": "IIIA", "comorbidities": ["COPD"]}
            }
        }


class ReviewDecision(BaseModel):
    """Clinician's review decision"""
    action: str = Field(..., pattern="^(approve|reject|request_changes)$")
    feedback: str = Field(..., min_length=10, description="Clinical reasoning")
    modified_recommendations: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "action": "approve",
                "feedback": "Recommendation appropriate given patient's FEV1 of 65% and stage IIIA disease. Concurrent chemoradiotherapy is guideline-concordant.",
                "modified_recommendations": None
            }
        }


class ReviewCaseResponse(BaseModel):
    """Review case information"""
    case_id: str
    patient_id: str
    case_type: str
    recommendations: dict
    confidence_score: float
    reason: str
    priority: int
    status: str
    submitted_at: datetime
    submitted_by: str
    reviewed_at: Optional[datetime]
    reviewed_by: Optional[str]
    reviewer_feedback: Optional[str]
    metadata: Optional[dict]


class QueueSummary(BaseModel):
    """Review queue statistics"""
    total_pending: int
    high_priority: int
    medium_priority: int
    low_priority: int
    avg_waiting_time_hours: float
    oldest_case_hours: float


# ==================== API Endpoints ====================

@router.post("/submit", response_model=ReviewCaseResponse, status_code=status.HTTP_201_CREATED)
async def submit_case_for_review(
    case: CaseSubmission,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Submit a case for human clinical review
    
    Use when:
    - Confidence score below threshold
    - Conflicting guidelines
    - Unusual patient characteristics
    - High-risk recommendations
    """
    user_id = current_user["sub"]
    
    # Submit to HITL service
    review_case = await hitl_service.submit_for_review(
        patient_id=case.patient_id,
        case_type=case.case_type,
        recommendations=case.recommendations,
        confidence_score=case.confidence_score,
        reason=case.reason,
        submitted_by=user_id,
        priority=case.priority,
        metadata=case.metadata
    )
    
    # Log audit event
    await audit_logger.log_event(
        action="HITL_CASE_SUBMITTED",
        user_id=user_id,
        resource_type="hitl_case",
        resource_id=review_case["case_id"],
        details={
            "patient_id": case.patient_id,
            "case_type": case.case_type,
            "confidence_score": case.confidence_score,
            "priority": case.priority
        }
    )
    
    return ReviewCaseResponse(**review_case)


@router.get("/queue", response_model=List[ReviewCaseResponse])
async def get_review_queue(
    status_filter: Optional[str] = None,
    priority: Optional[int] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get pending cases in review queue
    
    Filters:
    - status: pending, approved, rejected, expired
    - priority: 1 (high), 2 (medium), 3 (low)
    - limit: max number of cases to return
    """
    # Check clinician permission
    if current_user.get("role") not in ["admin", "clinician"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clinician access required"
        )
    
    # Get queue
    cases = await hitl_service.get_review_queue(
        status=ReviewStatus(status_filter) if status_filter else None,
        priority=priority,
        limit=limit
    )
    
    return [ReviewCaseResponse(**case) for case in cases]


@router.get("/queue/summary", response_model=QueueSummary)
async def get_queue_summary(current_user: dict = Depends(get_current_active_user)):
    """
    Get review queue statistics and metrics
    """
    # Check clinician permission
    if current_user.get("role") not in ["admin", "clinician"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clinician access required"
        )
    
    summary = await hitl_service.get_queue_summary()
    return QueueSummary(**summary)


@router.get("/cases/{case_id}", response_model=ReviewCaseResponse)
async def get_case_details(
    case_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get detailed information about a specific review case
    """
    case = await hitl_service.get_case(case_id)
    
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case {case_id} not found"
        )
    
    return ReviewCaseResponse(**case)


@router.post("/cases/{case_id}/review")
async def review_case(
    case_id: str,
    decision: ReviewDecision,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Submit a clinical review decision for a case
    
    Actions:
    - approve: Accept AI recommendation
    - reject: Reject recommendation (provide alternative)
    - request_changes: Request model refinement
    """
    # Check clinician permission
    if current_user.get("role") not in ["admin", "clinician"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clinician access required"
        )
    
    reviewer_id = current_user["sub"]
    
    # Process review
    if decision.action == "approve":
        result = await hitl_service.approve_case(
            case_id=case_id,
            reviewer_id=reviewer_id,
            feedback=decision.feedback
        )
    elif decision.action == "reject":
        result = await hitl_service.reject_case(
            case_id=case_id,
            reviewer_id=reviewer_id,
            feedback=decision.feedback,
            alternative_recommendation=decision.modified_recommendations
        )
    else:  # request_changes
        result = await hitl_service.request_changes(
            case_id=case_id,
            reviewer_id=reviewer_id,
            feedback=decision.feedback
        )
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case {case_id} not found or already reviewed"
        )
    
    # Log audit event
    await audit_logger.log_event(
        action=f"HITL_CASE_{decision.action.upper()}",
        user_id=reviewer_id,
        resource_type="hitl_case",
        resource_id=case_id,
        details={
            "action": decision.action,
            "feedback_length": len(decision.feedback)
        }
    )
    
    return {
        "message": f"Case {decision.action}d successfully",
        "case_id": case_id,
        "status": result["status"]
    }


@router.get("/my-reviews", response_model=List[ReviewCaseResponse])
async def get_my_reviews(
    limit: int = 20,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get cases reviewed by current user
    """
    reviewer_id = current_user["sub"]
    
    cases = await hitl_service.get_reviewed_by_user(reviewer_id, limit=limit)
    
    return [ReviewCaseResponse(**case) for case in cases]


@router.get("/metrics")
async def get_hitl_metrics(
    days: int = 30,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get HITL system metrics and performance stats
    
    Metrics include:
    - Total cases submitted/reviewed
    - Approval/rejection rates
    - Average review time
    - Inter-rater agreement
    """
    # Check admin/clinician permission
    if current_user.get("role") not in ["admin", "clinician"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clinician access required"
        )
    
    metrics = await hitl_service.get_metrics(days=days)
    
    return {
        "period_days": days,
        "metrics": metrics,
        "generated_at": datetime.now().isoformat()
    }


@router.delete("/cases/{case_id}")
async def delete_case(
    case_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Delete a review case (admin only, for expired/invalid cases)
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    success = await hitl_service.delete_case(case_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case {case_id} not found"
        )
    
    # Log audit event
    await audit_logger.log_event(
        action="HITL_CASE_DELETED",
        user_id=current_user["sub"],
        resource_type="hitl_case",
        resource_id=case_id,
        details={}
    )
    
    return {"message": f"Case {case_id} deleted successfully"}
