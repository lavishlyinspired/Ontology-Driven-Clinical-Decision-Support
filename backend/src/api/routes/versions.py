"""
Guideline Version Management API Routes
Handles NCCN guideline versioning, A/B testing, and rollbacks
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# Centralized logging
from ...logging_config import get_logger, log_execution

logger = get_logger(__name__)

from src.services.version_service import version_service
from src.services.audit_service import audit_logger
from src.api.routes.auth import get_current_active_user

router = APIRouter(prefix="/versions", tags=["Version Management"])


# ==================== Pydantic Models ====================

class GuidelineVersionCreate(BaseModel):
    """Create a new guideline version"""
    name: str = Field(..., description="Version name (e.g., NCCN 2024.1)")
    guideline_type: str = Field(..., description="Guideline type (NCCN, NICE, etc.)")
    version_number: str = Field(..., description="Semantic version (e.g., 2024.1.0)")
    changes: str = Field(..., description="Summary of changes from previous version")
    rules: Dict[str, Any] = Field(..., description="Complete guideline rules")
    effective_date: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "NCCN NSCLC 2024.1",
                "guideline_type": "NCCN",
                "version_number": "2024.1.0",
                "changes": "Updated immunotherapy recommendations for stage III disease",
                "rules": {
                    "IA1": {"treatment": "Lobectomy", "evidence": "1A"},
                    "IA2": {"treatment": "Lobectomy or SBRT", "evidence": "1A"}
                },
                "effective_date": "2024-01-15T00:00:00",
                "metadata": {"source_url": "https://nccn.org"}
            }
        }


class GuidelineVersionResponse(BaseModel):
    """Guideline version information"""
    version_id: str
    name: str
    guideline_type: str
    version_number: str
    changes: str
    status: str
    created_at: datetime
    created_by: str
    activated_at: Optional[datetime]
    deactivated_at: Optional[datetime]
    effective_date: Optional[datetime]
    usage_count: int
    metadata: Optional[Dict[str, Any]]


class ABTestCreate(BaseModel):
    """Create an A/B test between guideline versions"""
    name: str = Field(..., description="A/B test name")
    description: str
    control_version_id: str = Field(..., description="Current/control version")
    treatment_version_id: str = Field(..., description="New/treatment version")
    traffic_split: float = Field(default=0.5, ge=0.0, le=1.0, description="% traffic to treatment")
    duration_days: int = Field(default=30, ge=1, le=365)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "NCCN 2024.1 vs 2024.2",
                "description": "Test new immunotherapy recommendations",
                "control_version_id": "version_2024_1",
                "treatment_version_id": "version_2024_2",
                "traffic_split": 0.2,
                "duration_days": 30
            }
        }


class ABTestResponse(BaseModel):
    """A/B test information"""
    test_id: str
    name: str
    description: str
    control_version_id: str
    treatment_version_id: str
    traffic_split: float
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    results: Optional[Dict[str, Any]]


# ==================== API Endpoints ====================

@router.post("/", response_model=GuidelineVersionResponse, status_code=status.HTTP_201_CREATED)
async def create_guideline_version(
    version: GuidelineVersionCreate,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Create a new guideline version
    
    Requires admin role
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    user_id = current_user["sub"]
    
    try:
        # Create version
        new_version = await version_service.create_version(
            name=version.name,
            guideline_type=version.guideline_type,
            version_number=version.version_number,
            changes=version.changes,
            rules=version.rules,
            created_by=user_id,
            effective_date=version.effective_date,
            metadata=version.metadata
        )
        
        # Log audit event
        await audit_logger.log_event(
            action="GUIDELINE_VERSION_CREATED",
            user_id=user_id,
            resource_type="guideline_version",
            resource_id=new_version["version_id"],
            details={
                "name": version.name,
                "version_number": version.version_number
            }
        )
        
        return GuidelineVersionResponse(**new_version)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/", response_model=List[GuidelineVersionResponse])
async def list_guideline_versions(
    guideline_type: Optional[str] = None,
    status_filter: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user)
):
    """
    List all guideline versions
    
    Filters:
    - guideline_type: NCCN, NICE, etc.
    - status: draft, active, inactive, deprecated
    """
    versions = await version_service.list_versions(
        guideline_type=guideline_type,
        status=status_filter
    )
    
    return [GuidelineVersionResponse(**v) for v in versions]


@router.get("/{version_id}", response_model=GuidelineVersionResponse)
async def get_guideline_version(
    version_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get detailed information about a specific version
    """
    version = await version_service.get_version(version_id)
    
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version_id} not found"
        )
    
    return GuidelineVersionResponse(**version)


@router.post("/{version_id}/activate")
async def activate_guideline_version(
    version_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Activate a guideline version (makes it the active version)
    
    Automatically deactivates previous active version
    Requires admin role
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    success = await version_service.activate_version(version_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version_id} not found"
        )
    
    # Log audit event
    await audit_logger.log_event(
        action="GUIDELINE_VERSION_ACTIVATED",
        user_id=current_user["sub"],
        resource_type="guideline_version",
        resource_id=version_id,
        details={}
    )
    
    return {
        "message": f"Version {version_id} activated successfully",
        "version_id": version_id,
        "status": "active"
    }


@router.post("/{version_id}/deactivate")
async def deactivate_guideline_version(
    version_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Deactivate a guideline version
    
    Requires admin role
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    success = await version_service.deactivate_version(version_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version_id} not found"
        )
    
    # Log audit event
    await audit_logger.log_event(
        action="GUIDELINE_VERSION_DEACTIVATED",
        user_id=current_user["sub"],
        resource_type="guideline_version",
        resource_id=version_id,
        details={}
    )
    
    return {
        "message": f"Version {version_id} deactivated successfully",
        "version_id": version_id,
        "status": "inactive"
    }


@router.post("/{version_id}/rollback")
async def rollback_to_version(
    version_id: str,
    reason: str = Field(..., description="Reason for rollback"),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Rollback to a previous guideline version
    
    Emergency rollback in case of issues with current version
    Requires admin role
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    success = await version_service.rollback_to_version(version_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version_id} not found"
        )
    
    # Log audit event
    await audit_logger.log_event(
        action="GUIDELINE_VERSION_ROLLBACK",
        user_id=current_user["sub"],
        resource_type="guideline_version",
        resource_id=version_id,
        details={"reason": reason}
    )
    
    return {
        "message": f"Rolled back to version {version_id}",
        "version_id": version_id,
        "status": "active",
        "reason": reason
    }


@router.get("/{version_id}/compare/{other_version_id}")
async def compare_versions(
    version_id: str,
    other_version_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Compare two guideline versions and show differences
    """
    comparison = await version_service.compare_versions(version_id, other_version_id)
    
    if not comparison:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or both versions not found"
        )
    
    return comparison


# ==================== A/B Testing Endpoints ====================

@router.post("/ab-tests", response_model=ABTestResponse, status_code=status.HTTP_201_CREATED)
async def create_ab_test(
    test: ABTestCreate,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Create an A/B test between two guideline versions
    
    Allows gradual rollout and validation of new guidelines
    Requires admin role
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        ab_test = await version_service.create_ab_test(
            name=test.name,
            description=test.description,
            control_version_id=test.control_version_id,
            treatment_version_id=test.treatment_version_id,
            traffic_split=test.traffic_split,
            duration_days=test.duration_days
        )
        
        # Log audit event
        await audit_logger.log_event(
            action="AB_TEST_CREATED",
            user_id=current_user["sub"],
            resource_type="ab_test",
            resource_id=ab_test["test_id"],
            details={
                "name": test.name,
                "traffic_split": test.traffic_split
            }
        )
        
        return ABTestResponse(**ab_test)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/ab-tests", response_model=List[ABTestResponse])
async def list_ab_tests(
    status_filter: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user)
):
    """
    List all A/B tests
    
    Filters:
    - status: running, completed, stopped
    """
    tests = await version_service.list_ab_tests(status=status_filter)
    
    return [ABTestResponse(**t) for t in tests]


@router.get("/ab-tests/{test_id}", response_model=ABTestResponse)
async def get_ab_test(
    test_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get A/B test details and results
    """
    test = await version_service.get_ab_test(test_id)
    
    if not test:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"A/B test {test_id} not found"
        )
    
    return ABTestResponse(**test)


@router.post("/ab-tests/{test_id}/stop")
async def stop_ab_test(
    test_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Stop a running A/B test
    
    Requires admin role
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    success = await version_service.stop_ab_test(test_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"A/B test {test_id} not found or already stopped"
        )
    
    # Log audit event
    await audit_logger.log_event(
        action="AB_TEST_STOPPED",
        user_id=current_user["sub"],
        resource_type="ab_test",
        resource_id=test_id,
        details={}
    )
    
    return {
        "message": f"A/B test {test_id} stopped",
        "test_id": test_id,
        "status": "stopped"
    }
