"""
Batch Processing API Routes
Handles population-level analysis, bulk patient processing, and async jobs
"""

from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# Centralized logging
from ...logging_config import get_logger, log_execution

logger = get_logger(__name__)

from src.services.batch_service import batch_service, JobStatus
from src.services.audit_service import audit_logger
from src.api.routes.auth import get_current_active_user

router = APIRouter(prefix="/batch", tags=["Batch Processing"])


# ==================== Pydantic Models ====================

class JobType(str, Enum):
    """Supported batch job types"""
    POPULATION_ANALYSIS = "population_analysis"
    COHORT_SIMILARITY = "cohort_similarity"
    GUIDELINE_COMPLIANCE = "guideline_compliance"
    OUTCOME_PREDICTION = "outcome_prediction"
    BULK_IMPORT = "bulk_import"


class BatchJobSubmission(BaseModel):
    """Submit a batch processing job"""
    job_type: JobType
    name: str = Field(..., description="Job name")
    description: Optional[str] = None
    patient_ids: Optional[List[str]] = Field(None, description="List of patient IDs to process")
    cohort_filter: Optional[Dict[str, Any]] = Field(None, description="Cohort selection criteria")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Job-specific parameters")
    priority: int = Field(default=2, ge=1, le=3, description="Job priority (1=high, 2=medium, 3=low)")

    class Config:
        json_schema_extra = {
            "example": {
                "job_type": "population_analysis",
                "name": "Stage IIIA NSCLC Outcomes Analysis",
                "description": "Analyze treatment outcomes for stage IIIA patients",
                "cohort_filter": {
                    "tnm_stage": "IIIA",
                    "histology_type": "Adenocarcinoma",
                    "min_age": 50,
                    "max_age": 75
                },
                "parameters": {
                    "include_survival_analysis": True,
                    "time_horizon_years": 5
                },
                "priority": 1
            }
        }


class BatchJobResponse(BaseModel):
    """Batch job information"""
    job_id: str
    job_type: str
    name: str
    description: Optional[str]
    status: str
    progress: float
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    submitted_at: datetime
    submitted_by: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    results_summary: Optional[Dict[str, Any]]
    error_message: Optional[str]


class TaskResponse(BaseModel):
    """Individual task within a batch job"""
    task_id: str
    job_id: str
    task_type: str
    patient_id: Optional[str]
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]


# ==================== API Endpoints ====================

@router.post("/jobs", response_model=BatchJobResponse, status_code=status.HTTP_201_CREATED)
async def submit_batch_job(
    job: BatchJobSubmission,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Submit a batch processing job
    
    Job Types:
    - population_analysis: Analyze treatment patterns across population
    - cohort_similarity: Find similar patient cohorts
    - guideline_compliance: Audit guideline adherence
    - outcome_prediction: Predict outcomes for multiple patients
    - bulk_import: Import large datasets
    """
    user_id = current_user["sub"]
    
    try:
        # Prepare tasks based on job type
        if job.patient_ids:
            # Process specific patient list
            tasks = [
                {"task_type": "analyze_patient", "patient_id": pid, "parameters": job.parameters}
                for pid in job.patient_ids
            ]
        elif job.cohort_filter:
            # Create cohort-based tasks (would query Neo4j for matching patients)
            # For now, create placeholder task
            tasks = [
                {"task_type": "cohort_analysis", "cohort_filter": job.cohort_filter, "parameters": job.parameters}
            ]
        else:
            raise ValueError("Must provide either patient_ids or cohort_filter")
        
        # Submit job
        batch_job = await batch_service.submit_job(
            job_type=job.job_type.value,
            name=job.name,
            description=job.description,
            tasks=tasks,
            submitted_by=user_id,
            priority=job.priority
        )
        
        # Log audit event
        await audit_logger.log_event(
            action="BATCH_JOB_SUBMITTED",
            user_id=user_id,
            resource_type="batch_job",
            resource_id=batch_job["job_id"],
            details={
                "job_type": job.job_type.value,
                "name": job.name,
                "total_tasks": len(tasks)
            }
        )
        
        return BatchJobResponse(**batch_job)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/jobs/upload", response_model=BatchJobResponse, status_code=status.HTTP_201_CREATED)
async def submit_bulk_import_job(
    file: UploadFile = File(...),
    job_name: str = Field(..., description="Job name"),
    file_format: str = Field(default="csv", pattern="^(csv|json|fhir)$"),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Submit bulk import job from uploaded file
    
    Supports:
    - CSV: Patient data in CSV format
    - JSON: JSON array of patient records
    - FHIR: FHIR Bundle
    """
    user_id = current_user["sub"]
    
    try:
        # Read file content
        content = await file.read()
        
        # Parse based on format
        if file_format == "csv":
            # Parse CSV (simplified)
            import csv
            import io
            reader = csv.DictReader(io.StringIO(content.decode('utf-8')))
            tasks = [
                {"task_type": "import_patient", "patient_data": row}
                for row in reader
            ]
        elif file_format == "json":
            import json
            data = json.loads(content)
            tasks = [
                {"task_type": "import_patient", "patient_data": record}
                for record in data
            ]
        elif file_format == "fhir":
            # FHIR bundle import
            tasks = [
                {"task_type": "import_fhir_bundle", "bundle_data": content.decode('utf-8')}
            ]
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        # Submit job
        batch_job = await batch_service.submit_job(
            job_type=JobType.BULK_IMPORT.value,
            name=job_name,
            description=f"Bulk import from {file.filename}",
            tasks=tasks,
            submitted_by=user_id,
            priority=2
        )
        
        # Log audit event
        await audit_logger.log_event(
            action="BULK_IMPORT_SUBMITTED",
            user_id=user_id,
            resource_type="batch_job",
            resource_id=batch_job["job_id"],
            details={
                "filename": file.filename,
                "format": file_format,
                "total_records": len(tasks)
            }
        )
        
        return BatchJobResponse(**batch_job)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process file: {str(e)}"
        )


@router.get("/jobs", response_model=List[BatchJobResponse])
async def list_batch_jobs(
    status_filter: Optional[str] = None,
    job_type: Optional[JobType] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_active_user)
):
    """
    List batch jobs
    
    Filters:
    - status: pending, running, completed, failed, cancelled
    - job_type: population_analysis, cohort_similarity, etc.
    - limit: max number of jobs to return
    """
    jobs = await batch_service.list_jobs(
        status=JobStatus(status_filter) if status_filter else None,
        job_type=job_type.value if job_type else None,
        user_id=current_user["sub"],
        limit=limit
    )
    
    return [BatchJobResponse(**job) for job in jobs]


@router.get("/jobs/{job_id}", response_model=BatchJobResponse)
async def get_batch_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get batch job status and results
    """
    job = await batch_service.get_job_status(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Check authorization (user can only see their own jobs unless admin)
    if job["submitted_by"] != current_user["sub"] and current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return BatchJobResponse(**job)


@router.get("/jobs/{job_id}/tasks", response_model=List[TaskResponse])
async def get_job_tasks(
    job_id: str,
    status_filter: Optional[str] = None,
    limit: int = 100,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get all tasks for a specific batch job
    """
    # Check job exists and user has access
    job = await batch_service.get_job_status(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    if job["submitted_by"] != current_user["sub"] and current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get tasks
    tasks = await batch_service.get_job_tasks(
        job_id=job_id,
        status=status_filter,
        limit=limit
    )
    
    return [TaskResponse(**task) for task in tasks]


@router.post("/jobs/{job_id}/cancel")
async def cancel_batch_job(
    job_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Cancel a running or pending batch job
    """
    # Check job exists and user has access
    job = await batch_service.get_job_status(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    if job["submitted_by"] != current_user["sub"] and current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Cancel job
    success = await batch_service.cancel_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job cannot be cancelled (may already be completed)"
        )
    
    # Log audit event
    await audit_logger.log_event(
        action="BATCH_JOB_CANCELLED",
        user_id=current_user["sub"],
        resource_type="batch_job",
        resource_id=job_id,
        details={}
    )
    
    return {
        "message": f"Job {job_id} cancelled",
        "job_id": job_id,
        "status": "cancelled"
    }


@router.get("/jobs/{job_id}/results")
async def get_job_results(
    job_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get detailed results from a completed batch job
    """
    # Check job exists and user has access
    job = await batch_service.get_job_status(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    if job["submitted_by"] != current_user["sub"] and current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed (current status: {job['status']})"
        )
    
    # Get detailed results
    results = await batch_service.get_job_results(job_id)
    
    return {
        "job_id": job_id,
        "job_type": job["job_type"],
        "completed_at": job["completed_at"],
        "results": results
    }


@router.get("/queue/stats")
async def get_queue_statistics(current_user: dict = Depends(get_current_active_user)):
    """
    Get batch processing queue statistics
    """
    # Check admin/researcher permission
    if current_user.get("role") not in ["admin", "researcher"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or researcher access required"
        )
    
    stats = await batch_service.get_queue_stats()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "statistics": stats
    }


@router.delete("/jobs/{job_id}")
async def delete_batch_job(
    job_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Delete a batch job and all its tasks (admin only)
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    success = await batch_service.delete_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Log audit event
    await audit_logger.log_event(
        action="BATCH_JOB_DELETED",
        user_id=current_user["sub"],
        resource_type="batch_job",
        resource_id=job_id,
        details={}
    )
    
    return {"message": f"Job {job_id} deleted successfully"}
