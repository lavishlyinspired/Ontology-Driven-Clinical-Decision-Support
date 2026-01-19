"""
FHIR API Routes - Import/Export endpoints for EHR integration
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..services.fhir_service import fhir_service
from ..agents.integrated_workflow import integrated_workflow
from ..services.audit_service import audit_logger, AuditAction

router = APIRouter(prefix="/fhir", tags=["FHIR Integration"])


# ==================== Request Models ====================

class FHIRBundleImportRequest(BaseModel):
    """Request model for importing FHIR Bundle."""
    bundle: Dict[str, Any] = Field(..., description="FHIR R4 Bundle containing patient data")
    analysis_options: Optional[Dict[str, Any]] = Field(
        default={"use_ai_workflow": True, "persist": True},
        description="Options for analysis after import"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "bundle": {
                    "resourceType": "Bundle",
                    "type": "collection",
                    "entry": [
                        {
                            "resource": {
                                "resourceType": "Patient",
                                "id": "patient-123",
                                "name": [{"family": "Doe", "given": ["John"]}],
                                "gender": "male",
                                "birthDate": "1955-01-15"
                            }
                        }
                    ]
                },
                "analysis_options": {
                    "use_ai_workflow": True,
                    "persist": True
                }
            }
        }


class FHIRResourceImportRequest(BaseModel):
    """Request model for importing individual FHIR resources."""
    patient: Optional[Dict[str, Any]] = Field(None, description="FHIR Patient resource")
    conditions: Optional[List[Dict[str, Any]]] = Field(default=[], description="FHIR Condition resources")
    observations: Optional[List[Dict[str, Any]]] = Field(default=[], description="FHIR Observation resources")
    medications: Optional[List[Dict[str, Any]]] = Field(default=[], description="FHIR MedicationStatement resources")
    analysis_options: Optional[Dict[str, Any]] = Field(
        default={"use_ai_workflow": True, "persist": True},
        description="Options for analysis after import"
    )


class FHIRWebhookRequest(BaseModel):
    """Request model for FHIR webhooks from EHR systems."""
    resource: Dict[str, Any] = Field(..., description="Updated FHIR resource")
    event_type: str = Field(..., description="Type of event (create, update, delete)")
    source_system: str = Field(..., description="EHR system that sent the webhook")
    patient_id: Optional[str] = Field(None, description="Patient ID if available")


# ==================== FHIR Import Endpoints ====================

@router.post("/import/bundle")
async def import_fhir_bundle(
    request: FHIRBundleImportRequest,
    background_tasks: BackgroundTasks
):
    """
    Import FHIR Bundle and analyze patient.

    Converts FHIR R4 Bundle to LCA format, then runs full analysis workflow.
    Supports Patient, Condition, Observation, MedicationStatement resources.
    """
    try:
        # Import FHIR bundle to LCA format
        patient_data = fhir_service.import_bundle(request.bundle)

        # Add metadata
        patient_data['import_source'] = 'fhir_bundle'
        patient_data['import_timestamp'] = datetime.now().isoformat()

        # Audit the import
        audit_logger.log(
            user_id="system",
            username="FHIR Import",
            user_role="system",
            action=AuditAction.DATA_IMPORT,
            resource_type="patient",
            resource_id=patient_data.get('patient_id'),
            details={
                "source": "fhir_bundle",
                "resource_count": len(request.bundle.get('entry', [])),
                "analysis_requested": request.analysis_options.get('use_ai_workflow', True)
            }
        )

        # Run analysis if requested
        if request.analysis_options.get('use_ai_workflow', True):
            # Run in background for long-running analysis
            background_tasks.add_task(
                process_fhir_analysis_background,
                patient_data,
                request.analysis_options
            )

            return {
                "status": "accepted",
                "message": "FHIR bundle imported successfully. Analysis started in background.",
                "patient_id": patient_data.get('patient_id'),
                "analysis_status": "running"
            }

        return {
            "status": "success",
            "message": "FHIR bundle imported successfully",
            "patient_data": patient_data
        }

    except Exception as e:
        # Audit the failure
        audit_logger.log(
            user_id="system",
            username="FHIR Import",
            user_role="system",
            action=AuditAction.DATA_IMPORT,
            resource_type="patient",
            resource_id="unknown",
            details={
                "error": str(e),
                "source": "fhir_bundle"
            }
        )

        raise HTTPException(
            status_code=400,
            detail=f"Failed to import FHIR bundle: {str(e)}"
        )


@router.post("/import/resources")
async def import_fhir_resources(
    request: FHIRResourceImportRequest,
    background_tasks: BackgroundTasks
):
    """
    Import individual FHIR resources and analyze patient.

    Allows importing Patient, Conditions, Observations, etc. separately.
    """
    try:
        patient_data = {}

        # Import patient resource
        if request.patient:
            patient_data.update(fhir_service.import_patient(request.patient))

        # Import conditions
        for condition in request.conditions or []:
            condition_data = fhir_service.import_condition(condition)
            if 'conditions' not in patient_data:
                patient_data['conditions'] = []
            patient_data['conditions'].append(condition_data)

        # Import observations
        for observation in request.observations or []:
            obs_data = fhir_service.import_observation(observation)
            if 'observations' not in patient_data:
                patient_data['observations'] = []
            patient_data['observations'].append(obs_data)

        # Add metadata
        patient_data['import_source'] = 'fhir_resources'
        patient_data['import_timestamp'] = datetime.now().isoformat()

        # Audit the import
        audit_logger.log(
            user_id="system",
            username="FHIR Import",
            user_role="system",
            action=AuditAction.DATA_IMPORT,
            resource_type="patient",
            resource_id=patient_data.get('patient_id'),
            details={
                "source": "fhir_resources",
                "conditions_count": len(request.conditions or []),
                "observations_count": len(request.observations or []),
                "analysis_requested": request.analysis_options.get('use_ai_workflow', True)
            }
        )

        # Run analysis if requested
        if request.analysis_options.get('use_ai_workflow', True):
            background_tasks.add_task(
                process_fhir_analysis_background,
                patient_data,
                request.analysis_options
            )

            return {
                "status": "accepted",
                "message": "FHIR resources imported successfully. Analysis started in background.",
                "patient_id": patient_data.get('patient_id'),
                "analysis_status": "running"
            }

        return {
            "status": "success",
            "message": "FHIR resources imported successfully",
            "patient_data": patient_data
        }

    except Exception as e:
        audit_logger.log(
            user_id="system",
            username="FHIR Import",
            user_role="system",
            action=AuditAction.DATA_IMPORT,
            resource_type="patient",
            resource_id="unknown",
            details={
                "error": str(e),
                "source": "fhir_resources"
            }
        )

        raise HTTPException(
            status_code=400,
            detail=f"Failed to import FHIR resources: {str(e)}"
        )


@router.post("/webhook")
async def handle_fhir_webhook(
    request: FHIRWebhookRequest,
    background_tasks: BackgroundTasks
):
    """
    Handle FHIR webhooks from EHR systems.

    Processes real-time updates from connected EHR systems.
    Supports create, update, delete events for patient data.
    """
    try:
        resource_type = request.resource.get('resourceType')
        resource_id = request.resource.get('id')

        # Audit the webhook
        audit_logger.log(
            user_id="system",
            username=f"EHR Webhook ({request.source_system})",
            user_role="system",
            action=AuditAction.DATA_UPDATE,
            resource_type=resource_type.lower(),
            resource_id=resource_id,
            details={
                "event_type": request.event_type,
                "source_system": request.source_system,
                "resource_type": resource_type
            }
        )

        # Process based on resource type and event
        if resource_type == 'Patient':
            if request.event_type in ['create', 'update']:
                patient_data = fhir_service.import_patient(request.resource)
                # Update patient record in system
                background_tasks.add_task(
                    update_patient_from_webhook,
                    patient_data,
                    request.event_type
                )

            elif request.event_type == 'delete':
                # Handle patient deletion
                background_tasks.add_task(
                    delete_patient_from_webhook,
                    resource_id
                )

        elif resource_type == 'Condition':
            if request.event_type in ['create', 'update']:
                condition_data = fhir_service.import_condition(request.resource)
                background_tasks.add_task(
                    update_condition_from_webhook,
                    condition_data,
                    request.patient_id,
                    request.event_type
                )

        elif resource_type == 'Observation':
            if request.event_type in ['create', 'update']:
                observation_data = fhir_service.import_observation(request.resource)
                background_tasks.add_task(
                    update_observation_from_webhook,
                    observation_data,
                    request.patient_id,
                    request.event_type
                )

        return {
            "status": "accepted",
            "message": f"Processed {request.event_type} event for {resource_type} {resource_id}"
        }

    except Exception as e:
        audit_logger.log(
            user_id="system",
            username="EHR Webhook",
            user_role="system",
            action=AuditAction.DATA_UPDATE,
            resource_type="unknown",
            resource_id="unknown",
            details={
                "error": str(e),
                "source_system": request.source_system,
                "event_type": request.event_type
            }
        )

        raise HTTPException(
            status_code=400,
            detail=f"Failed to process FHIR webhook: {str(e)}"
        )


# ==================== FHIR Export Endpoints ====================

@router.post("/export/bundle/{patient_id}")
async def export_fhir_bundle(
    patient_id: str,
    include_analysis: bool = True
):
    """
    Export patient data and analysis as FHIR Bundle.

    Includes Patient, Condition, Observations, and CarePlan resources.
    """
    try:
        # Get patient data from database (simplified - would use actual DB)
        patient_data = await get_patient_data(patient_id)

        if not patient_data:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Get latest analysis if requested
        analysis_result = None
        if include_analysis:
            analysis_result = await get_latest_analysis(patient_id)

        # Export to FHIR bundle
        fhir_bundle = fhir_service.export_bundle(patient_data, analysis_result or {})

        # Audit the export
        audit_logger.log(
            user_id="system",
            username="FHIR Export",
            user_role="system",
            action=AuditAction.DATA_EXPORT,
            resource_type="patient",
            resource_id=patient_id,
            details={
                "format": "fhir_bundle",
                "include_analysis": include_analysis,
                "resource_count": len(fhir_bundle.get('entry', []))
            }
        )

        return {
            "status": "success",
            "bundle": fhir_bundle,
            "resource_count": len(fhir_bundle.get('entry', []))
        }

    except Exception as e:
        audit_logger.log(
            user_id="system",
            username="FHIR Export",
            user_role="system",
            action=AuditAction.DATA_EXPORT,
            resource_type="patient",
            resource_id=patient_id,
            details={
                "error": str(e),
                "format": "fhir_bundle"
            }
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to export FHIR bundle: {str(e)}"
        )


# ==================== Background Tasks ====================

async def process_fhir_analysis_background(
    patient_data: Dict[str, Any],
    analysis_options: Dict[str, Any]
):
    """Background task to process FHIR-imported patient through analysis workflow."""
    try:
        # Run integrated workflow
        result = await integrated_workflow.analyze_patient_comprehensive(
            patient_data,
            persist=analysis_options.get('persist', True)
        )

        # Audit successful analysis
        audit_logger.log(
            user_id="system",
            username="Background Analysis",
            user_role="system",
            action=AuditAction.ANALYSIS_RUN,
            resource_type="patient",
            resource_id=patient_data.get('patient_id'),
            details={
                "source": "fhir_import",
                "success": True,
                "confidence": result.get('overall_confidence', 0)
            }
        )

        print(f"✅ Background analysis completed for patient {patient_data.get('patient_id')}")

    except Exception as e:
        # Audit failed analysis
        audit_logger.log(
            user_id="system",
            username="Background Analysis",
            user_role="system",
            action=AuditAction.ANALYSIS_RUN,
            resource_type="patient",
            resource_id=patient_data.get('patient_id'),
            details={
                "source": "fhir_import",
                "success": False,
                "error": str(e)
            }
        )

        print(f"❌ Background analysis failed for patient {patient_data.get('patient_id')}: {e}")


async def update_patient_from_webhook(
    patient_data: Dict[str, Any],
    event_type: str
):
    """Update patient record from EHR webhook."""
    # Implementation would update patient in database
    print(f"📡 Updated patient {patient_data.get('patient_id')} from EHR webhook ({event_type})")


async def delete_patient_from_webhook(patient_id: str):
    """Delete patient record from EHR webhook."""
    # Implementation would mark patient as deleted in database
    print(f"📡 Deleted patient {patient_id} from EHR webhook")


async def update_condition_from_webhook(
    condition_data: Dict[str, Any],
    patient_id: str,
    event_type: str
):
    """Update condition from EHR webhook."""
    print(f"📡 Updated condition for patient {patient_id} from EHR webhook ({event_type})")


async def update_observation_from_webhook(
    observation_data: Dict[str, Any],
    patient_id: str,
    event_type: str
):
    """Update observation from EHR webhook."""
    print(f"📡 Updated observation for patient {patient_id} from EHR webhook ({event_type})")


# ==================== Helper Functions ====================

async def get_patient_data(patient_id: str) -> Optional[Dict[str, Any]]:
    """Get patient data from database (placeholder)."""
    # In real implementation, this would query the database
    # For now, return None to indicate not found
    return None


async def get_latest_analysis(patient_id: str) -> Optional[Dict[str, Any]]:
    """Get latest analysis for patient (placeholder)."""
    # In real implementation, this would query analysis history
    return None