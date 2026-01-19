"""
Comprehensive MCP Tools for 2025-2026 Enhanced LCA System

Includes tools for:
- Authentication & Authorization
- Audit Logging
- Human-in-the-Loop Review
- Advanced Analytics
- RAG-enhanced Retrieval
- WebSocket Communication
- Guideline Versioning
- Batch Processing
- FHIR Integration
"""

import json
import logging
from typing import Any, Dict, List
from mcp.types import TextContent

logger = logging.getLogger(__name__)


def register_comprehensive_tools(server, lca_server_instance):
    """
    Register comprehensive MCP tools for all 2025-2026 enhancements.

    Args:
        server: MCP Server instance
        lca_server_instance: LCAMCPServer instance for accessing shared resources
    """

    # ========================================
    # AUTHENTICATION & AUTHORIZATION TOOLS
    # ========================================

    @server.call_tool()
    async def authenticate_user(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Authenticate a user and return JWT token.

        Args:
            username: User username
            password: User password

        Returns:
            JWT token and user info on success
        """
        try:
            from src.services.auth_service import auth_service

            username = arguments.get("username")
            password = arguments.get("password")

            token, user_info = auth_service.authenticate_user(username, password)

            result = {
                "status": "success",
                "token": token,
                "user": {
                    "id": user_info["user_id"],
                    "username": user_info["username"],
                    "role": user_info["role"],
                    "permissions": user_info["permissions"]
                }
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def create_user(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Create a new user account.

        Args:
            username: Unique username
            password: User password
            role: User role (admin, clinician, researcher, viewer)
            email: User email (optional)
            full_name: Full name (optional)

        Returns:
            User creation confirmation
        """
        try:
            from src.services.auth_service import auth_service

            user_data = {
                "username": arguments.get("username"),
                "password": arguments.get("password"),
                "role": arguments.get("role", "viewer"),
                "email": arguments.get("email"),
                "full_name": arguments.get("full_name")
            }

            user_id = auth_service.create_user(user_data)

            result = {
                "status": "success",
                "user_id": user_id,
                "message": f"User {user_data['username']} created successfully"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    # ========================================
    # AUDIT LOGGING TOOLS
    # ========================================

    @server.call_tool()
    async def query_audit_logs(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Query audit logs with filtering options.

        Args:
            user_id: Filter by user ID
            action: Filter by action type
            resource_type: Filter by resource type
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            limit: Maximum results (default: 100)

        Returns:
            Filtered audit log entries
        """
        try:
            from src.services.audit_service import audit_logger

            filters = {
                "user_id": arguments.get("user_id"),
                "action": arguments.get("action"),
                "resource_type": arguments.get("resource_type"),
                "start_date": arguments.get("start_date"),
                "end_date": arguments.get("end_date"),
                "limit": arguments.get("limit", 100)
            }

            logs = audit_logger.query_logs(**filters)

            result = {
                "status": "success",
                "count": len(logs),
                "logs": [
                    {
                        "id": log.id,
                        "timestamp": log.timestamp.isoformat(),
                        "user_id": log.user_id,
                        "username": log.username,
                        "user_role": log.user_role,
                        "action": log.action.value,
                        "resource_type": log.resource_type,
                        "resource_id": log.resource_id,
                        "severity": log.severity.value,
                        "details": log.details
                    }
                    for log in logs
                ]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def export_audit_logs(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Export audit logs to file.

        Args:
            format: Export format (json, csv)
            filename: Output filename
            filters: Query filters (same as query_audit_logs)

        Returns:
            Export confirmation with file path
        """
        try:
            from src.services.audit_service import audit_logger

            export_format = arguments.get("format", "json")
            filename = arguments.get("filename", f"audit_export.{export_format}")
            filters = arguments.get("filters", {})

            file_path = audit_logger.export_logs(
                export_format=export_format,
                filename=filename,
                **filters
            )

            result = {
                "status": "success",
                "message": f"Audit logs exported to {file_path}",
                "format": export_format,
                "file_path": file_path
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    # ========================================
    # HUMAN-IN-THE-LOOP TOOLS
    # ========================================

    @server.call_tool()
    async def submit_for_review(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Submit a case for human review.

        Args:
            patient_id: Patient identifier
            analysis_result: Analysis results requiring review
            review_reason: Reason for review (low_confidence, conflict, complex_case)
            priority: Review priority (low, medium, high, critical)

        Returns:
            Review case ID and status
        """
        try:
            from src.services.hitl_service import hitl_service

            review_case = {
                "patient_id": arguments.get("patient_id"),
                "analysis_result": arguments.get("analysis_result"),
                "review_reason": arguments.get("review_reason", "manual_review"),
                "priority": arguments.get("priority", "medium"),
                "submitted_by": "mcp_tool"
            }

            case_id = hitl_service.submit_for_review(review_case)

            result = {
                "status": "success",
                "case_id": case_id,
                "message": "Case submitted for human review",
                "priority": review_case["priority"]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def get_pending_reviews(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Get pending review cases.

        Args:
            reviewer_id: Specific reviewer (optional)
            priority: Filter by priority
            limit: Maximum results (default: 50)

        Returns:
            List of pending review cases
        """
        try:
            from src.services.hitl_service import hitl_service

            filters = {
                "reviewer_id": arguments.get("reviewer_id"),
                "priority": arguments.get("priority"),
                "limit": arguments.get("limit", 50)
            }

            reviews = hitl_service.get_pending_reviews(**filters)

            result = {
                "status": "success",
                "count": len(reviews),
                "reviews": [
                    {
                        "case_id": review.case_id,
                        "patient_id": review.patient_id,
                        "priority": review.priority.value,
                        "review_reason": review.review_reason,
                        "submitted_at": review.submitted_at.isoformat(),
                        "status": review.status.value
                    }
                    for review in reviews
                ]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    # ========================================
    # ADVANCED ANALYTICS TOOLS
    # ========================================

    @server.call_tool()
    async def generate_survival_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Generate survival analysis for patient cohorts.

        Args:
            patient_criteria: Criteria to select patient cohort
            time_points: Time points for analysis (months)
            stratification_factors: Factors to stratify by

        Returns:
            Survival curves and statistics
        """
        try:
            from src.services.analytics_service import analytics_service

            analysis_request = {
                "patient_criteria": arguments.get("patient_criteria", {}),
                "time_points": arguments.get("time_points", [12, 24, 36, 60]),
                "stratification_factors": arguments.get("stratification_factors", [])
            }

            results = analytics_service.generate_survival_analysis(analysis_request)

            result = {
                "status": "success",
                "analysis_type": "survival_analysis",
                "cohort_size": results.get("cohort_size", 0),
                "median_survival": results.get("median_survival"),
                "survival_rates": results.get("survival_rates", {}),
                "stratified_results": results.get("stratified_results", {})
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def analyze_treatment_outcomes(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Analyze treatment outcomes and comparative effectiveness.

        Args:
            treatment_a: First treatment to compare
            treatment_b: Second treatment to compare
            patient_criteria: Patient selection criteria
            outcome_measures: Outcomes to analyze

        Returns:
            Comparative treatment analysis
        """
        try:
            from src.services.analytics_service import analytics_service

            analysis_request = {
                "treatment_a": arguments.get("treatment_a"),
                "treatment_b": arguments.get("treatment_b"),
                "patient_criteria": arguments.get("patient_criteria", {}),
                "outcome_measures": arguments.get("outcome_measures", ["survival", "response_rate"])
            }

            results = analytics_service.analyze_treatment_outcomes(analysis_request)

            result = {
                "status": "success",
                "analysis_type": "treatment_comparison",
                "treatments_compared": [analysis_request["treatment_a"], analysis_request["treatment_b"]],
                "sample_sizes": results.get("sample_sizes", {}),
                "outcome_results": results.get("outcome_results", {}),
                "statistical_significance": results.get("statistical_significance", {})
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    # ========================================
    # RAG-ENHANCED RETRIEVAL TOOLS
    # ========================================

    @server.call_tool()
    async def retrieve_guidelines(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Retrieve relevant clinical guidelines using RAG.

        Args:
            query: Natural language query about clinical guidelines
            patient_context: Patient-specific context
            max_results: Maximum results to return (default: 5)

        Returns:
            Relevant guideline excerpts with relevance scores
        """
        try:
            from src.services.rag_service import rag_service

            query = arguments.get("query")
            patient_context = arguments.get("patient_context", {})
            max_results = arguments.get("max_results", 5)

            results = rag_service.retrieve_guidelines(
                query=query,
                patient_context=patient_context,
                max_results=max_results
            )

            result = {
                "status": "success",
                "query": query,
                "results_count": len(results),
                "guidelines": [
                    {
                        "guideline_id": r.get("guideline_id"),
                        "title": r.get("title"),
                        "excerpt": r.get("excerpt"),
                        "relevance_score": r.get("relevance_score"),
                        "source": r.get("source"),
                        "evidence_level": r.get("evidence_level")
                    }
                    for r in results
                ]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def find_similar_cases(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Find similar clinical cases using semantic search.

        Args:
            patient_data: Current patient data
            max_results: Maximum similar cases (default: 10)
            similarity_threshold: Minimum similarity score (default: 0.7)

        Returns:
            Similar cases with similarity scores and outcomes
        """
        try:
            from src.services.rag_service import rag_service

            patient_data = arguments.get("patient_data", {})
            max_results = arguments.get("max_results", 10)
            similarity_threshold = arguments.get("similarity_threshold", 0.7)

            similar_cases = rag_service.find_similar_cases(
                patient_data=patient_data,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )

            result = {
                "status": "success",
                "patient_id": patient_data.get("patient_id"),
                "similar_cases_count": len(similar_cases),
                "similar_cases": [
                    {
                        "case_id": case.get("case_id"),
                        "patient_id": case.get("patient_id"),
                        "similarity_score": case.get("similarity_score"),
                        "stage": case.get("stage"),
                        "treatment": case.get("treatment"),
                        "outcome": case.get("outcome"),
                        "survival_months": case.get("survival_months")
                    }
                    for case in similar_cases
                ]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    # ========================================
    # GUIDELINE VERSIONING TOOLS
    # ========================================

    @server.call_tool()
    async def create_guideline_version(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Create a new version of clinical guidelines.

        Args:
            base_version: Version to base new version on
            changes: List of changes made
            author: Author of changes
            description: Description of changes

        Returns:
            New version information
        """
        try:
            from src.services.version_service import version_service

            version_data = {
                "base_version": arguments.get("base_version"),
                "changes": arguments.get("changes", []),
                "author": arguments.get("author", "system"),
                "description": arguments.get("description", "")
            }

            new_version = version_service.create_version(version_data)

            result = {
                "status": "success",
                "version_id": new_version.version_id,
                "version_number": new_version.version_number,
                "base_version": new_version.base_version,
                "created_at": new_version.created_at.isoformat(),
                "author": new_version.author
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def get_guideline_versions(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Get available guideline versions.

        Args:
            active_only: Return only active versions (default: true)
            limit: Maximum versions to return (default: 20)

        Returns:
            List of guideline versions
        """
        try:
            from src.services.version_service import version_service

            active_only = arguments.get("active_only", True)
            limit = arguments.get("limit", 20)

            versions = version_service.get_versions(active_only=active_only, limit=limit)

            result = {
                "status": "success",
                "count": len(versions),
                "versions": [
                    {
                        "version_id": v.version_id,
                        "version_number": v.version_number,
                        "status": v.status.value,
                        "created_at": v.created_at.isoformat(),
                        "author": v.author,
                        "description": v.description
                    }
                    for v in versions
                ]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    # ========================================
    # BATCH PROCESSING TOOLS
    # ========================================

    @server.call_tool()
    async def submit_batch_job(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Submit a batch processing job.

        Args:
            job_type: Type of batch job (analysis, export, import)
            job_config: Job-specific configuration
            priority: Job priority (low, medium, high)

        Returns:
            Job ID and status
        """
        try:
            from src.services.batch_service import batch_service

            job_data = {
                "job_type": arguments.get("job_type"),
                "job_config": arguments.get("job_config", {}),
                "priority": arguments.get("priority", "medium"),
                "submitted_by": "mcp_tool"
            }

            job_id = batch_service.submit_job(job_data)

            result = {
                "status": "success",
                "job_id": job_id,
                "job_type": job_data["job_type"],
                "priority": job_data["priority"],
                "message": "Batch job submitted successfully"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def get_batch_job_status(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Get status of batch job.

        Args:
            job_id: Batch job ID

        Returns:
            Job status and progress
        """
        try:
            from src.services.batch_service import batch_service

            job_id = arguments.get("job_id")
            status = batch_service.get_job_status(job_id)

            result = {
                "status": "success",
                "job_id": job_id,
                "job_status": status.get("status"),
                "progress": status.get("progress", 0),
                "created_at": status.get("created_at"),
                "completed_at": status.get("completed_at"),
                "results": status.get("results", {})
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    # ========================================
    # FHIR INTEGRATION TOOLS
    # ========================================

    @server.call_tool()
    async def import_fhir_bundle(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Import patient data from FHIR Bundle.

        Args:
            bundle: FHIR R4 Bundle JSON
            run_analysis: Whether to run analysis after import (default: true)

        Returns:
            Import results and analysis status
        """
        try:
            from src.services.fhir_service import fhir_service

            bundle = arguments.get("bundle")
            run_analysis = arguments.get("run_analysis", True)

            patient_data = fhir_service.import_bundle(bundle)

            result = {
                "status": "success",
                "patient_id": patient_data.get("patient_id"),
                "imported_resources": {
                    "conditions": len(patient_data.get("conditions", [])),
                    "observations": len(patient_data.get("observations", [])),
                    "medications": len(patient_data.get("medications", []))
                },
                "analysis_requested": run_analysis
            }

            if run_analysis:
                # Note: In real implementation, this would trigger background analysis
                result["analysis_status"] = "would_be_started_in_background"

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def export_fhir_bundle(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Export patient data as FHIR Bundle.

        Args:
            patient_id: Patient identifier
            include_analysis: Include analysis results (default: true)

        Returns:
            FHIR Bundle JSON
        """
        try:
            from src.services.fhir_service import fhir_service

            patient_id = arguments.get("patient_id")
            include_analysis = arguments.get("include_analysis", True)

            # Mock patient data - in real implementation, fetch from database
            patient_data = {
                "patient_id": patient_id,
                "name": "Mock Patient",
                "age": 65,
                "gender": "male",
                "diagnosis": "Lung adenocarcinoma",
                "stage": "IIIA",
                "biomarkers": {"EGFR": "positive"}
            }

            analysis_result = {
                "recommendation": {
                    "treatment": "Chemoradiation",
                    "rationale": "Stage IIIA NSCLC"
                }
            } if include_analysis else {}

            bundle = fhir_service.export_bundle(patient_data, analysis_result)

            result = {
                "status": "success",
                "patient_id": patient_id,
                "bundle": bundle,
                "resource_count": len(bundle.get("entry", [])),
                "include_analysis": include_analysis
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    # ========================================
    # WEBSOCKET COMMUNICATION TOOLS
    # ========================================

    @server.call_tool()
    async def send_websocket_notification(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Send real-time notification via WebSocket.

        Args:
            channel: WebSocket channel
            message: Notification message
            user_id: Target user (optional, broadcasts if not specified)

        Returns:
            Notification send confirmation
        """
        try:
            from src.services.websocket_service import websocket_service

            notification = {
                "channel": arguments.get("channel"),
                "message": arguments.get("message"),
                "user_id": arguments.get("user_id"),
                "timestamp": "now"
            }

            success = websocket_service.send_notification(notification)

            result = {
                "status": "success" if success else "failed",
                "channel": notification["channel"],
                "message": "Notification sent" if success else "Failed to send notification",
                "target_user": notification.get("user_id")
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def get_websocket_channels(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Get active WebSocket channels and connection counts.

        Returns:
            Channel status information
        """
        try:
            from src.services.websocket_service import websocket_service

            channels = websocket_service.get_channel_status()

            result = {
                "status": "success",
                "channels": channels
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]