"""
Advanced MCP Tools for Integrated Workflow and Provenance Tracking

New tools for:
- Complexity assessment
- Advanced workflow invocation  
- Provenance querying
- Workflow comparison
"""

import json
import logging
from typing import Any, Dict, List
from mcp.types import TextContent

logger = logging.getLogger(__name__)


def register_advanced_mcp_tools(server, lca_server_instance):
    """
    Register advanced MCP tools for workflow routing and provenance.

    Args:
        server: MCP Server instance
        lca_server_instance: LCAMCPServer instance for accessing shared resources
    """

    # ========================================
    # COMPLEXITY ASSESSMENT
    # ========================================

    @server.call_tool()
    async def assess_patient_complexity(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Assess patient complexity to determine optimal workflow.

        Analyzes patient data to classify complexity level:
        - SIMPLE: Early stage, good performance status, no complications
        - MODERATE: Intermediate stage or moderate comorbidities
        - COMPLEX: Advanced disease, poor PS, multiple comorbidities
        - CRITICAL: Very advanced disease with complex biomarker profiles

        Args:
            patient_data: Patient clinical data dictionary

        Returns:
            Complexity assessment with workflow recommendation
        """
        try:
            patient_data = arguments.get("patient_data", {})

            if not patient_data:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Missing patient_data",
                        "usage": "Provide patient clinical data for complexity assessment"
                    }, indent=2)
                )]

            # Get complexity assessment from service
            if hasattr(lca_server_instance, 'service'):
                import asyncio
                result = await lca_server_instance.service.assess_complexity(patient_data)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "patient_id": patient_data.get("patient_id", "unknown"),
                        "complexity": result["complexity"],
                        "recommended_workflow": result["recommended_workflow"],
                        "reason": result["reason"],
                        "factors": result["factors"],
                        "routing_decision": {
                            "use_advanced": result["recommended_workflow"] == "integrated",
                            "expected_agents": _get_expected_agents(result["recommended_workflow"]),
                            "estimated_time_ms": _estimate_processing_time(result["recommended_workflow"])
                        }
                    }, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Service not available",
                        "message": "Complexity assessment requires initialized service"
                    }, indent=2)
                )]

        except Exception as e:
            logger.error(f"Complexity assessment failed: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]

    # ========================================
    # ADVANCED WORKFLOW INVOCATION
    # ========================================

    @server.call_tool()
    async def run_advanced_integrated_workflow(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Run the advanced integrated workflow with all enhancements.

        Invokes the complete multi-agent system with:
        - Dynamic complexity-based routing
        - Specialized NSCLC/SCLC agents
        - Biomarker-driven precision medicine
        - Multi-agent negotiation
        - Advanced analytics (uncertainty, survival, clinical trials)
        - Full provenance tracking

        Args:
            patient_data: Patient clinical data
            persist: Whether to save results to Neo4j (default: False)

        Returns:
            Comprehensive analysis results with agent chain and provenance
        """
        try:
            patient_data = arguments.get("patient_data", {})
            persist = arguments.get("persist", False)

            if not patient_data:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Missing patient_data",
                        "usage": "Provide patient clinical data for analysis"
                    }, indent=2)
                )]

            # Force advanced workflow
            if hasattr(lca_server_instance, 'service'):
                result = await lca_server_instance.service.process_patient(
                    patient_data=patient_data,
                    use_ai_workflow=True,
                    force_advanced=True
                )

                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "patient_id": result.patient_id,
                        "workflow_type": result.workflow_type,
                        "complexity_level": result.complexity_level,
                        "execution_time_ms": result.execution_time_ms,
                        "recommendations": [
                            {
                                "treatment": rec.treatment_type,
                                "evidence_level": rec.evidence_level,
                                "confidence": rec.confidence_score,
                                "source": rec.rule_source
                            }
                            for rec in result.recommendations
                        ],
                        "mdt_summary": result.mdt_summary,
                        "provenance_record_id": result.provenance_record_id,
                        "patient_scenarios": result.patient_scenarios
                    }, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Service not available"
                    }, indent=2)
                )]

        except Exception as e:
            logger.error(f"Advanced workflow failed: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]

    # ========================================
    # PROVENANCE QUERYING
    # ========================================

    @server.call_tool()
    async def get_provenance_record(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Retrieve complete provenance record for audit/compliance.

        Provides full lineage tracking including:
        - Data sources and transformations
        - Agent execution chain
        - Model versions and parameters
        - Temporal evolution
        - Integrity checksums

        Args:
            record_id: Provenance record identifier

        Returns:
            Complete provenance graph with entities, activities, and agents
        """
        try:
            record_id = arguments.get("record_id")

            if not record_id:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Missing record_id",
                        "usage": "Provide record_id from previous workflow execution"
                    }, indent=2)
                )]

            if hasattr(lca_server_instance, 'service'):
                record = lca_server_instance.service.get_provenance_record(record_id)

                if not record:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": f"Record not found: {record_id}"
                        }, indent=2)
                    )]

                return [TextContent(
                    type="text",
                    text=json.dumps(record, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Service not available"
                    }, indent=2)
                )]

        except Exception as e:
            logger.error(f"Provenance query failed: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def query_patient_provenance_history(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Get all provenance records for a specific patient.

        Retrieves complete history of clinical decisions made for a patient,
        showing evolution of recommendations over time.

        Args:
            patient_id: Patient identifier

        Returns:
            List of all provenance records ordered by timestamp
        """
        try:
            patient_id = arguments.get("patient_id")

            if not patient_id:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Missing patient_id"
                    }, indent=2)
                )]

            if hasattr(lca_server_instance, 'service'):
                records = lca_server_instance.service.query_patient_provenance(patient_id)

                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "patient_id": patient_id,
                        "total_records": len(records),
                        "records": records
                    }, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Service not available"
                    }, indent=2)
                )]

        except Exception as e:
            logger.error(f"Patient provenance query failed: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]

    # ========================================
    # WORKFLOW COMPARISON
    # ========================================

    @server.call_tool()
    async def compare_workflow_outputs(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Compare basic vs advanced workflow outputs for same patient.

        Runs both workflows and shows differences in:
        - Recommendations
        - Confidence scores
        - Execution time
        - Agent involvement

        Useful for validating advanced workflow benefits for complex cases.

        Args:
            patient_data: Patient clinical data

        Returns:
            Side-by-side comparison of both workflows
        """
        try:
            patient_data = arguments.get("patient_data", {})

            if not patient_data:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Missing patient_data"
                    }, indent=2)
                )]

            if hasattr(lca_server_instance, 'service'):
                # Run basic workflow
                basic_result = await lca_server_instance.service.process_patient(
                    patient_data=patient_data,
                    use_ai_workflow=True,
                    force_advanced=False
                )

                # Run advanced workflow
                advanced_result = await lca_server_instance.service.process_patient(
                    patient_data=patient_data,
                    use_ai_workflow=True,
                    force_advanced=True
                )

                comparison = {
                    "patient_id": patient_data.get("patient_id"),
                    "basic_workflow": {
                        "type": basic_result.workflow_type,
                        "execution_time_ms": basic_result.execution_time_ms,
                        "recommendations_count": len(basic_result.recommendations),
                        "top_recommendation": basic_result.recommendations[0].treatment_type if basic_result.recommendations else None
                    },
                    "advanced_workflow": {
                        "type": advanced_result.workflow_type,
                        "execution_time_ms": advanced_result.execution_time_ms,
                        "recommendations_count": len(advanced_result.recommendations),
                        "top_recommendation": advanced_result.recommendations[0].treatment_type if advanced_result.recommendations else None,
                        "agent_chain": advanced_result.patient_scenarios
                    },
                    "differences": {
                        "time_delta_ms": advanced_result.execution_time_ms - basic_result.execution_time_ms,
                        "recommendations_match": (
                            basic_result.recommendations[0].treatment_type == advanced_result.recommendations[0].treatment_type
                            if basic_result.recommendations and advanced_result.recommendations
                            else False
                        )
                    }
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(comparison, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Service not available"
                    }, indent=2)
                )]

        except Exception as e:
            logger.error(f"Workflow comparison failed: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]

    logger.info("✓ Advanced MCP tools registered: complexity assessment, advanced workflow, provenance")


# Helper functions

def _get_expected_agents(workflow_type: str) -> List[str]:
    """Get list of agents expected for workflow type"""
    if workflow_type == "integrated":
        return [
            "IngestionAgent",
            "SemanticMappingAgent",
            "ClassificationAgent",
            "NSCLCAgent/SCLCAgent",
            "BiomarkerAgent",
            "ComorbidityAgent",
            "NegotiationProtocol",
            "UncertaintyQuantifier",
            "SurvivalAnalyzer",
            "ExplanationAgent"
        ]
    else:
        return [
            "ClassificationAgent",
            "RecommendationAgent",
            "ArgumentationAgent",
            "ExplanationAgent"
        ]


def _estimate_processing_time(workflow_type: str) -> int:
    """Estimate processing time in milliseconds"""
    if workflow_type == "integrated":
        return 45000  # ~45 seconds for full analytics
    else:
        return 20000  # ~20 seconds for basic workflow
