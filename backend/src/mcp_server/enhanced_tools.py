"""
Enhanced MCP Tools for Advanced Clinical Decision Support

New tools integrating 2025 improvements:
- Graph algorithms for patient similarity
- Temporal analysis for disease progression
- Biomarker-driven precision medicine
- Uncertainty quantification
- LOINC laboratory test integration
"""

import json
import logging
from typing import Any, Dict, List
from mcp.types import TextContent

# Import new modules
from ..db.graph_algorithms import Neo4jGraphAlgorithms
from ..db.temporal_analyzer import TemporalAnalyzer
from ..agents.biomarker_agent import BiomarkerAgent, BiomarkerProfile
from ..analytics.uncertainty_quantifier import UncertaintyQuantifier
from ..ontology.loinc_integrator import LOINCIntegrator

logger = logging.getLogger(__name__)


def register_enhanced_tools(server, lca_server_instance):
    """
    Register enhanced MCP tools for advanced clinical decision support.

    Args:
        server: MCP Server instance
        lca_server_instance: LCAMCPServer instance for accessing shared resources
    """

    # Initialize advanced tools
    graph_algorithms = Neo4jGraphAlgorithms()
    temporal_analyzer = TemporalAnalyzer()
    biomarker_agent = BiomarkerAgent()
    uncertainty_quantifier = UncertaintyQuantifier()
    loinc_integrator = LOINCIntegrator()

    # ========================================
    # GRAPH ALGORITHMS TOOLS
    # ========================================

    @server.call_tool()
    async def find_similar_patients_graph(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Find similar patients using advanced graph algorithms.

        Args:
            patient_id: Patient identifier
            k: Number of similar patients to find (default: 10)
            min_similarity: Minimum similarity threshold 0-1 (default: 0.7)
        """
        try:
            patient_id = arguments.get("patient_id")
            k = arguments.get("k", 10)
            min_similarity = arguments.get("min_similarity", 0.7)

            similar_patients = graph_algorithms.find_similar_patients_graph_based(
                patient_id=patient_id,
                k=k,
                min_similarity=min_similarity
            )

            result = {
                "status": "success",
                "patient_id": patient_id,
                "similar_patients_count": len(similar_patients),
                "similar_patients": [
                    {
                        "patient_id": p.patient_id,
                        "name": p.name,
                        "similarity_score": f"{p.similarity_score:.2f}",
                        "stage": p.tnm_stage,
                        "histology": p.histology_type,
                        "treatment": p.treatment_received,
                        "outcome": p.outcome,
                        "survival_days": p.survival_days
                    }
                    for p in similar_patients
                ]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def detect_treatment_communities(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Detect communities of patients with similar treatment patterns.

        Args:
            resolution: Community detection resolution (default: 1.0)
        """
        try:
            resolution = arguments.get("resolution", 1.0)

            communities = graph_algorithms.detect_treatment_communities(resolution=resolution)

            result = {
                "status": "success",
                "communities_found": len(communities),
                "communities": [
                    {
                        "community_id": comm_id,
                        "size": data["size"],
                        "patients": data["patients"][:10]  # Show first 10
                    }
                    for comm_id, data in list(communities.items())[:20]  # Show top 20 communities
                ]
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def find_optimal_treatment_paths(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Find optimal treatment sequences based on historical outcomes.

        Args:
            patient_id: Patient identifier
            target_outcome: Desired outcome (default: "Complete Response")
            max_path_length: Maximum treatment sequence length (default: 5)
        """
        try:
            patient_id = arguments.get("patient_id")
            target_outcome = arguments.get("target_outcome", "Complete Response")
            max_path_length = arguments.get("max_path_length", 5)

            paths = graph_algorithms.find_optimal_treatment_paths(
                patient_id=patient_id,
                target_outcome=target_outcome,
                max_path_length=max_path_length
            )

            result = {
                "status": "success",
                "patient_id": patient_id,
                "target_outcome": target_outcome,
                "paths_found": len(paths),
                "optimal_paths": paths
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    # ========================================
    # TEMPORAL ANALYSIS TOOLS
    # ========================================

    @server.call_tool()
    async def analyze_disease_progression(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Analyze disease progression over time for a patient.

        Args:
            patient_id: Patient identifier
        """
        try:
            patient_id = arguments.get("patient_id")

            progression = temporal_analyzer.analyze_disease_progression(patient_id)

            result = {
                "status": "success",
                "patient_id": patient_id,
                "total_assessments": progression.get("total_inferences", 0),
                "progression_events": progression.get("progression_events", 0),
                "current_scenario": progression.get("current_scenario"),
                "timeline": progression.get("timeline", [])
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def identify_intervention_windows(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Identify critical intervention windows based on temporal patterns.

        Args:
            patient_id: Patient identifier
            lookahead_days: Days to look ahead (default: 90)
        """
        try:
            patient_id = arguments.get("patient_id")
            lookahead_days = arguments.get("lookahead_days", 90)

            windows = temporal_analyzer.identify_intervention_windows(
                patient_id=patient_id,
                lookahead_days=lookahead_days
            )

            result = {
                "status": "success",
                **windows
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    # ========================================
    # BIOMARKER ANALYSIS TOOLS
    # ========================================

    @server.call_tool()
    async def analyze_biomarkers(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Get biomarker-driven treatment recommendations.

        Args:
            patient_data: Patient clinical data with biomarker results
        """
        try:
            patient_data = arguments.get("patient_data", arguments)

            # Convert to PatientFactWithCodes (simplified for demo)
            from ..db.models import PatientFactWithCodes

            patient = PatientFactWithCodes(
                patient_id=patient_data.get("patient_id", "UNKNOWN"),
                name=patient_data.get("name", "Unknown"),
                sex=patient_data.get("sex", "U"),
                age_at_diagnosis=patient_data.get("age", 0),
                tnm_stage=patient_data.get("tnm_stage", "IV"),
                histology_type=patient_data.get("histology_type", "Adenocarcinoma"),
                performance_status=patient_data.get("performance_status", 1),
                laterality=patient_data.get("laterality", "Right"),
                diagnosis="Lung Cancer",
                comorbidities=patient_data.get("comorbidities", [])
            )

            # Create biomarker profile
            biomarker_profile = BiomarkerProfile(
                egfr_mutation=patient_data.get("egfr_mutation"),
                egfr_mutation_type=patient_data.get("egfr_mutation_type"),
                alk_rearrangement=patient_data.get("alk_rearrangement"),
                ros1_rearrangement=patient_data.get("ros1_rearrangement"),
                braf_mutation=patient_data.get("braf_mutation"),
                pdl1_tps=patient_data.get("pdl1_tps"),
                met_exon14_skipping=patient_data.get("met_exon14")
            )

            # Get biomarker-driven recommendation
            proposal = biomarker_agent.execute(patient, biomarker_profile)

            result = {
                "status": "success",
                "patient_id": patient.patient_id,
                "biomarker_recommendation": {
                    "treatment": proposal.treatment,
                    "confidence": f"{proposal.confidence:.2%}",
                    "evidence_level": proposal.evidence_level,
                    "intent": proposal.treatment_intent,
                    "rationale": proposal.rationale,
                    "guideline": proposal.guideline_reference,
                    "risk_score": proposal.risk_score,
                    "contraindications": proposal.contraindications,
                    "expected_benefit": proposal.expected_benefit
                }
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    @server.call_tool()
    async def recommend_biomarker_testing(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Recommend which biomarker tests should be ordered.

        Args:
            patient_data: Patient clinical data
        """
        try:
            patient_data = arguments.get("patient_data", arguments)

            from ..db.models import PatientFactWithCodes

            patient = PatientFactWithCodes(
                patient_id=patient_data.get("patient_id", "UNKNOWN"),
                name=patient_data.get("name", "Unknown"),
                sex=patient_data.get("sex", "U"),
                age_at_diagnosis=patient_data.get("age", 0),
                tnm_stage=patient_data.get("tnm_stage", "IV"),
                histology_type=patient_data.get("histology_type", "Adenocarcinoma"),
                performance_status=patient_data.get("performance_status", 1),
                laterality=patient_data.get("laterality", "Right"),
                diagnosis="Lung Cancer",
                comorbidities=[]
            )

            recommended_tests = biomarker_agent.recommend_biomarker_testing(patient)

            result = {
                "status": "success",
                "patient_id": patient.patient_id,
                "histology": patient.histology_type,
                "stage": patient.tnm_stage,
                "recommended_tests": recommended_tests,
                "rationale": "Comprehensive biomarker profiling for precision medicine"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    # ========================================
    # UNCERTAINTY QUANTIFICATION TOOLS
    # ========================================

    @server.call_tool()
    async def quantify_uncertainty(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Quantify uncertainty in treatment recommendation.

        Args:
            patient_id: Patient identifier
            treatment: Treatment name
        """
        try:
            patient_id = arguments.get("patient_id")
            treatment = arguments.get("treatment")

            # This would typically get patient and recommendation data from Neo4j
            # Simplified for demonstration

            result = {
                "status": "success",
                "patient_id": patient_id,
                "treatment": treatment,
                "message": "Uncertainty quantification requires Neo4j connection with historical data",
                "note": "Full implementation calculates epistemic and aleatoric uncertainty based on similar cases"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    # ========================================
    # LOINC LABORATORY TOOLS
    # ========================================

    @server.call_tool()
    async def interpret_lab_results(arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Interpret laboratory results with LOINC coding.

        Args:
            lab_results: List of lab results [{"test_name": "...", "value": ..., "unit": "..."}]
            patient_age: Patient age (optional)
            patient_sex: Patient sex (optional)
        """
        try:
            lab_results_data = arguments.get("lab_results", [])
            patient_age = arguments.get("patient_age")
            patient_sex = arguments.get("patient_sex")

            interpreted_results = loinc_integrator.process_lab_panel(
                lab_results=lab_results_data,
                patient_age=patient_age,
                patient_sex=patient_sex
            )

            # Assess clinical significance
            assessment = loinc_integrator.assess_clinical_significance(interpreted_results)

            result = {
                "status": "success",
                "results": [
                    {
                        "test": r.test_name,
                        "loinc_code": r.loinc_code,
                        "value": r.value,
                        "unit": r.unit,
                        "reference_range": f"{r.reference_range_low}-{r.reference_range_high}" if r.reference_range_low else "N/A",
                        "interpretation": r.interpretation,
                        "snomed_code": r.snomed_code
                    }
                    for r in interpreted_results
                ],
                "clinical_assessment": assessment
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": str(e)}, indent=2)
            )]

    logger.info("✓ Enhanced MCP tools registered successfully")

    return {
        "graph_algorithms": graph_algorithms,
        "temporal_analyzer": temporal_analyzer,
        "biomarker_agent": biomarker_agent,
        "uncertainty_quantifier": uncertainty_quantifier,
        "loinc_integrator": loinc_integrator
    }
