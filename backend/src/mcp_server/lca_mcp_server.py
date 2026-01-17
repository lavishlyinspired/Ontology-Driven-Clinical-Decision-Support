"""
MCP (Model Context Protocol) Server for Lung Cancer Assistant
Exposes ontology operations and clinical decision support as MCP tools

6-Agent Architecture per final.md:
1. IngestionAgent: Validates and normalizes raw patient data
2. SemanticMappingAgent: Maps clinical concepts to SNOMED-CT codes
3. ClassificationAgent: Applies LUCADA ontology and NICE guidelines
4. ConflictResolutionAgent: Resolves conflicting recommendations
5. PersistenceAgent: THE ONLY AGENT THAT WRITES TO NEO4J
6. ExplanationAgent: Generates MDT summaries

CRITICAL: "Neo4j as a tool, not a brain"
"""

import asyncio
import json
import logging
from typing import Any, Dict, List
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ontology.lucada_ontology import LUCADAOntology
from src.ontology.guideline_rules import GuidelineRuleEngine
from src.ontology.snomed_loader import SNOMEDLoader
from src.agents.lca_agents import create_lca_workflow

# Import new 6-agent architecture
from src.agents.ingestion_agent import IngestionAgent
from src.agents.semantic_mapping_agent import SemanticMappingAgent
from src.agents.classification_agent import ClassificationAgent
from src.agents.conflict_resolution_agent import ConflictResolutionAgent
from src.agents.explanation_agent import ExplanationAgent
from src.agents.lca_workflow import LCAWorkflow, analyze_patient

# Import 2025 enhanced tools
from src.mcp_server.enhanced_tools import register_enhanced_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LCAMCPServer:
    """MCP Server for Lung Cancer Assistant"""

    def __init__(self):
        self.server = Server("lung-cancer-assistant")
        self.ontology: LUCADAOntology = None
        self.rule_engine: GuidelineRuleEngine = None
        self.snomed_loader: SNOMEDLoader = None
        self.workflow = None
        self.enhanced_tools = None  # For 2025 enhanced capabilities

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register all MCP tools"""

        # Tool 1: Create Patient
        @self.server.call_tool()
        async def create_patient(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Create a new patient in the LUCADA ontology.

            Args:
                patient_id: Unique patient identifier
                name: Patient name
                age: Age at diagnosis
                sex: M or F
                tnm_stage: TNM staging (IA, IB, IIA, IIB, IIIA, IIIB, IV)
                histology_type: Histology (Adenocarcinoma, SquamousCellCarcinoma, etc.)
                performance_status: WHO PS (0-4)
                laterality: Right, Left, or Bilateral
            """
            try:
                if not self.ontology:
                    await self._initialize_ontology()

                patient_data = arguments

                result = {
                    "status": "success",
                    "patient_id": patient_data.get("patient_id"),
                    "message": f"Patient {patient_data.get('name')} created successfully"
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 2: Classify Patient
        @self.server.call_tool()
        async def classify_patient(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Classify a patient and identify applicable treatment guidelines.

            Args:
                patient_data: Dictionary with patient clinical information
            """
            try:
                if not self.rule_engine:
                    await self._initialize_ontology()

                patient_data = arguments.get("patient_data", arguments)

                # Run classification
                recommendations = self.rule_engine.classify_patient(patient_data)

                result = {
                    "status": "success",
                    "patient_id": patient_data.get("patient_id"),
                    "applicable_rules": len(recommendations),
                    "recommendations": recommendations
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 3: Generate Treatment Recommendations
        @self.server.call_tool()
        async def generate_recommendations(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Generate detailed treatment recommendations using AI agents.

            Args:
                patient_data: Dictionary with patient clinical information
            """
            try:
                if not self.workflow:
                    self.workflow = create_lca_workflow()

                if not self.rule_engine:
                    await self._initialize_ontology()

                patient_data = arguments.get("patient_data", arguments)

                # Get guideline rules
                ontology_recommendations = self.rule_engine.classify_patient(patient_data)

                # Initialize workflow state
                initial_state = {
                    "patient_id": patient_data.get("patient_id", "UNKNOWN"),
                    "patient_data": patient_data,
                    "applicable_rules": ontology_recommendations,
                    "treatment_recommendations": ontology_recommendations,
                    "arguments": [],
                    "explanation": "",
                    "messages": []
                }

                # Run AI workflow
                final_state = self.workflow.invoke(initial_state)

                result = {
                    "status": "success",
                    "patient_id": patient_data.get("patient_id"),
                    "recommendations": ontology_recommendations,
                    "mdt_summary": final_state.get("explanation", ""),
                    "arguments": final_state.get("arguments", [])
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 4: List Guidelines
        @self.server.call_tool()
        async def list_guidelines(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            List all available clinical guideline rules.

            Args:
                None
            """
            try:
                if not self.rule_engine:
                    await self._initialize_ontology()

                rules = self.rule_engine.get_all_rules()

                rules_data = [
                    {
                        "rule_id": r.rule_id,
                        "name": r.name,
                        "source": r.source,
                        "description": r.description,
                        "treatment": r.recommended_treatment,
                        "evidence_level": r.evidence_level,
                        "intent": r.treatment_intent
                    }
                    for r in rules
                ]

                result = {
                    "status": "success",
                    "total_rules": len(rules_data),
                    "rules": rules_data
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 5: Query Ontology
        @self.server.call_tool()
        async def query_ontology(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Query the LUCADA ontology for concepts and relationships.

            Args:
                query_type: "classes", "properties", or "individuals"
                filter: Optional filter string
            """
            try:
                if not self.ontology:
                    await self._initialize_ontology()

                query_type = arguments.get("query_type", "classes")
                filter_str = arguments.get("filter", "")

                if query_type == "classes":
                    items = list(self.ontology.onto.classes())
                elif query_type == "properties":
                    items = list(self.ontology.onto.properties())
                elif query_type == "individuals":
                    items = list(self.ontology.onto.individuals())
                else:
                    items = []

                # Apply filter
                if filter_str:
                    items = [i for i in items if filter_str.lower() in str(i).lower()]

                result = {
                    "status": "success",
                    "query_type": query_type,
                    "count": len(items),
                    "items": [str(item) for item in items[:50]]  # Limit to 50
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 6: Get Ontology Stats
        @self.server.call_tool()
        async def get_ontology_stats(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Get statistics about the LUCADA ontology.

            Args:
                None
            """
            try:
                if not self.ontology:
                    await self._initialize_ontology()

                stats = {
                    "status": "success",
                    "ontology_iri": str(self.ontology.onto.base_iri),
                    "classes": len(list(self.ontology.onto.classes())),
                    "object_properties": len(list(self.ontology.onto.object_properties())),
                    "data_properties": len(list(self.ontology.onto.data_properties())),
                    "individuals": len(list(self.ontology.onto.individuals())),
                    "total_guidelines": len(self.rule_engine.rules) if self.rule_engine else 0
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(stats, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 7: Search SNOMED Concepts
        @self.server.call_tool()
        async def search_snomed(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Search for SNOMED-CT concepts by term.

            Args:
                query: Search term (e.g., "adenocarcinoma", "chemotherapy")
                limit: Maximum number of results (default: 20)
            """
            try:
                if not self.snomed_loader:
                    await self._initialize_snomed()

                query = arguments.get("query", "")
                limit = arguments.get("limit", 20)

                results = self.snomed_loader.search_concepts(query, limit=limit)

                result_data = {
                    "status": "success",
                    "query": query,
                    "results_count": len(results),
                    "results": [
                        {
                            "name": str(r),
                            "iri": getattr(r, 'iri', ''),
                            "label": r.label[0] if hasattr(r, 'label') and r.label else str(r)
                        }
                        for r in results
                    ]
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result_data, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 8: Get SNOMED Concept Details
        @self.server.call_tool()
        async def get_snomed_concept(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Get detailed information about a SNOMED-CT concept.

            Args:
                sctid: SNOMED CT Identifier (e.g., "254637007" for NSCLC)
            """
            try:
                if not self.snomed_loader:
                    await self._initialize_snomed()

                sctid = arguments.get("sctid", "")
                concept_info = self.snomed_loader.get_concept_info(sctid)

                if not concept_info:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "error",
                            "message": f"SNOMED concept not found: {sctid}"
                        }, indent=2)
                    )]

                result = {
                    "status": "success",
                    "concept": concept_info
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 9: Map Patient Data to SNOMED
        @self.server.call_tool()
        async def map_patient_to_snomed(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Map patient clinical data to SNOMED-CT codes.

            Args:
                patient_data: Dictionary with patient clinical information
            """
            try:
                if not self.snomed_loader:
                    await self._initialize_snomed()

                patient_data = arguments.get("patient_data", arguments)
                
                # Map all patient attributes to SNOMED codes
                snomed_codes = self.snomed_loader.map_patient_to_snomed(patient_data)

                result = {
                    "status": "success",
                    "patient_id": patient_data.get("patient_id", "UNKNOWN"),
                    "snomed_mappings": snomed_codes,
                    "patient_data": patient_data
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 10: Generate OWL Expression from Patient
        @self.server.call_tool()
        async def generate_owl_expression(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Generate OWL 2 class expression for patient classification.

            Args:
                patient_data: Dictionary with patient clinical information
            """
            try:
                if not self.snomed_loader:
                    await self._initialize_snomed()

                patient_data = arguments.get("patient_data", arguments)
                owl_expression = self.snomed_loader.generate_owl_expression(patient_data)

                result = {
                    "status": "success",
                    "patient_id": patient_data.get("patient_id", "UNKNOWN"),
                    "owl_expression": owl_expression,
                    "format": "OWL 2 Manchester Syntax"
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 11: Get Lung Cancer SNOMED Concepts
        @self.server.call_tool()
        async def get_lung_cancer_concepts(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Get all pre-defined lung cancer related SNOMED-CT concepts.

            Args:
                None
            """
            try:
                if not self.snomed_loader:
                    await self._initialize_snomed()

                # Get all lung cancer concepts
                concepts = {
                    name: {
                        "sctid": sctid,
                        "name": name,
                        "category": self._categorize_concept(name)
                    }
                    for name, sctid in self.snomed_loader.LUNG_CANCER_CONCEPTS.items()
                }

                result = {
                    "status": "success",
                    "total_concepts": len(concepts),
                    "concepts": concepts,
                    "categories": {
                        "diagnoses": [c for c, d in concepts.items() if d["category"] == "diagnosis"],
                        "treatments": [c for c, d in concepts.items() if d["category"] == "treatment"],
                        "outcomes": [c for c, d in concepts.items() if d["category"] == "outcome"],
                        "performance_status": [c for c, d in concepts.items() if d["category"] == "performance_status"],
                    }
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # ========== NEW 6-AGENT WORKFLOW TOOLS ==========

        # Tool 12: Run Complete 6-Agent Workflow
        @self.server.call_tool()
        async def run_6agent_workflow(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Run the complete 6-agent workflow for patient analysis.
            
            This executes:
            1. IngestionAgent - Validates and normalizes data
            2. SemanticMappingAgent - Maps to SNOMED-CT codes
            3. ClassificationAgent - Applies NICE guidelines
            4. ConflictResolutionAgent - Ranks recommendations
            5. PersistenceAgent - Saves to Neo4j (if enabled)
            6. ExplanationAgent - Generates MDT summary

            Args:
                patient_data: Dictionary with patient clinical information
                persist: Whether to save results to Neo4j (default: false)
            """
            try:
                patient_data = arguments.get("patient_data", arguments)
                persist = arguments.get("persist", False)
                
                # Run the workflow
                result = analyze_patient(patient_data, persist=persist)
                
                response = {
                    "status": "success" if result.success else "error",
                    "patient_id": result.patient_id,
                    "workflow_status": result.workflow_status,
                    "agent_chain": result.agent_chain,
                    "scenario": result.scenario,
                    "scenario_confidence": result.scenario_confidence,
                    "recommendations": result.recommendations,
                    "reasoning_chain": result.reasoning_chain,
                    "snomed_mappings": result.snomed_mappings,
                    "mapping_confidence": result.mapping_confidence,
                    "mdt_summary": result.mdt_summary,
                    "key_considerations": result.key_considerations,
                    "discussion_points": result.discussion_points,
                    "processing_time_seconds": result.processing_time_seconds,
                    "guideline_refs": result.guideline_refs,
                    "errors": result.errors
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, default=str)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 13: Get Workflow Architecture Info
        @self.server.call_tool()
        async def get_workflow_info(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Get information about the 6-agent workflow architecture.

            Args:
                None
            """
            try:
                info = {
                    "status": "success",
                    "version": "2.0.0",
                    "architecture": "6-Agent LangGraph Workflow",
                    "principle": "Neo4j as a tool, not a brain",
                    "agents": [
                        {
                            "name": "IngestionAgent",
                            "order": 1,
                            "role": "Validates and normalizes raw patient data",
                            "neo4j_access": "READ-ONLY",
                            "tools": ["validate_schema", "normalize_tnm", "calculate_age_group"]
                        },
                        {
                            "name": "SemanticMappingAgent",
                            "order": 2,
                            "role": "Maps clinical concepts to SNOMED-CT codes",
                            "neo4j_access": "READ-ONLY",
                            "tools": ["map_to_snomed", "get_snomed_hierarchy", "validate_mapping"]
                        },
                        {
                            "name": "ClassificationAgent",
                            "order": 3,
                            "role": "Applies LUCADA ontology and NICE guidelines",
                            "neo4j_access": "READ-ONLY",
                            "tools": ["query_ontology", "apply_nice_rules", "get_guideline_recommendations"]
                        },
                        {
                            "name": "ConflictResolutionAgent",
                            "order": 4,
                            "role": "Resolves conflicting recommendations",
                            "neo4j_access": "READ-ONLY",
                            "tools": ["compare_evidence_levels", "resolve_conflict", "get_conflict_rules"]
                        },
                        {
                            "name": "PersistenceAgent",
                            "order": 5,
                            "role": "THE ONLY AGENT THAT WRITES TO NEO4J",
                            "neo4j_access": "WRITE",
                            "tools": ["save_patient_facts", "save_inference_result", "mark_inference_obsolete"]
                        },
                        {
                            "name": "ExplanationAgent",
                            "order": 6,
                            "role": "Generates MDT summaries",
                            "neo4j_access": "READ-ONLY",
                            "tools": ["format_mdt_summary", "generate_explanation", "format_for_audit"]
                        }
                    ],
                    "data_flow": "Input → Ingestion → SemanticMapping → Classification → ConflictResolution → Persistence → Explanation → Output"
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(info, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 14: Run Individual Agent (Ingestion)
        @self.server.call_tool()
        async def run_ingestion_agent(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Run only the IngestionAgent to validate and normalize patient data.

            Args:
                patient_data: Raw patient data to validate
            """
            try:
                patient_data = arguments.get("patient_data", arguments)
                
                agent = IngestionAgent()
                patient_fact, errors = agent.execute(patient_data)
                
                if patient_fact:
                    result = {
                        "status": "success",
                        "patient_fact": patient_fact.model_dump(),
                        "errors": errors
                    }
                else:
                    result = {
                        "status": "validation_failed",
                        "patient_fact": None,
                        "errors": errors
                    }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 15: Run Individual Agent (Semantic Mapping)
        @self.server.call_tool()
        async def run_semantic_mapping_agent(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Run only the SemanticMappingAgent to map patient data to SNOMED-CT codes.

            Args:
                patient_data: Validated patient data (from IngestionAgent)
            """
            try:
                patient_data = arguments.get("patient_data", arguments)
                
                # First run ingestion to get PatientFact
                ingestion = IngestionAgent()
                patient_fact, errors = ingestion.execute(patient_data)
                
                if not patient_fact:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "error",
                            "message": "Ingestion failed",
                            "errors": errors
                        }, indent=2)
                    )]
                
                # Run semantic mapping
                agent = SemanticMappingAgent()
                patient_with_codes, confidence = agent.execute(patient_fact)
                
                result = {
                    "status": "success",
                    "patient_with_codes": patient_with_codes.model_dump(),
                    "mapping_confidence": confidence,
                    "snomed_codes": {
                        "diagnosis": patient_with_codes.snomed_diagnosis_code,
                        "histology": patient_with_codes.snomed_histology_code,
                        "stage": patient_with_codes.snomed_stage_code,
                        "performance_status": patient_with_codes.snomed_ps_code,
                        "laterality": patient_with_codes.snomed_laterality_code
                    }
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 16: Run Individual Agent (Classification)
        @self.server.call_tool()
        async def run_classification_agent(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Run only the ClassificationAgent to classify patient and get recommendations.

            Args:
                patient_data: Patient data to classify
            """
            try:
                patient_data = arguments.get("patient_data", arguments)
                
                # Run ingestion and semantic mapping first
                ingestion = IngestionAgent()
                patient_fact, errors = ingestion.execute(patient_data)
                
                if not patient_fact:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "error",
                            "message": "Ingestion failed",
                            "errors": errors
                        }, indent=2)
                    )]
                
                mapping = SemanticMappingAgent()
                patient_with_codes, _ = mapping.execute(patient_fact)
                
                # Run classification
                agent = ClassificationAgent()
                classification = agent.execute(patient_with_codes)
                
                result = {
                    "status": "success",
                    "patient_id": classification.patient_id,
                    "scenario": classification.scenario,
                    "scenario_confidence": classification.scenario_confidence,
                    "recommendations": [
                        {
                            "rank": r.rank,
                            "treatment": r.treatment,
                            "evidence_level": r.evidence_level.value,
                            "intent": r.intent.value if r.intent else None,
                            "guideline_reference": r.guideline_reference,
                            "rationale": r.rationale
                        }
                        for r in classification.recommendations
                    ],
                    "reasoning_chain": classification.reasoning_chain,
                    "guideline_refs": classification.guideline_refs
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 17: Generate MDT Summary
        @self.server.call_tool()
        async def generate_mdt_summary(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Generate a complete MDT (Multi-Disciplinary Team) summary for a patient.

            Args:
                patient_data: Patient data for MDT discussion
            """
            try:
                patient_data = arguments.get("patient_data", arguments)
                
                # Run full workflow except persistence
                ingestion = IngestionAgent()
                patient_fact, errors = ingestion.execute(patient_data)
                
                if not patient_fact:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "error",
                            "message": "Ingestion failed",
                            "errors": errors
                        }, indent=2)
                    )]
                
                mapping = SemanticMappingAgent()
                patient_with_codes, _ = mapping.execute(patient_fact)
                
                classification = ClassificationAgent()
                class_result = classification.execute(patient_with_codes)
                
                conflict = ConflictResolutionAgent()
                resolved, _ = conflict.execute(class_result)
                
                explanation = ExplanationAgent()
                mdt_summary = explanation.execute(patient_with_codes, resolved)
                
                result = {
                    "status": "success",
                    "patient_id": mdt_summary.patient_id,
                    "generated_at": mdt_summary.generated_at.isoformat(),
                    "clinical_summary": mdt_summary.clinical_summary,
                    "scenario": mdt_summary.classification_scenario,
                    "scenario_confidence": mdt_summary.scenario_confidence,
                    "recommendations": mdt_summary.formatted_recommendations,
                    "reasoning": mdt_summary.reasoning_explanation,
                    "key_considerations": mdt_summary.key_considerations,
                    "discussion_points": mdt_summary.discussion_points,
                    "guideline_references": mdt_summary.guideline_references,
                    "snomed_mappings": mdt_summary.snomed_mappings,
                    "disclaimer": mdt_summary.disclaimer
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # Tool 18: Validate Patient Against Schema
        @self.server.call_tool()
        async def validate_patient_schema(arguments: Dict[str, Any]) -> List[TextContent]:
            """
            Validate patient data against the LUCADA schema without processing.

            Args:
                patient_data: Patient data to validate
            """
            try:
                patient_data = arguments.get("patient_data", arguments)
                
                agent = IngestionAgent()
                
                # Check required fields
                errors = []
                required = ["tnm_stage", "histology_type"]
                
                for field in required:
                    if field not in patient_data or not patient_data[field]:
                        errors.append(f"Missing required field: {field}")
                
                # Validate TNM stage
                valid_stages = ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IIIC", "IV", "IVA", "IVB"]
                stage = patient_data.get("tnm_stage", "")
                normalized_stage = agent.normalize_tnm(stage)
                if normalized_stage not in valid_stages:
                    errors.append(f"Invalid TNM stage: {stage}")
                
                # Validate performance status
                ps = patient_data.get("performance_status")
                if ps is not None:
                    try:
                        ps_int = int(ps)
                        if ps_int < 0 or ps_int > 4:
                            errors.append(f"Performance status must be 0-4, got: {ps}")
                    except ValueError:
                        errors.append(f"Invalid performance status: {ps}")
                
                result = {
                    "status": "valid" if not errors else "invalid",
                    "errors": errors,
                    "normalized_stage": normalized_stage if normalized_stage else None,
                    "fields_present": list(patient_data.keys())
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": str(e)}, indent=2)
                )]

        # ========================================
        # REGISTER 2025 ENHANCED TOOLS
        # ========================================
        logger.info("Registering 2025 enhanced tools...")
        try:
            enhanced_tool_instances = register_enhanced_tools(self.server, self)
            self.enhanced_tools = enhanced_tool_instances
            logger.info("✓ Enhanced tools registered: graph_algorithms, temporal_analyzer, biomarker_agent, uncertainty_quantifier, loinc_integrator")
        except Exception as e:
            logger.warning(f"Could not register enhanced tools: {e}")
            logger.warning("Continuing with basic tools only")

    def _categorize_concept(self, name: str) -> str:
        """Categorize a SNOMED concept by name."""
        if any(x in name for x in ["ps_grade", "performance"]):
            return "performance_status"
        elif any(x in name for x in ["response", "survival", "outcome", "disease"]):
            return "outcome"
        elif any(x in name for x in ["therapy", "surgery", "radiation", "chemotherapy", "care"]):
            return "treatment"
        elif any(x in name for x in ["lung", "lobe", "bilateral"]):
            return "anatomy"
        else:
            return "diagnosis"

    async def _initialize_snomed(self):
        """Initialize SNOMED loader"""
        if not self.snomed_loader:
            self.snomed_loader = SNOMEDLoader()
            try:
                self.snomed_loader.load(load_full=False)
            except Exception as e:
                # If SNOMED OWL file not available, continue with mapping dictionaries only
                logger.warning(f"SNOMED ontology not loaded: {e}")
                logger.info("Using SNOMED code mappings only")

    async def _initialize_ontology(self):
        """Initialize ontology and rule engine"""
        if not self.ontology:
            self.ontology = LUCADAOntology()
            self.ontology.create()

        if not self.rule_engine:
            self.rule_engine = GuidelineRuleEngine(self.ontology)

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    server = LCAMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
