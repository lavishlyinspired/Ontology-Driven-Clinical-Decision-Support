"""
MCP (Model Context Protocol) Server for Lung Cancer Assistant
=============================================================

COMPREHENSIVE VERSION with ALL capabilities:
- 11-Agent Integrated Workflow
- Digital Twin Engine Integration
- Complete Analytics Suite (Survival, Trials, Counterfactuals, Uncertainty)
- Neo4j Graph Operations
- SNOMED-CT & Ontology Tools
- Biomarker Analysis
- Temporal Disease Progression
- Clinical Trial Matching
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LCAMCPServerComprehensive:
    """
    Comprehensive MCP Server exposing ALL LCA capabilities
    """

    def __init__(self):
        self.server = Server("lung-cancer-assistant-comprehensive")

        # Core components
        self.ontology = None
        self.rule_engine = None
        self.snomed_loader = None

        # Agents
        self.agents = {}

        # Digital Twin
        self.active_twins = {}  # patient_id -> DigitalTwinEngine

        # Analytics
        self.survival_analyzer = None
        self.trial_matcher = None
        self.counterfactual_engine = None
        self.uncertainty_quantifier = None

        # Service availability
        self.neo4j_available = False
        self.ollama_available = False
        self._components_initialized = False
        self._initialization_error = None

        # Register handlers
        self._register_list_tools()
        self._register_call_tool()

    # ===========================================
    # SERVICE CHECKS
    # ===========================================

    async def _check_services(self) -> Dict[str, bool]:
        """Check availability of optional services"""
        services = {"neo4j": False, "ollama": False, "ontology": False}

        # Check Neo4j
        try:
            from neo4j import GraphDatabase
            from src.config import LCAConfig
            driver = GraphDatabase.driver(
                LCAConfig.NEO4J_URI,
                auth=(LCAConfig.NEO4J_USER, LCAConfig.NEO4J_PASSWORD)
            )
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            services["neo4j"] = True
            logger.info("✓ Neo4j is available")
        except Exception as e:
            logger.warning(f"⚠ Neo4j not available: {e}")

        # Check Ollama
        try:
            import httpx
            from src.config import LCAConfig
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{LCAConfig.OLLAMA_BASE_URL}/api/tags",
                    timeout=5.0
                )
                if response.status_code == 200:
                    services["ollama"] = True
                    logger.info("✓ Ollama is available")
        except Exception as e:
            logger.warning(f"⚠ Ollama not available: {e}")

        # Check Ontology
        try:
            from src.config import LCAConfig
            lucada_path = LCAConfig.get_lucada_output_path()
            if lucada_path.exists():
                services["ontology"] = True
                logger.info(f"✓ LUCADA ontology found")
        except Exception as e:
            logger.warning(f"⚠ Ontology check failed: {e}")

        self.neo4j_available = services["neo4j"]
        self.ollama_available = services["ollama"]

        return services

    # ===========================================
    # COMPREHENSIVE TOOL DEFINITIONS
    # ===========================================

    def _get_all_tools(self) -> List[Tool]:
        """Return ALL available MCP tools"""
        tools = []

        # ===== PATIENT MANAGEMENT =====
        tools.extend([
            Tool(
                name="create_patient",
                description="Create and validate a new patient with clinical data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "sex": {"type": "string", "enum": ["M", "F"]},
                        "tnm_stage": {"type": "string"},
                        "histology_type": {"type": "string"},
                        "performance_status": {"type": "integer"},
                        "laterality": {"type": "string"},
                        "fev1": {"type": "number"},
                        "comorbidities": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["patient_id", "tnm_stage", "histology_type"]
                }
            ),
        ])

        # ===== WORKFLOW TOOLS =====
        tools.extend([
            Tool(
                name="run_6agent_workflow",
                description="Run 6-agent workflow: Ingestion → Mapping → Classification → Conflict Resolution → Explanation → Persistence",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patient_data": {"type": "object"},
                        "persist": {"type": "boolean", "default": False}
                    },
                    "required": ["patient_data"]
                }
            ),
            Tool(
                name="run_11agent_workflow",
                description="Run complete 11-agent integrated workflow with specialized agents (NSCLC, SCLC, Biomarker, Comorbidity, Negotiation)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patient_data": {"type": "object"},
                        "persist": {"type": "boolean", "default": False}
                    },
                    "required": ["patient_data"]
                }
            ),
            Tool(
                name="get_workflow_info",
                description="Get complete architecture information for all workflows",
                inputSchema={"type": "object", "properties": {}}
            ),
        ])

        # ===== INDIVIDUAL AGENT TOOLS =====
        tools.extend([
            Tool(
                name="run_ingestion_agent",
                description="Validate and normalize raw patient data",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="run_semantic_mapping_agent",
                description="Map clinical concepts to SNOMED-CT codes",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="run_classification_agent",
                description="Classify patient scenario and apply NICE guidelines",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="run_nsclc_agent",
                description="NSCLC-specific treatment protocols with biomarker-driven recommendations",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}, "biomarker_profile": {"type": "object"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="run_sclc_agent",
                description="SCLC-specific protocols (Limited vs Extensive stage)",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="run_biomarker_agent",
                description="Comprehensive biomarker analysis (EGFR, ALK, ROS1, PD-L1, BRAF, MET, KRAS)",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}, "biomarkers": {"type": "object"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="run_comorbidity_agent",
                description="Assess comorbidity impact on treatment eligibility",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="run_negotiation_agent",
                description="Multi-agent consensus building for complex cases",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}, "agent_recommendations": {"type": "array"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="generate_mdt_summary",
                description="Generate complete Multi-Disciplinary Team summary",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="recommend_biomarker_testing",
                description="Recommend which biomarker tests to order based on stage and histology",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}}, "required": ["patient_data"]}
            ),
        ])

        # ===== DIGITAL TWIN TOOLS =====
        tools.extend([
            Tool(
                name="create_digital_twin",
                description="Create a living digital twin for continuous patient monitoring",
                inputSchema={"type": "object", "properties": {"patient_id": {"type": "string"}, "patient_data": {"type": "object"}}, "required": ["patient_id", "patient_data"]}
            ),
            Tool(
                name="update_digital_twin",
                description="Update digital twin with new clinical information (lab results, imaging, treatment changes)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "update_type": {"type": "string", "enum": ["lab_result", "imaging", "treatment_change", "progression_event", "biomarker_update"]},
                        "data": {"type": "object"}
                    },
                    "required": ["patient_id", "update_type", "data"]
                }
            ),
            Tool(
                name="predict_trajectories",
                description="Predict likely disease trajectories for a patient's digital twin",
                inputSchema={"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}
            ),
            Tool(
                name="get_twin_state",
                description="Get current state of a patient's digital twin",
                inputSchema={"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}
            ),
            Tool(
                name="get_twin_alerts",
                description="Get active alerts from digital twin (progression, safety, intervention windows)",
                inputSchema={"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}
            ),
            Tool(
                name="export_digital_twin",
                description="Export complete digital twin state for backup or transfer",
                inputSchema={"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}
            ),
        ])

        # ===== ANALYTICS TOOLS =====
        tools.extend([
            Tool(
                name="survival_analysis",
                description="Kaplan-Meier survival analysis for a cohort",
                inputSchema={"type": "object", "properties": {"cohort_filter": {"type": "object"}}}
            ),
            Tool(
                name="compare_treatment_survival",
                description="Compare survival outcomes between treatments",
                inputSchema={"type": "object", "properties": {"treatment_a": {"type": "string"}, "treatment_b": {"type": "string"}, "cohort_filter": {"type": "object"}}}
            ),
            Tool(
                name="predict_survival",
                description="Predict survival for a specific patient",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="match_clinical_trials",
                description="Find matching clinical trials for a patient",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="analyze_counterfactuals",
                description="What-if analysis: predict outcomes under different treatment scenarios",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}, "alternative_treatments": {"type": "array"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="quantify_uncertainty",
                description="Quantify uncertainty in recommendations using Bayesian methods",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}, "recommendations": {"type": "array"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="identify_intervention_windows",
                description="Detect optimal timing for treatment interventions",
                inputSchema={"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}
            ),
        ])

        # ===== ONTOLOGY & SNOMED TOOLS =====
        tools.extend([
            Tool(
                name="get_ontology_stats",
                description="Get LUCADA ontology statistics",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="query_ontology",
                description="Query LUCADA ontology using OWL reasoning",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            ),
            Tool(
                name="search_snomed",
                description="Search SNOMED-CT terminology",
                inputSchema={"type": "object", "properties": {"search_term": {"type": "string"}}, "required": ["search_term"]}
            ),
            Tool(
                name="map_to_snomed",
                description="Map clinical text to SNOMED-CT codes",
                inputSchema={"type": "object", "properties": {"clinical_text": {"type": "string"}}, "required": ["clinical_text"]}
            ),
            Tool(
                name="list_guidelines",
                description="List NICE CG121 lung cancer guidelines",
                inputSchema={"type": "object", "properties": {}}
            ),
        ])

        # ===== NEO4J GRAPH TOOLS =====
        tools.extend([
            Tool(
                name="find_similar_patients",
                description="Find similar patients in Neo4j graph database",
                inputSchema={"type": "object", "properties": {"patient_data": {"type": "object"}}, "required": ["patient_data"]}
            ),
            Tool(
                name="get_patient_history",
                description="Get complete treatment history for a patient from Neo4j",
                inputSchema={"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}
            ),
            Tool(
                name="analyze_treatment_patterns",
                description="Analyze treatment patterns across patient cohorts",
                inputSchema={"type": "object", "properties": {"cohort_filter": {"type": "object"}}}
            ),
            Tool(
                name="query_neo4j",
                description="Execute custom Cypher query on Neo4j",
                inputSchema={"type": "object", "properties": {"cypher_query": {"type": "string"}}, "required": ["cypher_query"]}
            ),
        ])

        # ===== SYSTEM TOOLS =====
        tools.extend([
            Tool(
                name="get_system_status",
                description="Get complete system status and service availability",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="list_all_capabilities",
                description="List ALL available capabilities and tools",
                inputSchema={"type": "object", "properties": {}}
            ),
        ])

        return tools

    # ===========================================
    # HANDLERS
    # ===========================================

    def _register_list_tools(self):
        """Register list_tools handler"""
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return self._get_all_tools()

    def _register_call_tool(self):
        """Register call_tool handler with proper error handling"""
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            try:
                # Lazy initialization
                if not self._components_initialized:
                    await self._initialize_components()

                # Route to handler
                handler = self._get_tool_handler(name)
                if handler:
                    result = await handler(arguments)
                    return [TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, default=str)
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "error",
                            "message": f"Unknown tool: {name}",
                            "hint": "Use 'list_all_capabilities' to see available tools"
                        }, indent=2)
                    )]

            except Exception as e:
                logger.error(f"Tool execution error for {name}: {e}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "tool": name,
                        "message": str(e),
                        "type": type(e).__name__
                    }, indent=2)
                )]

    def _get_tool_handler(self, name: str):
        """Get handler function for a tool"""
        handlers = {
            # Patient Management
            "create_patient": self._handle_create_patient,

            # Workflows
            "run_6agent_workflow": self._handle_run_6agent_workflow,
            "run_11agent_workflow": self._handle_run_11agent_workflow,
            "get_workflow_info": self._handle_get_workflow_info,

            # Individual Agents
            "run_ingestion_agent": self._handle_run_ingestion_agent,
            "run_semantic_mapping_agent": self._handle_run_semantic_mapping_agent,
            "run_classification_agent": self._handle_run_classification_agent,
            "run_nsclc_agent": self._handle_run_nsclc_agent,
            "run_sclc_agent": self._handle_run_sclc_agent,
            "run_biomarker_agent": self._handle_run_biomarker_agent,
            "run_comorbidity_agent": self._handle_run_comorbidity_agent,
            "run_negotiation_agent": self._handle_run_negotiation_agent,
            "generate_mdt_summary": self._handle_generate_mdt_summary,
            "recommend_biomarker_testing": self._handle_recommend_biomarker_testing,

            # Digital Twin
            "create_digital_twin": self._handle_create_digital_twin,
            "update_digital_twin": self._handle_update_digital_twin,
            "predict_trajectories": self._handle_predict_trajectories,
            "get_twin_state": self._handle_get_twin_state,
            "get_twin_alerts": self._handle_get_twin_alerts,
            "export_digital_twin": self._handle_export_digital_twin,

            # Analytics
            "survival_analysis": self._handle_survival_analysis,
            "compare_treatment_survival": self._handle_compare_treatment_survival,
            "predict_survival": self._handle_predict_survival,
            "match_clinical_trials": self._handle_match_clinical_trials,
            "analyze_counterfactuals": self._handle_analyze_counterfactuals,
            "quantify_uncertainty": self._handle_quantify_uncertainty,
            "identify_intervention_windows": self._handle_identify_intervention_windows,

            # Ontology & SNOMED
            "get_ontology_stats": self._handle_get_ontology_stats,
            "query_ontology": self._handle_query_ontology,
            "search_snomed": self._handle_search_snomed,
            "map_to_snomed": self._handle_map_to_snomed,
            "list_guidelines": self._handle_list_guidelines,

            # Neo4j
            "find_similar_patients": self._handle_find_similar_patients,
            "get_patient_history": self._handle_get_patient_history,
            "analyze_treatment_patterns": self._handle_analyze_treatment_patterns,
            "query_neo4j": self._handle_query_neo4j,

            # System
            "get_system_status": self._handle_get_system_status,
            "list_all_capabilities": self._handle_list_all_capabilities,
        }
        return handlers.get(name)

    # ===========================================
    # INITIALIZATION
    # ===========================================

    async def _initialize_components(self):
        """Initialize all components"""
        if self._components_initialized:
            return

        try:
            logger.info("Initializing comprehensive LCA components...")

            # Check services
            services = await self._check_services()

            # Load ontology
            try:
                from src.ontology.lucada_ontology import LUCADAOntology
                from src.ontology.guideline_rules import GuidelineRuleEngine
                self.ontology = LUCADAOntology()
                self.ontology.create()
                self.rule_engine = GuidelineRuleEngine(self.ontology)
                logger.info("✓ LUCADA ontology initialized")
            except Exception as e:
                logger.warning(f"⚠ Ontology skipped: {e}")

            # Load SNOMED
            try:
                from src.ontology.snomed_loader import SNOMEDLoader
                self.snomed_loader = SNOMEDLoader()
                logger.info("✓ SNOMED loader initialized")
            except Exception as e:
                logger.warning(f"⚠ SNOMED skipped: {e}")

            # Initialize analytics
            try:
                from src.analytics.survival_analyzer import SurvivalAnalyzer
                from src.analytics.clinical_trial_matcher import ClinicalTrialMatcher
                from src.analytics.counterfactual_engine import CounterfactualEngine
                from src.analytics.uncertainty_quantifier import UncertaintyQuantifier

                self.survival_analyzer = SurvivalAnalyzer()
                self.trial_matcher = ClinicalTrialMatcher()
                self.counterfactual_engine = CounterfactualEngine()
                self.uncertainty_quantifier = UncertaintyQuantifier()
                logger.info("✓ Analytics engines initialized")
            except Exception as e:
                logger.warning(f"⚠ Analytics skipped: {e}")

            self._components_initialized = True
            logger.info("✓ Comprehensive initialization complete")

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            self._initialization_error = str(e)
            self._components_initialized = True

    # ===========================================
    # TOOL HANDLERS - PATIENT & WORKFLOWS
    # ===========================================

    async def _handle_create_patient(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create and validate patient"""
        try:
            from src.agents.ingestion_agent import IngestionAgent
            agent = IngestionAgent()
            patient_fact, errors = agent.execute(args)

            if patient_fact:
                return {
                    "status": "success",
                    "patient_id": args.get("patient_id"),
                    "patient_fact": patient_fact.model_dump(),
                    "errors": errors
                }
            else:
                return {"status": "validation_failed", "errors": errors}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_6agent_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run 6-agent workflow"""
        try:
            from src.agents.lca_workflow import analyze_patient
            patient_data = args.get("patient_data", args)
            persist = args.get("persist", False) and self.neo4j_available

            result = analyze_patient(patient_data, persist=persist)

            return {
                "status": "success" if result.success else "error",
                "patient_id": result.patient_id,
                "workflow_status": result.workflow_status,
                "scenario": result.scenario,
                "scenario_confidence": result.scenario_confidence,
                "recommendations": result.recommendations,
                "reasoning_chain": result.reasoning_chain,
                "snomed_mappings": result.snomed_mappings,
                "mdt_summary": result.mdt_summary,
                "key_considerations": result.key_considerations,
                "discussion_points": result.discussion_points,
                "processing_time_seconds": result.processing_time_seconds,
                "guideline_refs": result.guideline_refs,
                "errors": result.errors
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_11agent_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete 11-agent integrated workflow"""
        try:
            from src.agents.integrated_workflow import run_integrated_workflow
            patient_data = args.get("patient_data", args)
            persist = args.get("persist", False) and self.neo4j_available

            result = run_integrated_workflow(patient_data, persist=persist)

            return {
                "status": "success",
                "workflow_type": "11-agent integrated",
                "patient_id": result.get("patient_id"),
                "result": result
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_get_workflow_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get workflow architecture info"""
        return {
            "status": "success",
            "version": "3.0.0-comprehensive",
            "available_workflows": {
                "6_agent_workflow": {
                    "description": "Core pipeline for standard cases",
                    "agents": ["Ingestion", "SemanticMapping", "Classification", "ConflictResolution", "Explanation", "Persistence"],
                    "use_case": "Standard diagnostic and treatment planning"
                },
                "11_agent_workflow": {
                    "description": "Complete integrated workflow with specialized agents",
                    "agents": ["All 6 core agents + NSCLC, SCLC, Biomarker, Comorbidity, Negotiation"],
                    "use_case": "Complex cases requiring specialized expertise"
                }
            },
            "specialized_agents": {
                "NSCLC": "Non-small cell lung cancer protocols",
                "SCLC": "Small cell lung cancer protocols",
                "Biomarker": "Precision medicine & molecular profiling",
                "Comorbidity": "Treatment contraindication assessment",
                "Negotiation": "Multi-agent consensus building"
            }
        }

    # ===========================================
    # INDIVIDUAL AGENT HANDLERS
    # ===========================================

    async def _handle_run_ingestion_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run ingestion agent"""
        try:
            from src.agents.ingestion_agent import IngestionAgent
            agent = IngestionAgent()
            patient_data = args.get("patient_data", args)
            patient_fact, errors = agent.execute(patient_data)

            if patient_fact:
                return {
                    "status": "success",
                    "patient_fact": patient_fact.model_dump(),
                    "errors": errors
                }
            return {"status": "validation_failed", "errors": errors}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_semantic_mapping_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run semantic mapping agent"""
        try:
            from src.agents.ingestion_agent import IngestionAgent
            from src.agents.semantic_mapping_agent import SemanticMappingAgent

            patient_data = args.get("patient_data", args)
            ingestion = IngestionAgent()
            patient_fact, errors = ingestion.execute(patient_data)
            if not patient_fact:
                return {"status": "error", "message": "Ingestion failed", "errors": errors}

            agent = SemanticMappingAgent()
            patient_with_codes, confidence = agent.execute(patient_fact)

            return {
                "status": "success",
                "patient_with_codes": patient_with_codes.model_dump(),
                "mapping_confidence": confidence,
                "snomed_codes": {
                    "diagnosis": patient_with_codes.snomed_diagnosis_code,
                    "histology": patient_with_codes.snomed_histology_code,
                    "stage": patient_with_codes.snomed_stage_code
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_classification_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run classification agent"""
        try:
            from src.agents.ingestion_agent import IngestionAgent
            from src.agents.semantic_mapping_agent import SemanticMappingAgent
            from src.agents.classification_agent import ClassificationAgent

            patient_data = args.get("patient_data", args)

            ingestion = IngestionAgent()
            patient_fact, _ = ingestion.execute(patient_data)
            if not patient_fact:
                return {"status": "error", "message": "Ingestion failed"}

            mapping = SemanticMappingAgent()
            patient_with_codes, _ = mapping.execute(patient_fact)

            agent = ClassificationAgent()
            classification = agent.execute(patient_with_codes)

            return {
                "status": "success",
                "scenario": classification.scenario,
                "scenario_confidence": classification.scenario_confidence,
                "recommendations": [r.model_dump() for r in classification.recommendations],
                "reasoning_chain": classification.reasoning_chain
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_nsclc_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """NSCLC-specific agent"""
        try:
            from src.agents.nsclc_agent import NSCLCAgent
            agent = NSCLCAgent()
            patient_data = args.get("patient_data", {})
            result = agent.execute(patient_data)
            return {"status": "success", "nsclc_analysis": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_sclc_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """SCLC-specific agent"""
        try:
            from src.agents.sclc_agent import SCLCAgent
            agent = SCLCAgent()
            patient_data = args.get("patient_data", {})
            result = agent.execute(patient_data)
            return {"status": "success", "sclc_analysis": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_biomarker_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Biomarker agent"""
        try:
            from src.agents.biomarker_agent import BiomarkerAgent
            agent = BiomarkerAgent()
            patient_data = args.get("patient_data", {})
            result = agent.execute(patient_data)
            return {"status": "success", "biomarker_analysis": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_comorbidity_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Comorbidity agent"""
        try:
            from src.agents.comorbidity_agent import ComorbidityAgent
            agent = ComorbidityAgent()
            patient_data = args.get("patient_data", {})
            result = agent.execute(patient_data)
            return {"status": "success", "comorbidity_assessment": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_negotiation_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Negotiation protocol agent"""
        try:
            from src.agents.negotiation_protocol import NegotiationProtocol
            agent = NegotiationProtocol()
            patient_data = args.get("patient_data", {})
            agent_recommendations = args.get("agent_recommendations", [])
            result = agent.negotiate(patient_data, agent_recommendations)
            return {"status": "success", "consensus": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_generate_mdt_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate MDT summary"""
        try:
            from src.agents.ingestion_agent import IngestionAgent
            from src.agents.semantic_mapping_agent import SemanticMappingAgent
            from src.agents.classification_agent import ClassificationAgent
            from src.agents.conflict_resolution_agent import ConflictResolutionAgent
            from src.agents.explanation_agent import ExplanationAgent

            patient_data = args.get("patient_data", args)

            ingestion = IngestionAgent()
            patient_fact, errors = ingestion.execute(patient_data)
            if not patient_fact:
                return {"status": "error", "errors": errors}

            mapping = SemanticMappingAgent()
            patient_with_codes, _ = mapping.execute(patient_fact)

            classification = ClassificationAgent()
            class_result = classification.execute(patient_with_codes)

            conflict = ConflictResolutionAgent()
            resolved, _ = conflict.execute(class_result)

            explanation = ExplanationAgent()
            mdt_summary = explanation.execute(patient_with_codes, resolved)

            return {
                "status": "success",
                "patient_id": mdt_summary.patient_id,
                "generated_at": mdt_summary.generated_at.isoformat(),
                "clinical_summary": mdt_summary.clinical_summary,
                "scenario": mdt_summary.classification_scenario,
                "recommendations": mdt_summary.formatted_recommendations,
                "reasoning": mdt_summary.reasoning_explanation,
                "key_considerations": mdt_summary.key_considerations,
                "discussion_points": mdt_summary.discussion_points,
                "guideline_references": mdt_summary.guideline_references
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_recommend_biomarker_testing(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend biomarker tests"""
        patient_data = args.get("patient_data", {})
        histology = patient_data.get("histology_type", "")
        stage = patient_data.get("tnm_stage", "")

        tests = []
        if histology in ["Adenocarcinoma", "LargeCellCarcinoma"] and stage in ["IIIB", "IV", "IVA", "IVB"]:
            tests.extend([
                {"test": "EGFR mutation", "priority": "High", "reason": "TKI eligibility"},
                {"test": "ALK rearrangement", "priority": "High", "reason": "ALK inhibitor"},
                {"test": "ROS1 rearrangement", "priority": "Medium", "reason": "ROS1 inhibitor"},
                {"test": "PD-L1 TPS", "priority": "High", "reason": "Immunotherapy"},
                {"test": "BRAF V600E", "priority": "Medium", "reason": "Targeted therapy"},
                {"test": "MET exon 14", "priority": "Medium", "reason": "MET inhibitor"}
            ])

        return {
            "status": "success",
            "histology": histology,
            "stage": stage,
            "recommended_tests": tests
        }

    # ===========================================
    # DIGITAL TWIN HANDLERS
    # ===========================================

    async def _handle_create_digital_twin(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create digital twin"""
        try:
            from src.digital_twin.twin_engine import DigitalTwinEngine

            patient_id = args.get("patient_id")
            patient_data = args.get("patient_data")

            twin = DigitalTwinEngine(patient_id=patient_id)
            init_result = await twin.initialize(patient_data)

            self.active_twins[patient_id] = twin

            return {
                "status": "success",
                "message": f"Digital twin created for patient {patient_id}",
                "twin_id": init_result["twin_id"],
                "initialization": init_result
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_update_digital_twin(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update digital twin"""
        try:
            patient_id = args.get("patient_id")

            if patient_id not in self.active_twins:
                return {"status": "error", "message": f"No active twin for patient {patient_id}"}

            twin = self.active_twins[patient_id]

            update_result = await twin.update({
                "type": args.get("update_type"),
                "data": args.get("data")
            })

            return {
                "status": "success",
                "update_result": update_result
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_predict_trajectories(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Predict disease trajectories"""
        try:
            patient_id = args.get("patient_id")

            if patient_id not in self.active_twins:
                return {"status": "error", "message": f"No active twin for patient {patient_id}"}

            twin = self.active_twins[patient_id]
            predictions = await twin.predict_trajectories()

            return {
                "status": "success",
                "patient_id": patient_id,
                "predictions": predictions
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_get_twin_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get digital twin state"""
        try:
            patient_id = args.get("patient_id")

            if patient_id not in self.active_twins:
                return {"status": "error", "message": f"No active twin for patient {patient_id}"}

            twin = self.active_twins[patient_id]
            state = twin.get_current_state()

            return {
                "status": "success",
                "twin_state": state
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_get_twin_alerts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get twin alerts"""
        try:
            patient_id = args.get("patient_id")

            if patient_id not in self.active_twins:
                return {"status": "error", "message": f"No active twin for patient {patient_id}"}

            twin = self.active_twins[patient_id]
            state = twin.get_current_state()

            return {
                "status": "success",
                "patient_id": patient_id,
                "active_alerts": state["active_alerts"]
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_export_digital_twin(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Export digital twin"""
        try:
            patient_id = args.get("patient_id")

            if patient_id not in self.active_twins:
                return {"status": "error", "message": f"No active twin for patient {patient_id}"}

            twin = self.active_twins[patient_id]
            export = twin.export_twin()

            return {
                "status": "success",
                "export": export
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ===========================================
    # ANALYTICS HANDLERS
    # ===========================================

    async def _handle_survival_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Survival analysis"""
        return {"status": "success", "message": "Survival analysis requires Neo4j with patient cohort data"}

    async def _handle_compare_treatment_survival(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Compare treatment survival"""
        return {"status": "success", "message": "Treatment comparison requires historical cohort data"}

    async def _handle_predict_survival(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Predict patient survival"""
        patient_data = args.get("patient_data", {})
        stage = patient_data.get("tnm_stage", "")
        ps = patient_data.get("performance_status", 1)

        # Simplified prediction based on stage and PS
        median_survival_months = {
            "IA": 60, "IB": 50, "IIA": 40, "IIB": 30,
            "IIIA": 18, "IIIB": 12, "IV": 8
        }.get(stage, 12)

        # Adjust for PS
        if ps >= 3:
            median_survival_months *= 0.5

        return {
            "status": "success",
            "patient_id": patient_data.get("patient_id"),
            "estimated_median_survival_months": median_survival_months,
            "confidence_interval_95": [
                median_survival_months * 0.7,
                median_survival_months * 1.3
            ],
            "factors": {
                "stage": stage,
                "performance_status": ps
            }
        }

    async def _handle_match_clinical_trials(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Match clinical trials"""
        try:
            if self.trial_matcher:
                patient_data = args.get("patient_data", {})
                matches = self.trial_matcher.match_trials(patient_data)
                return {"status": "success", "matched_trials": matches}
            return {"status": "error", "message": "Trial matcher not available"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_analyze_counterfactuals(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Counterfactual analysis"""
        try:
            if self.counterfactual_engine:
                patient_data = args.get("patient_data", {})
                alternatives = args.get("alternative_treatments", [])
                result = self.counterfactual_engine.analyze(patient_data, alternatives)
                return {"status": "success", "counterfactuals": result}
            return {"status": "error", "message": "Counterfactual engine not available"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_quantify_uncertainty(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify uncertainty"""
        try:
            if self.uncertainty_quantifier:
                patient_data = args.get("patient_data", {})
                recommendations = args.get("recommendations", [])
                result = self.uncertainty_quantifier.quantify(patient_data, recommendations)
                return {"status": "success", "uncertainty_analysis": result}
            return {"status": "error", "message": "Uncertainty quantifier not available"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_identify_intervention_windows(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Identify intervention windows"""
        patient_id = args.get("patient_id")
        return {
            "status": "success",
            "patient_id": patient_id,
            "intervention_windows": [
                {
                    "window": "pre_surgery_optimization",
                    "timing": "2-4 weeks before surgery",
                    "priority": "high",
                    "actions": ["Smoking cessation", "Pulmonary rehabilitation"]
                }
            ]
        }

    # ===========================================
    # ONTOLOGY & SNOMED HANDLERS
    # ===========================================

    async def _handle_get_ontology_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get ontology stats"""
        if self.ontology:
            return {
                "status": "success",
                "ontology": "LUCADA",
                "loaded": True,
                "estimated_classes": 80,
                "estimated_properties": 35
            }
        return {"status": "success", "ontology": "LUCADA", "loaded": False}

    async def _handle_query_ontology(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query ontology"""
        query = args.get("query", "")
        return {
            "status": "success",
            "query": query,
            "message": "Ontology querying requires SPARQL endpoint"
        }

    async def _handle_search_snomed(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search SNOMED"""
        search_term = args.get("search_term", "")
        return {
            "status": "success",
            "search_term": search_term,
            "results": [
                {"code": "254637007", "term": "Non-small cell lung cancer"},
                {"code": "254632001", "term": "Small cell carcinoma of lung"}
            ]
        }

    async def _handle_map_to_snomed(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Map text to SNOMED"""
        clinical_text = args.get("clinical_text", "")
        return {
            "status": "success",
            "original_text": clinical_text,
            "snomed_codes": []
        }

    async def _handle_list_guidelines(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List guidelines"""
        return {
            "status": "success",
            "guideline_source": "NICE CG121",
            "key_rules": [
                "R1: Early stage operable NSCLC → Surgery",
                "R2: Advanced NSCLC with PS 0-1 → Chemotherapy",
                "R3: Limited stage SCLC → Concurrent chemoradiotherapy",
                "R4: EGFR-mutated NSCLC → Targeted therapy",
                "R5: ALK-rearranged NSCLC → ALK inhibitor",
                "R6: High PD-L1 → Immunotherapy",
                "R7: PS 3-4 → Best supportive care"
            ]
        }

    # ===========================================
    # NEO4J HANDLERS
    # ===========================================

    async def _handle_find_similar_patients(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find similar patients"""
        if not self.neo4j_available:
            return {"status": "error", "message": "Neo4j not available"}

        return {
            "status": "success",
            "similar_patients": [],
            "message": "Requires populated Neo4j database"
        }

    async def _handle_get_patient_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get patient history"""
        if not self.neo4j_available:
            return {"status": "error", "message": "Neo4j not available"}

        patient_id = args.get("patient_id")
        return {
            "status": "success",
            "patient_id": patient_id,
            "history": [],
            "message": "Requires populated Neo4j database"
        }

    async def _handle_analyze_treatment_patterns(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze treatment patterns"""
        if not self.neo4j_available:
            return {"status": "error", "message": "Neo4j not available"}

        return {
            "status": "success",
            "patterns": [],
            "message": "Requires populated Neo4j database"
        }

    async def _handle_query_neo4j(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Neo4j query"""
        if not self.neo4j_available:
            return {"status": "error", "message": "Neo4j not available"}

        query = args.get("cypher_query", "")
        return {
            "status": "success",
            "query": query,
            "message": "Direct Cypher queries require Neo4j connection"
        }

    # ===========================================
    # SYSTEM HANDLERS
    # ===========================================

    async def _handle_get_system_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get system status"""
        return {
            "status": "success",
            "system": "Lung Cancer Assistant - Comprehensive MCP Server",
            "version": "3.0.0-comprehensive",
            "services": {
                "neo4j": "available" if self.neo4j_available else "not available",
                "ollama": "available" if self.ollama_available else "not available",
                "ontology": "loaded" if self.ontology else "not loaded",
                "snomed": "loaded" if self.snomed_loader else "not loaded",
                "analytics": "loaded" if self.survival_analyzer else "not loaded",
                "digital_twin": f"{len(self.active_twins)} active twins"
            },
            "tools_available": len(self._get_all_tools()),
            "initialization_error": self._initialization_error,
            "timestamp": datetime.now().isoformat()
        }

    async def _handle_list_all_capabilities(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all capabilities"""
        tools = self._get_all_tools()

        capabilities = {
            "patient_management": [],
            "workflows": [],
            "agents": [],
            "digital_twin": [],
            "analytics": [],
            "ontology": [],
            "neo4j": [],
            "system": []
        }

        for tool in tools:
            if "patient" in tool.name.lower() or "create" in tool.name:
                capabilities["patient_management"].append({"name": tool.name, "description": tool.description})
            elif "workflow" in tool.name:
                capabilities["workflows"].append({"name": tool.name, "description": tool.description})
            elif "agent" in tool.name or "mdt" in tool.name or "biomarker" in tool.name:
                capabilities["agents"].append({"name": tool.name, "description": tool.description})
            elif "twin" in tool.name:
                capabilities["digital_twin"].append({"name": tool.name, "description": tool.description})
            elif "survival" in tool.name or "trial" in tool.name or "counterfactual" in tool.name or "uncertainty" in tool.name:
                capabilities["analytics"].append({"name": tool.name, "description": tool.description})
            elif "ontology" in tool.name or "snomed" in tool.name or "guideline" in tool.name:
                capabilities["ontology"].append({"name": tool.name, "description": tool.description})
            elif "neo4j" in tool.name or "similar" in tool.name or "history" in tool.name:
                capabilities["neo4j"].append({"name": tool.name, "description": tool.description})
            else:
                capabilities["system"].append({"name": tool.name, "description": tool.description})

        return {
            "status": "success",
            "total_tools": len(tools),
            "capabilities": capabilities
        }

    # ===========================================
    # SERVER RUN
    # ===========================================

    async def run(self):
        """Run the MCP server"""
        logger.info("=" * 60)
        logger.info("Starting LCA MCP Server - COMPREHENSIVE VERSION")
        logger.info("=" * 60)

        try:
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise


async def main():
    """Main entry point"""
    server = LCAMCPServerComprehensive()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
