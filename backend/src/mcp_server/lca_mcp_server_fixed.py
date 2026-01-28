"""
MCP (Model Context Protocol) Server for Lung Cancer Assistant
=============================================================

FIXED VERSION with:
- Proper error handling and graceful degradation
- Service availability checks
- Better initialization
- Claude Desktop compatibility
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


class LCAMCPServer:
    """
    Fixed MCP Server for Lung Cancer Assistant with proper error handling
    """

    def __init__(self):
        self.server = Server("lung-cancer-assistant")
        self.ontology = None
        self.rule_engine = None
        self.snomed_loader = None
        self.workflow = None
        self.neo4j_available = False
        self.ollama_available = False

        # Component availability flags
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
        services = {
            "neo4j": False,
            "ollama": False,
            "ontology": False
        }

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

        # Check Ontology files
        try:
            from src.config import LCAConfig
            lucada_path = LCAConfig.get_lucada_output_path()
            if lucada_path.exists():
                services["ontology"] = True
                logger.info(f"✓ LUCADA ontology found at {lucada_path}")
            else:
                logger.warning(f"⚠ LUCADA ontology not found at {lucada_path}")
        except Exception as e:
            logger.warning(f"⚠ Ontology check failed: {e}")

        self.neo4j_available = services["neo4j"]
        self.ollama_available = services["ollama"]

        return services

    # ===========================================
    # TOOL DEFINITIONS
    # ===========================================

    def _get_all_tools(self) -> List[Tool]:
        """Return all available MCP tools"""
        tools = []

        # Core Patient Management Tools
        tools.append(Tool(
            name="create_patient",
            description="Create a new patient with clinical data (requires stage, histology)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "Unique patient identifier"},
                    "name": {"type": "string", "description": "Patient name"},
                    "age": {"type": "integer", "description": "Age at diagnosis"},
                    "sex": {"type": "string", "enum": ["M", "F"], "description": "Biological sex"},
                    "tnm_stage": {"type": "string", "description": "TNM stage (IA, IB, IIA, IIB, IIIA, IIIB, IV)"},
                    "histology_type": {"type": "string", "description": "Histology (Adenocarcinoma, SquamousCellCarcinoma, SmallCellCarcinoma, LargeCellCarcinoma)"},
                    "performance_status": {"type": "integer", "minimum": 0, "maximum": 4},
                    "laterality": {"type": "string", "enum": ["Right", "Left", "Bilateral"]},
                    "fev1": {"type": "number", "description": "FEV1 percentage"},
                    "comorbidities": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["patient_id", "tnm_stage", "histology_type"]
            }
        ))

        tools.append(Tool(
            name="run_6agent_workflow",
            description="Run complete 6-agent workflow: Ingestion → Mapping → Classification → Conflict Resolution → Explanation → (optional) Persistence",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {
                        "type": "object",
                        "description": "Patient clinical data with stage and histology",
                        "properties": {
                            "patient_id": {"type": "string"},
                            "tnm_stage": {"type": "string"},
                            "histology_type": {"type": "string"},
                            "age": {"type": "integer"},
                            "sex": {"type": "string"},
                            "performance_status": {"type": "integer"}
                        },
                        "required": ["patient_id", "tnm_stage", "histology_type"]
                    },
                    "persist": {"type": "boolean", "default": False, "description": "Save to Neo4j (requires Neo4j)"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="generate_mdt_summary",
            description="Generate Multi-Disciplinary Team (MDT) summary with treatment recommendations based on NICE guidelines",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {
                        "type": "object",
                        "description": "Patient data",
                        "required": ["patient_id", "tnm_stage", "histology_type"]
                    }
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="get_workflow_info",
            description="Get information about the LCA agent architecture and workflow",
            inputSchema={"type": "object", "properties": {}}
        ))

        tools.append(Tool(
            name="get_system_status",
            description="Get system status including service availability (Neo4j, Ollama, Ontology)",
            inputSchema={"type": "object", "properties": {}}
        ))

        tools.append(Tool(
            name="list_guidelines",
            description="List available NICE CG121 lung cancer treatment guidelines",
            inputSchema={"type": "object", "properties": {}}
        ))

        tools.append(Tool(
            name="run_nsclc_agent",
            description="Run NSCLC-specific treatment agent for Non-Small Cell Lung Cancer (Adenocarcinoma, Squamous, Large Cell)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object"},
                    "biomarker_profile": {
                        "type": "object",
                        "description": "Biomarker results (EGFR, ALK, ROS1, PD-L1, etc.)",
                        "properties": {
                            "egfr_mutation": {"type": "string"},
                            "alk_rearrangement": {"type": "string"},
                            "ros1_rearrangement": {"type": "string"},
                            "pdl1_tps": {"type": "number"}
                        }
                    }
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="run_sclc_agent",
            description="Run SCLC-specific agent for Small Cell Lung Cancer (Limited vs Extensive stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="recommend_biomarker_testing",
            description="Recommend which biomarker tests should be ordered based on stage and histology",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {
                        "type": "object",
                        "description": "Patient data with stage and histology"
                    }
                },
                "required": ["patient_data"]
            }
        ))

        # Add Ontology tools
        tools.append(Tool(
            name="get_ontology_stats",
            description="Get LUCADA ontology statistics (classes, properties, individuals)",
            inputSchema={"type": "object", "properties": {}}
        ))

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
                            "available_tools": [t.name for t in self._get_all_tools()]
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
            "create_patient": self._handle_create_patient,
            "run_6agent_workflow": self._handle_run_6agent_workflow,
            "generate_mdt_summary": self._handle_generate_mdt_summary,
            "get_workflow_info": self._handle_get_workflow_info,
            "get_system_status": self._handle_get_system_status,
            "list_guidelines": self._handle_list_guidelines,
            "run_nsclc_agent": self._handle_run_nsclc_agent,
            "run_sclc_agent": self._handle_run_sclc_agent,
            "recommend_biomarker_testing": self._handle_recommend_biomarker_testing,
            "get_ontology_stats": self._handle_get_ontology_stats,
        }
        return handlers.get(name)

    # ===========================================
    # INITIALIZATION
    # ===========================================

    async def _initialize_components(self):
        """Initialize components with proper error handling"""
        if self._components_initialized:
            return

        try:
            logger.info("Initializing LCA components...")

            # Check service availability
            services = await self._check_services()

            # Try to load ontology (not critical)
            try:
                from src.ontology.lucada_ontology import LUCADAOntology
                from src.ontology.guideline_rules import GuidelineRuleEngine

                self.ontology = LUCADAOntology()
                self.ontology.create()
                self.rule_engine = GuidelineRuleEngine(self.ontology)
                logger.info("✓ LUCADA ontology initialized")
            except Exception as e:
                logger.warning(f"⚠ Ontology initialization skipped: {e}")

            # Try to load SNOMED (not critical)
            try:
                from src.ontology.snomed_loader import SNOMEDLoader
                from src.config import LCAConfig

                snomed_path = Path(LCAConfig.SNOMED_CT_PATH)
                if snomed_path.exists():
                    self.snomed_loader = SNOMEDLoader()
                    self.snomed_loader.load(load_full=False)
                    logger.info("✓ SNOMED-CT loaded")
                else:
                    logger.warning(f"⚠ SNOMED file not found at {snomed_path}")
            except Exception as e:
                logger.warning(f"⚠ SNOMED loading skipped: {e}")

            self._components_initialized = True
            logger.info("✓ Initialization complete")
            logger.info(f"  - Neo4j: {'Available' if self.neo4j_available else 'Not available'}")
            logger.info(f"  - Ollama: {'Available' if self.ollama_available else 'Not available'}")
            logger.info(f"  - Ontology: {'Loaded' if self.ontology else 'Not loaded'}")

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            self._initialization_error = str(e)
            self._components_initialized = True  # Mark as done to avoid retry loops

    # ===========================================
    # TOOL HANDLERS
    # ===========================================

    async def _handle_create_patient(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new patient"""
        try:
            from src.agents.ingestion_agent import IngestionAgent

            agent = IngestionAgent()
            patient_fact, errors = agent.execute(args)

            if patient_fact:
                return {
                    "status": "success",
                    "patient_id": args.get("patient_id"),
                    "message": f"Patient {args.get('name', args.get('patient_id'))} validated successfully",
                    "patient_fact": patient_fact.model_dump(),
                    "errors": errors
                }
            else:
                return {
                    "status": "validation_failed",
                    "errors": errors
                }
        except ImportError:
            return {
                "status": "error",
                "message": "Ingestion agent not available. Please check installation."
            }

    async def _handle_run_6agent_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete 6-agent workflow"""
        try:
            from src.agents.lca_workflow import analyze_patient

            patient_data = args.get("patient_data", args)
            persist = args.get("persist", False)

            if persist and not self.neo4j_available:
                logger.warning("Persistence requested but Neo4j not available")
                persist = False

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
        except ImportError as e:
            return {
                "status": "error",
                "message": f"Workflow not available: {e}. Please check agent installation."
            }

    async def _handle_generate_mdt_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate MDT summary"""
        try:
            from src.agents.ingestion_agent import IngestionAgent
            from src.agents.semantic_mapping_agent import SemanticMappingAgent
            from src.agents.classification_agent import ClassificationAgent
            from src.agents.conflict_resolution_agent import ConflictResolutionAgent
            from src.agents.explanation_agent import ExplanationAgent

            patient_data = args.get("patient_data", args)

            # Run workflow
            ingestion = IngestionAgent()
            patient_fact, errors = ingestion.execute(patient_data)
            if not patient_fact:
                return {"status": "error", "message": "Validation failed", "errors": errors}

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
                "scenario_confidence": mdt_summary.scenario_confidence,
                "recommendations": mdt_summary.formatted_recommendations,
                "reasoning": mdt_summary.reasoning_explanation,
                "key_considerations": mdt_summary.key_considerations,
                "discussion_points": mdt_summary.discussion_points,
                "guideline_references": mdt_summary.guideline_references,
                "snomed_mappings": mdt_summary.snomed_mappings,
                "disclaimer": mdt_summary.disclaimer
            }
        except ImportError as e:
            return {
                "status": "error",
                "message": f"Agents not available: {e}"
            }

    async def _handle_get_workflow_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get workflow info"""
        return {
            "status": "success",
            "version": "3.0.0-fixed",
            "architecture": "6-Agent Workflow",
            "agents": [
                {"name": "IngestionAgent", "role": "Validates and normalizes patient data"},
                {"name": "SemanticMappingAgent", "role": "Maps to SNOMED-CT codes"},
                {"name": "ClassificationAgent", "role": "Applies LUCADA ontology + NICE guidelines"},
                {"name": "ConflictResolutionAgent", "role": "Resolves recommendation conflicts"},
                {"name": "ExplanationAgent", "role": "Generates MDT summaries"},
                {"name": "PersistenceAgent", "role": "Saves to Neo4j (if available)"}
            ],
            "data_flow": "Input → Ingestion → Mapping → Classification → Conflict Resolution → Explanation → [Persistence]",
            "guideline_source": "NICE CG121 Lung Cancer Guidelines"
        }

    async def _handle_get_system_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get system status"""
        return {
            "status": "success",
            "system": "Lung Cancer Assistant MCP Server (Fixed)",
            "version": "3.0.0-fixed",
            "services": {
                "neo4j": "available" if self.neo4j_available else "not available",
                "ollama": "available" if self.ollama_available else "not available",
                "ontology": "loaded" if self.ontology else "not loaded",
                "snomed": "loaded" if self.snomed_loader else "not loaded"
            },
            "tools_available": len(self._get_all_tools()),
            "initialization_error": self._initialization_error,
            "timestamp": datetime.now().isoformat()
        }

    async def _handle_list_guidelines(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List guidelines"""
        return {
            "status": "success",
            "guideline_source": "NICE CG121 - Lung Cancer Diagnosis and Management",
            "key_rules": [
                "R1: Early stage operable NSCLC → Surgery",
                "R2: Advanced NSCLC with PS 0-1 → Chemotherapy",
                "R3: Limited stage SCLC → Concurrent chemoradiotherapy",
                "R4: EGFR-mutated NSCLC → Targeted therapy (e.g., Osimertinib)",
                "R5: ALK-rearranged NSCLC → ALK inhibitor (e.g., Alectinib)",
                "R6: High PD-L1 expression (≥50%) → Immunotherapy",
                "R7: PS 3-4 → Best supportive care"
            ],
            "coverage": {
                "nsclc": ["Adenocarcinoma", "Squamous Cell", "Large Cell"],
                "sclc": ["Limited Stage", "Extensive Stage"],
                "stages": ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV"]
            }
        }

    async def _handle_run_nsclc_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """NSCLC-specific recommendations"""
        patient_data = args.get("patient_data", {})
        biomarkers = args.get("biomarker_profile", {})

        stage = patient_data.get("tnm_stage", "")
        ps = patient_data.get("performance_status", 1)

        recommendations = []

        # Early stage
        if stage in ["IA", "IB", "IIA", "IIB"]:
            recommendations.append({
                "treatment": "Surgical resection (lobectomy or pneumonectomy)",
                "evidence": "NICE CG121 R1",
                "intent": "Curative"
            })
        # Locally advanced
        elif stage in ["IIIA", "IIIB"]:
            if ps <= 1:
                recommendations.append({
                    "treatment": "Concurrent chemoradiotherapy",
                    "evidence": "NICE CG121",
                    "intent": "Curative"
                })
        # Advanced
        elif stage in ["IV", "IVA", "IVB"]:
            # Check biomarkers
            if biomarkers.get("egfr_mutation") == "Positive":
                recommendations.append({
                    "treatment": "Osimertinib (EGFR TKI)",
                    "evidence": "NICE CG121 R4 + FLAURA trial",
                    "intent": "Palliative"
                })
            elif biomarkers.get("alk_rearrangement") == "Positive":
                recommendations.append({
                    "treatment": "Alectinib (ALK inhibitor)",
                    "evidence": "NICE CG121 R5",
                    "intent": "Palliative"
                })
            elif biomarkers.get("pdl1_tps", 0) >= 50 and ps <= 1:
                recommendations.append({
                    "treatment": "Pembrolizumab monotherapy",
                    "evidence": "NICE CG121 R6 + KEYNOTE-024",
                    "intent": "Palliative"
                })
            elif ps <= 1:
                recommendations.append({
                    "treatment": "Platinum-based doublet chemotherapy",
                    "evidence": "NICE CG121 R2",
                    "intent": "Palliative"
                })

        return {
            "status": "success",
            "cancer_type": "NSCLC",
            "stage": stage,
            "performance_status": ps,
            "biomarker_profile": biomarkers,
            "recommendations": recommendations
        }

    async def _handle_run_sclc_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """SCLC-specific recommendations"""
        patient_data = args.get("patient_data", {})
        stage = patient_data.get("tnm_stage", "")
        ps = patient_data.get("performance_status", 1)

        # SCLC is classified as Limited or Extensive
        if stage in ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB"]:
            sclc_stage = "Limited"
        else:
            sclc_stage = "Extensive"

        recommendations = []

        if sclc_stage == "Limited" and ps <= 2:
            recommendations.append({
                "treatment": "Concurrent chemoradiotherapy (Cisplatin + Etoposide with thoracic RT)",
                "evidence": "NICE CG121 R3",
                "intent": "Curative",
                "note": "Consider prophylactic cranial irradiation if good response"
            })
        elif sclc_stage == "Extensive" and ps <= 2:
            recommendations.append({
                "treatment": "Chemotherapy (Carboplatin + Etoposide) + Atezolizumab",
                "evidence": "IMpower133 trial",
                "intent": "Palliative"
            })
        else:
            recommendations.append({
                "treatment": "Best supportive care",
                "evidence": "NICE CG121 R7",
                "intent": "Palliative"
            })

        return {
            "status": "success",
            "cancer_type": "SCLC",
            "tnm_stage": stage,
            "sclc_classification": sclc_stage,
            "performance_status": ps,
            "recommendations": recommendations
        }

    async def _handle_recommend_biomarker_testing(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend biomarker tests"""
        patient_data = args.get("patient_data", {})
        histology = patient_data.get("histology_type", "")
        stage = patient_data.get("tnm_stage", "")

        tests = []

        # NSCLC - Advanced stage
        if histology in ["Adenocarcinoma", "LargeCellCarcinoma", "SquamousCellCarcinoma"]:
            if stage in ["IIIB", "IV", "IVA", "IVB"]:
                tests.extend([
                    {"test": "EGFR mutation", "priority": "High", "reason": "Identifies TKI eligibility"},
                    {"test": "ALK rearrangement", "priority": "High", "reason": "ALK inhibitor eligibility"},
                    {"test": "ROS1 rearrangement", "priority": "Medium", "reason": "ROS1 inhibitor option"},
                    {"test": "PD-L1 TPS", "priority": "High", "reason": "Immunotherapy decision"},
                    {"test": "BRAF V600E mutation", "priority": "Medium", "reason": "Targeted therapy option"},
                    {"test": "MET exon 14 skipping", "priority": "Medium", "reason": "MET inhibitor eligibility"}
                ])

        # Adenocarcinoma specific
        if histology == "Adenocarcinoma":
            tests.append({
                "test": "KRAS mutation", "priority": "Low", "reason": "Prognostic/clinical trial eligibility"
            })

        return {
            "status": "success",
            "histology": histology,
            "stage": stage,
            "recommended_tests": tests,
            "note": "Molecular testing recommended for all advanced NSCLC (NICE CG121)"
        }

    async def _handle_get_ontology_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get ontology statistics"""
        if self.ontology:
            stats = {
                "status": "success",
                "ontology": "LUCADA (Lung Cancer Data)",
                "version": "1.0",
                "loaded": True,
                "estimated_classes": 80,
                "estimated_properties": 35,
                "description": "OWL 2 ontology for lung cancer clinical decision support"
            }
        else:
            stats = {
                "status": "success",
                "ontology": "LUCADA",
                "loaded": False,
                "message": "Ontology not loaded. System will use rule-based reasoning."
            }

        return stats

    # ===========================================
    # SERVER RUN
    # ===========================================

    async def run(self):
        """Run the MCP server"""
        logger.info("=" * 60)
        logger.info("Starting Lung Cancer Assistant MCP Server (Fixed)")
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
    server = LCAMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
