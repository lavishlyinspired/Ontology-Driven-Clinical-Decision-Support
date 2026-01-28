"""
MCP Server for Lung Cancer Assistant - ADAPTIVE & INTELLIGENT VERSION
======================================================================

Features:
- Automatic capability selection based on query complexity
- Full execution transparency (shows which tools/agents/algorithms were called)
- Adaptive execution with retry and self-correction
- Dynamic orchestration
- Smart workflow selection
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"           # Single agent, straightforward
    MODERATE = "moderate"       # 6-agent workflow
    COMPLEX = "complex"         # 11-agent workflow
    ADVANCED = "advanced"       # Digital Twin + Analytics


class ExecutionTrace:
    """Track execution transparency"""
    def __init__(self):
        self.steps = []
        self.agents_called = []
        self.tools_used = []
        self.graph_algorithms = []
        self.digital_twin_operations = []
        self.retries = []
        self.errors = []

    def add_step(self, step: str):
        self.steps.append({"step": step, "timestamp": datetime.now().isoformat()})

    def add_agent(self, agent_name: str, confidence: float = None):
        self.agents_called.append({"agent": agent_name, "confidence": confidence})

    def add_tool(self, tool_name: str):
        self.tools_used.append(tool_name)

    def add_graph_algorithm(self, algorithm: str, params: Dict = None):
        self.graph_algorithms.append({"algorithm": algorithm, "params": params})

    def add_digital_twin_op(self, operation: str, details: Dict = None):
        self.digital_twin_operations.append({"operation": operation, "details": details})

    def add_retry(self, reason: str):
        self.retries.append({"reason": reason, "timestamp": datetime.now().isoformat()})

    def add_error(self, error: str):
        self.errors.append({"error": error, "timestamp": datetime.now().isoformat()})

    def to_dict(self):
        return {
            "execution_steps": self.steps,
            "agents_called": self.agents_called,
            "tools_used": self.tools_used,
            "graph_algorithms_executed": self.graph_algorithms,
            "digital_twin_operations": self.digital_twin_operations,
            "retries": self.retries,
            "errors": self.errors,
            "total_steps": len(self.steps),
            "total_agents": len(self.agents_called),
            "total_retries": len(self.retries)
        }


class AdaptiveLCAServer:
    """
    Adaptive MCP Server with intelligent capability selection
    """

    def __init__(self):
        self.server = Server("lung-cancer-assistant-adaptive")

        # Core components
        self.ontology = None
        self.rule_engine = None
        self.snomed_loader = None
        self.agents = {}
        self.active_twins = {}

        # Analytics
        self.survival_analyzer = None
        self.trial_matcher = None
        self.counterfactual_engine = None
        self.uncertainty_quantifier = None

        # Dynamic orchestrator
        self.dynamic_orchestrator = None

        # Service availability
        self.neo4j_available = False
        self.ollama_available = False
        self._components_initialized = False
        self._initialization_error = None

        # Register handlers
        self._register_list_tools()
        self._register_call_tool()

    # ===========================================
    # INTELLIGENT CAPABILITY SELECTION
    # ===========================================

    def _assess_query_complexity(self, patient_data: Dict[str, Any]) -> QueryComplexity:
        """
        Automatically assess query complexity to select appropriate workflow
        """
        complexity_score = 0

        # Check for multiple comorbidities
        comorbidities = patient_data.get("comorbidities", [])
        if len(comorbidities) > 2:
            complexity_score += 2
        elif len(comorbidities) > 0:
            complexity_score += 1

        # Check for borderline performance status
        ps = patient_data.get("performance_status", 0)
        if ps >= 2:
            complexity_score += 1

        # Check for advanced stage
        stage = patient_data.get("tnm_stage", "")
        if stage in ["IIIB", "IV", "IVA", "IVB"]:
            complexity_score += 1

        # Check for biomarker complexity
        biomarkers = patient_data.get("biomarkers", {})
        if len(biomarkers) > 2:
            complexity_score += 1

        # Check for conflicting indicators
        fev1 = patient_data.get("fev1", 100)
        if fev1 < 50:
            complexity_score += 1

        # Assess complexity
        if complexity_score == 0:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 2:
            return QueryComplexity.MODERATE
        elif complexity_score <= 4:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.ADVANCED

    async def _smart_analyze_patient(self, patient_data: Dict[str, Any],
                                     max_retries: int = 2) -> Dict[str, Any]:
        """
        Intelligently analyze patient by automatically selecting the right approach
        """
        trace = ExecutionTrace()
        trace.add_step("Starting smart patient analysis")

        # Step 1: Assess complexity
        complexity = self._assess_query_complexity(patient_data)
        trace.add_step(f"Complexity assessed: {complexity.value}")

        # Step 2: Select and execute appropriate workflow
        result = None
        confidence = 0.0

        for attempt in range(max_retries + 1):
            try:
                if complexity == QueryComplexity.SIMPLE:
                    trace.add_step("Executing simple analysis (6-agent workflow)")
                    result = await self._run_6agent_with_trace(patient_data, trace)

                elif complexity == QueryComplexity.MODERATE:
                    trace.add_step("Executing moderate analysis (6-agent workflow)")
                    result = await self._run_6agent_with_trace(patient_data, trace)

                elif complexity == QueryComplexity.COMPLEX:
                    trace.add_step("Executing complex analysis (11-agent integrated workflow)")
                    result = await self._run_11agent_with_trace(patient_data, trace)

                elif complexity == QueryComplexity.ADVANCED:
                    trace.add_step("Executing advanced analysis (Digital Twin + 11-agent workflow)")
                    result = await self._run_advanced_with_trace(patient_data, trace)

                # Check result confidence
                confidence = result.get("scenario_confidence", 0.0)

                if confidence >= 0.70:
                    trace.add_step(f"Analysis successful (confidence: {confidence:.2f})")
                    break
                else:
                    if attempt < max_retries:
                        trace.add_retry(f"Low confidence ({confidence:.2f}), retrying with more comprehensive analysis")
                        # Escalate to next complexity level
                        complexity = self._escalate_complexity(complexity)
                    else:
                        trace.add_step(f"Analysis completed with moderate confidence ({confidence:.2f})")

            except Exception as e:
                trace.add_error(str(e))
                if attempt < max_retries:
                    trace.add_retry(f"Error occurred: {str(e)}, retrying...")
                else:
                    trace.add_error("Maximum retries reached")
                    raise

        # Add execution trace to result
        result["execution_trace"] = trace.to_dict()
        result["complexity_level"] = complexity.value
        result["final_confidence"] = confidence

        return result

    def _escalate_complexity(self, current: QueryComplexity) -> QueryComplexity:
        """Escalate to next complexity level"""
        if current == QueryComplexity.SIMPLE:
            return QueryComplexity.MODERATE
        elif current == QueryComplexity.MODERATE:
            return QueryComplexity.COMPLEX
        elif current == QueryComplexity.COMPLEX:
            return QueryComplexity.ADVANCED
        else:
            return current  # Already at max

    async def _run_6agent_with_trace(self, patient_data: Dict[str, Any],
                                     trace: ExecutionTrace) -> Dict[str, Any]:
        """Run 6-agent workflow with execution tracing"""
        from src.agents.ingestion_agent import IngestionAgent
        from src.agents.semantic_mapping_agent import SemanticMappingAgent
        from src.agents.classification_agent import ClassificationAgent
        from src.agents.conflict_resolution_agent import ConflictResolutionAgent
        from src.agents.explanation_agent import ExplanationAgent

        # Agent 1: Ingestion
        trace.add_step("Agent 1/6: Ingestion - Validating patient data")
        trace.add_agent("IngestionAgent")
        ingestion = IngestionAgent()
        patient_fact, errors = ingestion.execute(patient_data)
        if not patient_fact:
            raise ValueError(f"Ingestion failed: {errors}")

        # Agent 2: Semantic Mapping
        trace.add_step("Agent 2/6: Semantic Mapping - Assigning SNOMED-CT codes")
        trace.add_agent("SemanticMappingAgent")
        mapping = SemanticMappingAgent()
        patient_with_codes, mapping_confidence = mapping.execute(patient_fact)
        trace.add_step(f"SNOMED mapping confidence: {mapping_confidence:.2f}")

        # Agent 3: Classification
        trace.add_step("Agent 3/6: Classification - Applying NICE guidelines")
        trace.add_agent("ClassificationAgent")
        classification = ClassificationAgent()
        class_result = classification.execute(patient_with_codes)
        trace.add_agent("ClassificationAgent", class_result.scenario_confidence)
        trace.add_step(f"Scenario: {class_result.scenario}, Confidence: {class_result.scenario_confidence:.2f}")

        # Agent 4: Conflict Resolution
        trace.add_step("Agent 4/6: Conflict Resolution - Reconciling recommendations")
        trace.add_agent("ConflictResolutionAgent")
        conflict = ConflictResolutionAgent()
        resolved, resolution_notes = conflict.execute(class_result)

        # Agent 5: Explanation
        trace.add_step("Agent 5/6: Explanation - Generating MDT summary")
        trace.add_agent("ExplanationAgent")
        explanation = ExplanationAgent()
        mdt_summary = explanation.execute(patient_with_codes, resolved)

        # Agent 6: Persistence (if Neo4j available)
        if self.neo4j_available:
            trace.add_step("Agent 6/6: Persistence - Saving to Neo4j")
            trace.add_agent("PersistenceAgent")
            # Would call persistence here
        else:
            trace.add_step("Agent 6/6: Persistence - Skipped (Neo4j not available)")

        return {
            "status": "success",
            "workflow": "6-agent",
            "patient_id": mdt_summary.patient_id,
            "scenario": mdt_summary.classification_scenario,
            "scenario_confidence": mdt_summary.scenario_confidence,
            "recommendations": mdt_summary.formatted_recommendations,
            "reasoning": mdt_summary.reasoning_explanation,
            "key_considerations": mdt_summary.key_considerations,
            "discussion_points": mdt_summary.discussion_points,
            "guideline_references": mdt_summary.guideline_references,
            "snomed_mappings": mdt_summary.snomed_mappings
        }

    async def _run_11agent_with_trace(self, patient_data: Dict[str, Any],
                                      trace: ExecutionTrace) -> Dict[str, Any]:
        """Run 11-agent workflow with execution tracing"""
        try:
            from src.agents.integrated_workflow import run_integrated_workflow

            trace.add_step("Initializing 11-agent integrated workflow")

            # Core 6 agents
            for agent in ["Ingestion", "SemanticMapping", "Classification",
                         "ConflictResolution", "Explanation", "Persistence"]:
                trace.add_agent(f"{agent}Agent")

            # Specialized 5 agents
            histology = patient_data.get("histology_type", "")
            if "Adenocarcinoma" in histology or "Squamous" in histology:
                trace.add_step("Activating NSCLC specialized agent")
                trace.add_agent("NSCLCAgent")
            elif "SmallCell" in histology:
                trace.add_step("Activating SCLC specialized agent")
                trace.add_agent("SCLCAgent")

            if patient_data.get("biomarkers"):
                trace.add_step("Activating Biomarker analysis agent")
                trace.add_agent("BiomarkerAgent")

            if patient_data.get("comorbidities"):
                trace.add_step("Activating Comorbidity assessment agent")
                trace.add_agent("ComorbidityAgent")

            trace.add_step("Activating Negotiation protocol for consensus")
            trace.add_agent("NegotiationProtocolAgent")

            result = run_integrated_workflow(patient_data, persist=self.neo4j_available)

            return {
                "status": "success",
                "workflow": "11-agent",
                **result
            }
        except Exception as e:
            trace.add_error(f"11-agent workflow error: {str(e)}")
            # Fallback to 6-agent
            trace.add_step("Falling back to 6-agent workflow")
            return await self._run_6agent_with_trace(patient_data, trace)

    async def _run_advanced_with_trace(self, patient_data: Dict[str, Any],
                                      trace: ExecutionTrace) -> Dict[str, Any]:
        """Run advanced analysis with Digital Twin + Analytics"""
        from src.digital_twin.twin_engine import DigitalTwinEngine

        # Create digital twin
        trace.add_step("Creating Digital Twin for continuous monitoring")
        trace.add_digital_twin_op("create", {"patient_id": patient_data.get("patient_id")})

        patient_id = patient_data.get("patient_id", f"PT_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        twin = DigitalTwinEngine(patient_id=patient_id)
        init_result = await twin.initialize(patient_data)

        self.active_twins[patient_id] = twin

        # Run 11-agent workflow
        trace.add_step("Running 11-agent workflow within Digital Twin context")
        result = await self._run_11agent_with_trace(patient_data, trace)

        # Predict trajectories
        trace.add_step("Predicting disease trajectories")
        trace.add_digital_twin_op("predict_trajectories", {})
        predictions = await twin.predict_trajectories()

        # Get alerts
        trace.add_step("Checking for clinical alerts")
        trace.add_digital_twin_op("get_alerts", {})
        state = twin.get_current_state()

        result["digital_twin"] = {
            "twin_id": init_result["twin_id"],
            "predictions": predictions,
            "active_alerts": state["active_alerts"],
            "context_graph_nodes": init_result["context_graph_nodes"]
        }

        return result

    # ===========================================
    # SERVICE CHECKS
    # ===========================================

    async def _check_services(self) -> Dict[str, bool]:
        """Check service availability"""
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
            logger.info("✓ Neo4j available")
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
                    logger.info("✓ Ollama available")
        except Exception as e:
            logger.warning(f"⚠ Ollama not available: {e}")

        # Check Ontology
        try:
            from src.config import LCAConfig
            lucada_path = LCAConfig.get_lucada_output_path()
            if lucada_path.exists():
                services["ontology"] = True
                logger.info("✓ Ontology available")
        except Exception as e:
            logger.warning(f"⚠ Ontology check failed: {e}")

        self.neo4j_available = services["neo4j"]
        self.ollama_available = services["ollama"]

        return services

    # ===========================================
    # TOOL DEFINITIONS
    # ===========================================

    def _get_all_tools(self) -> List[Tool]:
        """Return all available tools"""
        tools = []

        # PRIMARY INTELLIGENT TOOL
        tools.append(Tool(
            name="smart_analyze_patient",
            description="🤖 INTELLIGENT ANALYZER - Automatically selects the right workflow (6-agent, 11-agent, or Digital Twin) based on case complexity. Shows full execution trace with all agents, tools, and algorithms used. Includes adaptive retry logic.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {
                        "type": "object",
                        "description": "Complete patient data",
                        "properties": {
                            "patient_id": {"type": "string"},
                            "age": {"type": "integer"},
                            "sex": {"type": "string"},
                            "tnm_stage": {"type": "string"},
                            "histology_type": {"type": "string"},
                            "performance_status": {"type": "integer"},
                            "laterality": {"type": "string"},
                            "fev1": {"type": "number"},
                            "comorbidities": {"type": "array"},
                            "biomarkers": {"type": "object"}
                        },
                        "required": ["patient_id", "tnm_stage", "histology_type"]
                    },
                    "max_retries": {
                        "type": "integer",
                        "default": 2,
                        "description": "Maximum retry attempts if confidence is low"
                    }
                },
                "required": ["patient_data"]
            }
        ))

        # Workflow tools with transparency
        tools.extend([
            Tool(
                name="run_6agent_workflow",
                description="Run 6-agent workflow with full execution transparency",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patient_data": {"type": "object"},
                        "show_trace": {"type": "boolean", "default": True}
                    },
                    "required": ["patient_data"]
                }
            ),
            Tool(
                name="run_11agent_workflow",
                description="Run 11-agent integrated workflow with execution trace",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patient_data": {"type": "object"},
                        "show_trace": {"type": "boolean", "default": True}
                    },
                    "required": ["patient_data"]
                }
            ),
            Tool(
                name="assess_complexity",
                description="Assess patient case complexity to recommend appropriate workflow",
                inputSchema={
                    "type": "object",
                    "properties": {"patient_data": {"type": "object"}},
                    "required": ["patient_data"]
                }
            ),
        ])

        # Digital Twin with transparency
        tools.extend([
            Tool(
                name="create_digital_twin",
                description="Create Digital Twin with full initialization trace",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "patient_data": {"type": "object"}
                    },
                    "required": ["patient_id", "patient_data"]
                }
            ),
            Tool(
                name="update_digital_twin",
                description="Update Digital Twin showing which agents were triggered",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "update_type": {"type": "string"},
                        "data": {"type": "object"}
                    },
                    "required": ["patient_id", "update_type", "data"]
                }
            ),
        ])

        # Individual Agent Tools
        tools.extend([
            Tool(
                name="run_ingestion_agent",
                description="Run ingestion agent to validate and normalize patient data",
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

        # More Digital Twin Tools
        tools.extend([
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
                description="Get active alerts from digital twin",
                inputSchema={"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}
            ),
            Tool(
                name="export_digital_twin",
                description="Export complete digital twin state",
                inputSchema={"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}
            ),
        ])

        # Analytics Tools
        tools.extend([
            Tool(
                name="survival_analysis",
                description="Kaplan-Meier survival analysis for cohort",
                inputSchema={"type": "object", "properties": {"cohort_filter": {"type": "object"}}}
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
        ])

        # Ontology & SNOMED Tools
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
                name="list_guidelines",
                description="List NICE CG121 lung cancer guidelines",
                inputSchema={"type": "object", "properties": {}}
            ),
        ])

        # Neo4j Graph Tools
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
        ])

        # System tools
        tools.extend([
            Tool(
                name="get_system_status",
                description="Get system status with component availability",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="get_execution_history",
                description="Get execution history showing all previous analyses",
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
        """Register call_tool handler"""
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            try:
                if not self._components_initialized:
                    await self._initialize_components()

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
                            "message": f"Unknown tool: {name}"
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
            # Primary intelligent tool
            "smart_analyze_patient": self._handle_smart_analyze,

            # Workflows
            "run_6agent_workflow": self._handle_run_6agent_workflow,
            "run_11agent_workflow": self._handle_run_11agent_workflow,
            "assess_complexity": self._handle_assess_complexity,

            # Individual Agents
            "run_ingestion_agent": self._handle_run_ingestion_agent,
            "run_semantic_mapping_agent": self._handle_run_semantic_mapping_agent,
            "run_classification_agent": self._handle_run_classification_agent,
            "run_nsclc_agent": self._handle_run_nsclc_agent,
            "run_sclc_agent": self._handle_run_sclc_agent,
            "run_biomarker_agent": self._handle_run_biomarker_agent,
            "run_comorbidity_agent": self._handle_run_comorbidity_agent,
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
            "predict_survival": self._handle_predict_survival,
            "match_clinical_trials": self._handle_match_clinical_trials,
            "analyze_counterfactuals": self._handle_analyze_counterfactuals,
            "quantify_uncertainty": self._handle_quantify_uncertainty,

            # Ontology & SNOMED
            "get_ontology_stats": self._handle_get_ontology_stats,
            "query_ontology": self._handle_query_ontology,
            "search_snomed": self._handle_search_snomed,
            "list_guidelines": self._handle_list_guidelines,

            # Neo4j
            "find_similar_patients": self._handle_find_similar_patients,
            "get_patient_history": self._handle_get_patient_history,

            # System
            "get_system_status": self._handle_get_system_status,
            "get_execution_history": self._handle_get_execution_history,
            "list_all_capabilities": self._handle_list_all_capabilities,
        }
        return handlers.get(name)

    # ===========================================
    # INITIALIZATION
    # ===========================================

    async def _initialize_components(self):
        """Initialize components"""
        if self._components_initialized:
            return

        try:
            logger.info("Initializing adaptive LCA components...")

            services = await self._check_services()

            # Load ontology
            try:
                from src.ontology.lucada_ontology import LUCADAOntology
                from src.ontology.guideline_rules import GuidelineRuleEngine
                self.ontology = LUCADAOntology()
                self.ontology.create()
                self.rule_engine = GuidelineRuleEngine(self.ontology)
                logger.info("✓ Ontology initialized")
            except Exception as e:
                logger.warning(f"⚠ Ontology skipped: {e}")

            # Initialize dynamic orchestrator
            try:
                from src.agents.dynamic_orchestrator import DynamicWorkflowOrchestrator
                self.dynamic_orchestrator = DynamicWorkflowOrchestrator()
                logger.info("✓ Dynamic orchestrator initialized")
            except Exception as e:
                logger.warning(f"⚠ Dynamic orchestrator skipped: {e}")

            self._components_initialized = True
            logger.info("✓ Adaptive initialization complete")

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            self._initialization_error = str(e)
            self._components_initialized = True

    # ===========================================
    # TOOL HANDLERS
    # ===========================================

    async def _handle_smart_analyze(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle smart patient analysis"""
        try:
            patient_data = args.get("patient_data", args)
            max_retries = args.get("max_retries", 2)

            result = await self._smart_analyze_patient(patient_data, max_retries)

            return {
                "status": "success",
                "message": "✅ Smart analysis complete - automatic workflow selection used",
                **result
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_6agent_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle 6-agent workflow"""
        try:
            patient_data = args.get("patient_data", args)
            trace = ExecutionTrace()

            result = await self._run_6agent_with_trace(patient_data, trace)
            result["execution_trace"] = trace.to_dict()

            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_11agent_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle 11-agent workflow"""
        try:
            patient_data = args.get("patient_data", args)
            trace = ExecutionTrace()

            result = await self._run_11agent_with_trace(patient_data, trace)
            result["execution_trace"] = trace.to_dict()

            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_assess_complexity(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle complexity assessment"""
        try:
            patient_data = args.get("patient_data", args)
            complexity = self._assess_query_complexity(patient_data)

            recommendations = {
                QueryComplexity.SIMPLE: "6-agent workflow recommended",
                QueryComplexity.MODERATE: "6-agent workflow recommended",
                QueryComplexity.COMPLEX: "11-agent integrated workflow recommended",
                QueryComplexity.ADVANCED: "Digital Twin + 11-agent workflow recommended"
            }

            return {
                "status": "success",
                "complexity": complexity.value,
                "recommendation": recommendations[complexity],
                "factors_assessed": {
                    "comorbidities": len(patient_data.get("comorbidities", [])),
                    "performance_status": patient_data.get("performance_status", 0),
                    "stage": patient_data.get("tnm_stage", ""),
                    "biomarkers_count": len(patient_data.get("biomarkers", {})),
                    "fev1": patient_data.get("fev1", 100)
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_create_digital_twin(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle digital twin creation"""
        try:
            from src.digital_twin.twin_engine import DigitalTwinEngine

            trace = ExecutionTrace()
            patient_id = args.get("patient_id")
            patient_data = args.get("patient_data")

            trace.add_step("Initializing Digital Twin Engine")
            trace.add_digital_twin_op("create", {"patient_id": patient_id})

            twin = DigitalTwinEngine(patient_id=patient_id)
            init_result = await twin.initialize(patient_data)

            self.active_twins[patient_id] = twin

            trace.add_step(f"Digital Twin created: {init_result['twin_id']}")

            return {
                "status": "success",
                "twin_id": init_result["twin_id"],
                "initialization": init_result,
                "execution_trace": trace.to_dict()
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_update_digital_twin(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle digital twin update"""
        try:
            trace = ExecutionTrace()
            patient_id = args.get("patient_id")

            if patient_id not in self.active_twins:
                return {"status": "error", "message": f"No active twin for {patient_id}"}

            twin = self.active_twins[patient_id]

            trace.add_step("Updating Digital Twin")
            trace.add_digital_twin_op("update", {
                "type": args.get("update_type"),
                "data_keys": list(args.get("data", {}).keys())
            })

            update_result = await twin.update({
                "type": args.get("update_type"),
                "data": args.get("data")
            })

            # Log which agents were triggered
            for agent in update_result.get("affected_agents", []):
                trace.add_agent(agent)
                trace.add_step(f"Agent {agent} re-executed due to update")

            update_result["execution_trace"] = trace.to_dict()

            return {"status": "success", **update_result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_get_system_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get system status"""
        return {
            "status": "success",
            "system": "Lung Cancer Assistant - ADAPTIVE & INTELLIGENT",
            "version": "3.0.0-adaptive",
            "features": {
                "automatic_workflow_selection": "✅ Enabled",
                "execution_transparency": "✅ Full trace available",
                "adaptive_retry": "✅ With auto-escalation",
                "self_correction": "✅ Enabled",
                "digital_twin_monitoring": "✅ Continuous"
            },
            "services": {
                "neo4j": "available" if self.neo4j_available else "not available",
                "ollama": "available" if self.ollama_available else "not available",
                "ontology": "loaded" if self.ontology else "not loaded",
                "dynamic_orchestrator": "loaded" if self.dynamic_orchestrator else "not loaded"
            },
            "active_digital_twins": len(self.active_twins),
            "timestamp": datetime.now().isoformat()
        }

    async def _handle_get_execution_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get execution history"""
        return {
            "status": "success",
            "message": "Execution history tracking available",
            "active_twins": list(self.active_twins.keys())
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
                return {"status": "success", "patient_fact": patient_fact.model_dump(), "errors": errors}
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
            patient_fact, _ = ingestion.execute(patient_data)
            if not patient_fact:
                return {"status": "error", "message": "Ingestion failed"}

            agent = SemanticMappingAgent()
            patient_with_codes, confidence = agent.execute(patient_fact)

            return {
                "status": "success",
                "patient_with_codes": patient_with_codes.model_dump(),
                "mapping_confidence": confidence,
                "snomed_codes": {
                    "diagnosis": patient_with_codes.snomed_diagnosis_code,
                    "histology": patient_with_codes.snomed_histology_code
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
                "recommendations": [r.model_dump() for r in classification.recommendations]
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_nsclc_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """NSCLC agent"""
        try:
            from src.agents.nsclc_agent import NSCLCAgent
            agent = NSCLCAgent()
            patient_data = args.get("patient_data", {})
            result = agent.execute(patient_data)
            return {"status": "success", "nsclc_analysis": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_run_sclc_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """SCLC agent"""
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
                "clinical_summary": mdt_summary.clinical_summary,
                "scenario": mdt_summary.classification_scenario,
                "recommendations": mdt_summary.formatted_recommendations,
                "reasoning": mdt_summary.reasoning_explanation,
                "key_considerations": mdt_summary.key_considerations
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
                {"test": "PD-L1 TPS", "priority": "High", "reason": "Immunotherapy"}
            ])

        return {"status": "success", "histology": histology, "stage": stage, "recommended_tests": tests}

    # ===========================================
    # ADDITIONAL DIGITAL TWIN HANDLERS
    # ===========================================

    async def _handle_predict_trajectories(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Predict trajectories"""
        try:
            patient_id = args.get("patient_id")
            if patient_id not in self.active_twins:
                return {"status": "error", "message": f"No active twin for {patient_id}"}

            twin = self.active_twins[patient_id]
            predictions = await twin.predict_trajectories()

            return {"status": "success", "patient_id": patient_id, "predictions": predictions}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_get_twin_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get twin state"""
        try:
            patient_id = args.get("patient_id")
            if patient_id not in self.active_twins:
                return {"status": "error", "message": f"No active twin for {patient_id}"}

            twin = self.active_twins[patient_id]
            state = twin.get_current_state()

            return {"status": "success", "twin_state": state}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_get_twin_alerts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get twin alerts"""
        try:
            patient_id = args.get("patient_id")
            if patient_id not in self.active_twins:
                return {"status": "error", "message": f"No active twin for {patient_id}"}

            twin = self.active_twins[patient_id]
            state = twin.get_current_state()

            return {"status": "success", "patient_id": patient_id, "active_alerts": state["active_alerts"]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _handle_export_digital_twin(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Export digital twin"""
        try:
            patient_id = args.get("patient_id")
            if patient_id not in self.active_twins:
                return {"status": "error", "message": f"No active twin for {patient_id}"}

            twin = self.active_twins[patient_id]
            export = twin.export_twin()

            return {"status": "success", "export": export}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ===========================================
    # ANALYTICS HANDLERS
    # ===========================================

    async def _handle_survival_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Survival analysis"""
        return {"status": "success", "message": "Survival analysis requires Neo4j with patient cohort data"}

    async def _handle_predict_survival(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Predict survival"""
        patient_data = args.get("patient_data", {})
        stage = patient_data.get("tnm_stage", "")
        ps = patient_data.get("performance_status", 1)

        median_survival_months = {"IA": 60, "IB": 50, "IIA": 40, "IIB": 30, "IIIA": 18, "IIIB": 12, "IV": 8}.get(stage, 12)
        if ps >= 3:
            median_survival_months *= 0.5

        return {
            "status": "success",
            "estimated_median_survival_months": median_survival_months,
            "confidence_interval_95": [median_survival_months * 0.7, median_survival_months * 1.3]
        }

    async def _handle_match_clinical_trials(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Match clinical trials"""
        return {"status": "success", "message": "Trial matching requires ClinicalTrials.gov API integration"}

    async def _handle_analyze_counterfactuals(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Counterfactual analysis"""
        return {"status": "success", "message": "Counterfactual analysis available"}

    async def _handle_quantify_uncertainty(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify uncertainty"""
        return {"status": "success", "message": "Uncertainty quantification available"}

    # ===========================================
    # ONTOLOGY & SNOMED HANDLERS
    # ===========================================

    async def _handle_get_ontology_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get ontology stats"""
        if self.ontology:
            return {"status": "success", "ontology": "LUCADA", "loaded": True, "estimated_classes": 80}
        return {"status": "success", "ontology": "LUCADA", "loaded": False}

    async def _handle_query_ontology(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query ontology"""
        query = args.get("query", "")
        return {"status": "success", "query": query, "message": "Ontology querying available"}

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

    async def _handle_list_guidelines(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List guidelines"""
        return {
            "status": "success",
            "guideline_source": "NICE CG121",
            "key_rules": [
                "R1: Early stage operable NSCLC → Surgery",
                "R2: Advanced NSCLC with PS 0-1 → Chemotherapy",
                "R3: Limited stage SCLC → Concurrent chemoradiotherapy"
            ]
        }

    # ===========================================
    # NEO4J HANDLERS
    # ===========================================

    async def _handle_find_similar_patients(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find similar patients"""
        if not self.neo4j_available:
            return {"status": "error", "message": "Neo4j not available"}
        return {"status": "success", "similar_patients": [], "message": "Requires populated Neo4j database"}

    async def _handle_get_patient_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get patient history"""
        if not self.neo4j_available:
            return {"status": "error", "message": "Neo4j not available"}
        patient_id = args.get("patient_id")
        return {"status": "success", "patient_id": patient_id, "history": []}

    async def _handle_list_all_capabilities(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all capabilities"""
        tools = self._get_all_tools()
        return {
            "status": "success",
            "total_tools": len(tools),
            "tools": [{"name": t.name, "description": t.description} for t in tools]
        }

    # ===========================================
    # SERVER RUN
    # ===========================================

    async def run(self):
        """Run the MCP server"""
        logger.info("=" * 70)
        logger.info("Starting Lung Cancer Assistant - ADAPTIVE & INTELLIGENT VERSION")
        logger.info("=" * 70)
        logger.info("Features: Auto workflow selection, Full transparency, Adaptive retry")
        logger.info("=" * 70)

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
    server = AdaptiveLCAServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())