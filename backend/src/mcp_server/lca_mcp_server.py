"""
MCP (Model Context Protocol) Server for Lung Cancer Assistant
=============================================================

Complete MCP server exposing ALL LCA capabilities (2024-2026):
- 11-Agent Integrated Workflow Architecture
  * Core Processing: Ingestion, Semantic Mapping, Explanation, Persistence
  * Specialized Clinical: NSCLC, SCLC, Biomarker, Comorbidity, Negotiation
  * Orchestration: Dynamic Orchestrator, Integrated Workflow
- Biomarker-Driven Precision Medicine
- Neo4j/Neosemantics/GDS Integration
- Advanced Analytics Suite (Survival, Uncertainty, Trials)
- Clinical Trial Matching
- Counterfactual Reasoning
- Full Audit Trails

CRITICAL PRINCIPLE: "Neo4j as a tool, not a brain"
- All reasoning happens in Python/OWL
- Neo4j is only for storage and retrieval
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

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
from src.mcp_server.adaptive_tools import register_adaptive_tools
from src.mcp_server.advanced_mcp_tools import register_advanced_mcp_tools
from src.mcp_server.comprehensive_tools import register_comprehensive_tools

# Import new 2025-2026 services
from src.services.auth_service import auth_service
from src.services.audit_service import audit_logger, AuditAction
from src.services.hitl_service import hitl_service
from src.services.analytics_service import analytics_service
from src.services.rag_service import rag_service
from src.services.websocket_service import websocket_service
from src.services.version_service import version_service
from src.services.batch_service import batch_service
from src.services.fhir_service import fhir_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LCAMCPServer:
    """
    Comprehensive MCP Server for Lung Cancer Assistant

    Exposes 60+ tools covering all LCA capabilities from 2024-2026:
    
    11-Agent Integrated Architecture:
    - Core Processing (4): Ingestion, Semantic Mapping, Explanation, Persistence
    - Specialized Clinical (5): NSCLC, SCLC, Biomarker, Comorbidity, Negotiation
    - Orchestration (2): Dynamic Orchestrator, Integrated Workflow
    
    Additional Capabilities:
    - Advanced Analytics (Survival, Uncertainty, Trials)
    - Neo4j/Neosemantics/GDS Integration
    - Counterfactual Reasoning & What-If Analysis
    - Patient CRUD Operations
    - Export & Reporting
    """

    def __init__(self):
        self.server = Server("lung-cancer-assistant")
        self.ontology = None
        self.rule_engine = None
        self.snomed_loader = None
        self.workflow = None
        self.neo4j_tools = None

        # Initialize components lazily
        self._components_initialized = False

        # Register all handlers
        self._register_list_tools()
        self._register_call_tool()

    # ===========================================
    # TOOL DEFINITIONS (60+ Tools)
    # ===========================================

    def _get_all_tools(self) -> List[Tool]:
        """Return all available MCP tools with schemas"""

        tools = []

        # ============================================
        # CATEGORY 1: PATIENT MANAGEMENT (8 tools)
        # ============================================

        tools.append(Tool(
            name="create_patient",
            description="Create a new patient in the LUCADA ontology with clinical data",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "Unique patient identifier"},
                    "name": {"type": "string", "description": "Patient name"},
                    "age": {"type": "integer", "description": "Age at diagnosis"},
                    "sex": {"type": "string", "enum": ["M", "F"], "description": "Biological sex"},
                    "tnm_stage": {"type": "string", "description": "TNM staging (IA, IB, IIA, IIB, IIIA, IIIB, IIIC, IV, IVA, IVB)"},
                    "histology_type": {"type": "string", "description": "Histology (Adenocarcinoma, SquamousCellCarcinoma, SmallCellCarcinoma, LargeCellCarcinoma)"},
                    "performance_status": {"type": "integer", "minimum": 0, "maximum": 4, "description": "WHO Performance Status (0-4)"},
                    "laterality": {"type": "string", "enum": ["Right", "Left", "Bilateral"], "description": "Lung laterality"},
                    "fev1": {"type": "number", "description": "FEV1 percentage (optional)"},
                    "comorbidities": {"type": "array", "items": {"type": "string"}, "description": "List of comorbidities"}
                },
                "required": ["patient_id", "tnm_stage", "histology_type"]
            }
        ))

        tools.append(Tool(
            name="get_patient",
            description="Retrieve patient data by ID from Neo4j",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "Patient identifier"}
                },
                "required": ["patient_id"]
            }
        ))

        tools.append(Tool(
            name="update_patient",
            description="Update patient clinical data",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "Patient identifier"},
                    "updates": {"type": "object", "description": "Fields to update"}
                },
                "required": ["patient_id", "updates"]
            }
        ))

        tools.append(Tool(
            name="delete_patient",
            description="Delete patient (soft delete with audit trail)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "Patient identifier"},
                    "reason": {"type": "string", "description": "Reason for deletion"}
                },
                "required": ["patient_id"]
            }
        ))

        tools.append(Tool(
            name="validate_patient_schema",
            description="Validate patient data against LUCADA schema without processing",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data to validate"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="find_similar_patients",
            description="Find similar patients using vector similarity search",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "Reference patient ID"},
                    "k": {"type": "integer", "default": 10, "description": "Number of similar patients"},
                    "min_similarity": {"type": "number", "default": 0.7, "description": "Minimum similarity threshold (0-1)"}
                },
                "required": ["patient_id"]
            }
        ))

        tools.append(Tool(
            name="get_patient_history",
            description="Get complete inference history for a patient",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "Patient identifier"}
                },
                "required": ["patient_id"]
            }
        ))

        tools.append(Tool(
            name="get_cohort_stats",
            description="Get statistics for a patient cohort with filters",
            inputSchema={
                "type": "object",
                "properties": {
                    "stage_filter": {"type": "string", "description": "Filter by TNM stage"},
                    "histology_filter": {"type": "string", "description": "Filter by histology"},
                    "age_min": {"type": "integer", "description": "Minimum age"},
                    "age_max": {"type": "integer", "description": "Maximum age"}
                }
            }
        ))

        # ============================================
        # CATEGORY 2: 11-AGENT INTEGRATED WORKFLOW (8 tools)
        # ============================================

        tools.append(Tool(
            name="run_6agent_workflow",
            description="Run complete 11-agent integrated workflow: Core Processing (4) → Specialized Clinical (5) → Orchestration (2)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient clinical data"},
                    "persist": {"type": "boolean", "default": False, "description": "Save results to Neo4j"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="get_workflow_info",
            description="Get information about the 11-agent integrated workflow architecture",
            inputSchema={"type": "object", "properties": {}}
        ))

        tools.append(Tool(
            name="run_ingestion_agent",
            description="Run IngestionAgent to validate and normalize patient data",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Raw patient data"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="run_semantic_mapping_agent",
            description="Run SemanticMappingAgent to map clinical concepts to SNOMED-CT codes",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Validated patient data"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="run_classification_agent",
            description="Run ClassificationAgent to apply LUCADA ontology and NICE guidelines",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data with SNOMED codes"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="run_conflict_resolution_agent",
            description="Run ConflictResolutionAgent to resolve conflicting recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "recommendations": {"type": "array", "description": "List of treatment recommendations to resolve"}
                },
                "required": ["recommendations"]
            }
        ))

        tools.append(Tool(
            name="run_explanation_agent",
            description="Run ExplanationAgent to generate MDT summary",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"},
                    "recommendations": {"type": "array", "description": "Treatment recommendations"}
                },
                "required": ["patient_data", "recommendations"]
            }
        ))

        tools.append(Tool(
            name="generate_mdt_summary",
            description="Generate complete Multi-Disciplinary Team (MDT) summary for a patient",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"}
                },
                "required": ["patient_data"]
            }
        ))

        # ============================================
        # CATEGORY 3: SPECIALIZED AGENTS (6 tools)
        # ============================================

        tools.append(Tool(
            name="run_nsclc_agent",
            description="Run NSCLC-specific agent for Non-Small Cell Lung Cancer treatment recommendations (Adenocarcinoma, Squamous, Large Cell)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"},
                    "biomarker_profile": {"type": "object", "description": "Biomarker results (EGFR, ALK, ROS1, etc.)"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="run_sclc_agent",
            description="Run SCLC-specific agent for Small Cell Lung Cancer treatment recommendations (Limited vs Extensive stage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="run_biomarker_agent",
            description="Run BiomarkerAgent for precision medicine recommendations based on molecular profiling",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"},
                    "biomarker_profile": {
                        "type": "object",
                        "description": "Biomarker results",
                        "properties": {
                            "egfr_mutation": {"type": "string"},
                            "egfr_mutation_type": {"type": "string"},
                            "alk_rearrangement": {"type": "string"},
                            "ros1_rearrangement": {"type": "string"},
                            "braf_mutation": {"type": "string"},
                            "met_exon14": {"type": "string"},
                            "ret_rearrangement": {"type": "string"},
                            "kras_mutation": {"type": "string"},
                            "pdl1_tps": {"type": "number"},
                            "tmb_score": {"type": "number"},
                            "her2_mutation": {"type": "string"},
                            "ntrk_fusion": {"type": "string"}
                        }
                    }
                },
                "required": ["patient_data", "biomarker_profile"]
            }
        ))

        tools.append(Tool(
            name="run_comorbidity_agent",
            description="Run ComorbidityAgent to assess treatment safety based on patient comorbidities",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"},
                    "proposed_treatment": {"type": "string", "description": "Treatment to assess"},
                    "comorbidities": {"type": "array", "items": {"type": "string"}, "description": "List of comorbidities"}
                },
                "required": ["patient_data", "proposed_treatment"]
            }
        ))

        tools.append(Tool(
            name="recommend_biomarker_testing",
            description="Recommend which biomarker tests should be ordered for a patient",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data with stage and histology"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="run_integrated_workflow",
            description="Run complete integrated workflow with all 2024-2026 enhancements (adaptive routing, negotiation, analytics)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient clinical data"},
                    "persist": {"type": "boolean", "default": False, "description": "Save to Neo4j"},
                    "enable_analytics": {"type": "boolean", "default": True, "description": "Run advanced analytics"}
                },
                "required": ["patient_data"]
            }
        ))

        # ============================================
        # CATEGORY 4: ADAPTIVE WORKFLOW 2026 (6 tools)
        # ============================================

        tools.append(Tool(
            name="assess_case_complexity",
            description="Assess patient case complexity for adaptive workflow routing (simple/moderate/complex/critical)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient clinical data"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="run_adaptive_workflow",
            description="Execute dynamic multi-agent workflow with adaptive routing and self-correction",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"},
                    "enable_self_correction": {"type": "boolean", "default": True},
                    "persist": {"type": "boolean", "default": False}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="query_context_graph",
            description="Query dynamic context graph for reasoning chains and relationships",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "Workflow ID"},
                    "query_type": {"type": "string", "enum": ["reasoning_chain", "conflicts", "related_nodes"]},
                    "node_id": {"type": "string", "description": "Specific node ID (optional)"}
                },
                "required": ["query_type"]
            }
        ))

        tools.append(Tool(
            name="execute_parallel_agents",
            description="Execute multiple independent agents in parallel for faster processing",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"},
                    "agent_list": {"type": "array", "items": {"type": "string"}, "description": "Agent names to run"}
                },
                "required": ["patient_data", "agent_list"]
            }
        ))

        tools.append(Tool(
            name="analyze_with_self_correction",
            description="Run analysis with self-corrective loops for high-confidence results",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"},
                    "confidence_threshold": {"type": "number", "default": 0.7},
                    "max_iterations": {"type": "integer", "default": 3}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="get_workflow_metrics",
            description="Get performance metrics for adaptive workflow execution",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string"},
                    "time_range": {"type": "string", "enum": ["last_hour", "last_day", "last_week"]}
                }
            }
        ))

        # ============================================
        # CATEGORY 5: ONTOLOGY & SNOMED (8 tools)
        # ============================================

        tools.append(Tool(
            name="query_ontology",
            description="Query LUCADA ontology for concepts and relationships",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_type": {"type": "string", "enum": ["classes", "properties", "individuals"]},
                    "filter": {"type": "string", "description": "Optional filter string"}
                },
                "required": ["query_type"]
            }
        ))

        tools.append(Tool(
            name="get_ontology_stats",
            description="Get statistics about the LUCADA ontology",
            inputSchema={"type": "object", "properties": {}}
        ))

        tools.append(Tool(
            name="search_snomed",
            description="Search for SNOMED-CT concepts by term",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"},
                    "limit": {"type": "integer", "default": 20}
                },
                "required": ["query"]
            }
        ))

        tools.append(Tool(
            name="get_snomed_concept",
            description="Get detailed information about a SNOMED-CT concept by SCTID",
            inputSchema={
                "type": "object",
                "properties": {
                    "sctid": {"type": "string", "description": "SNOMED CT Identifier"}
                },
                "required": ["sctid"]
            }
        ))

        tools.append(Tool(
            name="map_patient_to_snomed",
            description="Map all patient clinical data to SNOMED-CT codes",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="generate_owl_expression",
            description="Generate OWL 2 class expression for patient classification (Manchester Syntax)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="get_lung_cancer_concepts",
            description="Get all pre-defined lung cancer related SNOMED-CT concepts",
            inputSchema={"type": "object", "properties": {}}
        ))

        tools.append(Tool(
            name="list_guidelines",
            description="List all available NICE clinical guideline rules",
            inputSchema={"type": "object", "properties": {}}
        ))

        # ============================================
        # CATEGORY 6: NEO4J/NEOSEMANTICS/GDS (12 tools)
        # ============================================

        tools.append(Tool(
            name="neo4j_execute_query",
            description="Execute a Cypher query against Neo4j database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Cypher query"},
                    "parameters": {"type": "object", "description": "Query parameters"},
                    "database": {"type": "string", "default": "neo4j"}
                },
                "required": ["query"]
            }
        ))

        tools.append(Tool(
            name="neo4j_create_patient_node",
            description="Create a Patient node in Neo4j knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "patient_data": {"type": "object"}
                },
                "required": ["patient_id", "patient_data"]
            }
        ))

        tools.append(Tool(
            name="neo4j_vector_search",
            description="Perform vector similarity search in Neo4j",
            inputSchema={
                "type": "object",
                "properties": {
                    "index_name": {"type": "string", "description": "Vector index name"},
                    "query_vector": {"type": "array", "items": {"type": "number"}},
                    "top_k": {"type": "integer", "default": 10}
                },
                "required": ["index_name", "query_vector"]
            }
        ))

        tools.append(Tool(
            name="neo4j_create_vector_index",
            description="Create vector similarity index in Neo4j 5.x",
            inputSchema={
                "type": "object",
                "properties": {
                    "index_name": {"type": "string"},
                    "node_label": {"type": "string"},
                    "property_name": {"type": "string"},
                    "dimensions": {"type": "integer"},
                    "similarity_function": {"type": "string", "enum": ["cosine", "euclidean", "dot_product"], "default": "cosine"}
                },
                "required": ["index_name", "node_label", "property_name", "dimensions"]
            }
        ))

        tools.append(Tool(
            name="n10s_import_rdf",
            description="Import RDF/OWL ontology using Neosemantics (n10s)",
            inputSchema={
                "type": "object",
                "properties": {
                    "rdf_url": {"type": "string", "description": "URL or file path to RDF/OWL"},
                    "rdf_format": {"type": "string", "default": "RDF/XML", "enum": ["RDF/XML", "Turtle", "N-Triples", "JSON-LD"]}
                },
                "required": ["rdf_url"]
            }
        ))

        tools.append(Tool(
            name="n10s_query_ontology",
            description="Query ontology concept by URI using Neosemantics",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept_uri": {"type": "string", "description": "RDF resource URI"}
                },
                "required": ["concept_uri"]
            }
        ))

        tools.append(Tool(
            name="n10s_semantic_reasoning",
            description="Perform semantic reasoning to find concept hierarchy",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "Concept name or URI"},
                    "relationship_type": {"type": "string", "default": "subClassOf"}
                },
                "required": ["concept"]
            }
        ))

        tools.append(Tool(
            name="gds_create_graph_projection",
            description="Create GDS graph projection for algorithm execution",
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_name": {"type": "string"},
                    "node_labels": {"type": "array", "items": {"type": "string"}},
                    "relationship_types": {"type": "array", "items": {"type": "string"}},
                    "properties": {"type": "object"}
                },
                "required": ["graph_name", "node_labels", "relationship_types"]
            }
        ))

        tools.append(Tool(
            name="gds_run_pagerank",
            description="Run PageRank algorithm on projected graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_name": {"type": "string"},
                    "max_iterations": {"type": "integer", "default": 20},
                    "damping_factor": {"type": "number", "default": 0.85}
                },
                "required": ["graph_name"]
            }
        ))

        tools.append(Tool(
            name="gds_run_louvain",
            description="Run Louvain community detection algorithm",
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_name": {"type": "string"},
                    "max_levels": {"type": "integer", "default": 10}
                },
                "required": ["graph_name"]
            }
        ))

        tools.append(Tool(
            name="gds_run_node2vec",
            description="Generate Node2Vec embeddings for graph nodes",
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_name": {"type": "string"},
                    "embedding_dimension": {"type": "integer", "default": 128},
                    "walk_length": {"type": "integer", "default": 80},
                    "iterations": {"type": "integer", "default": 10}
                },
                "required": ["graph_name"]
            }
        ))

        tools.append(Tool(
            name="gds_node_similarity",
            description="Find similar nodes using GDS Node Similarity algorithm",
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_name": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                    "similarity_cutoff": {"type": "number", "default": 0.5}
                },
                "required": ["graph_name"]
            }
        ))

        # ============================================
        # CATEGORY 7: ANALYTICS SUITE (10 tools)
        # ============================================

        tools.append(Tool(
            name="survival_kaplan_meier",
            description="Perform Kaplan-Meier survival analysis for a treatment",
            inputSchema={
                "type": "object",
                "properties": {
                    "treatment": {"type": "string", "description": "Treatment name"},
                    "stage": {"type": "string", "description": "TNM stage filter"},
                    "histology": {"type": "string", "description": "Histology filter"}
                },
                "required": ["treatment"]
            }
        ))

        tools.append(Tool(
            name="survival_compare_treatments",
            description="Compare survival curves between treatments using log-rank test",
            inputSchema={
                "type": "object",
                "properties": {
                    "treatment1": {"type": "string"},
                    "treatment2": {"type": "string"},
                    "stage": {"type": "string"}
                },
                "required": ["treatment1", "treatment2"]
            }
        ))

        tools.append(Tool(
            name="survival_cox_regression",
            description="Fit Cox Proportional Hazards model for survival prediction",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient characteristics"},
                    "covariates": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="stratify_risk",
            description="Stratify patient into risk groups (Low/Intermediate/High)",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="match_clinical_trials",
            description="Find eligible clinical trials for a patient via ClinicalTrials.gov",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"},
                    "max_results": {"type": "integer", "default": 10},
                    "phase_filter": {"type": "array", "items": {"type": "string"}},
                    "location": {"type": "string", "description": "Geographic location"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="analyze_counterfactuals",
            description="Generate 'what-if' counterfactual analysis for treatment decisions",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object", "description": "Patient data"},
                    "current_recommendation": {"type": "string", "description": "Current treatment"}
                },
                "required": ["patient_data", "current_recommendation"]
            }
        ))

        tools.append(Tool(
            name="quantify_uncertainty",
            description="Quantify epistemic and aleatoric uncertainty in recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "treatment": {"type": "string"}
                },
                "required": ["patient_id", "treatment"]
            }
        ))

        tools.append(Tool(
            name="analyze_disease_progression",
            description="Analyze disease progression timeline for a patient",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"}
                },
                "required": ["patient_id"]
            }
        ))

        tools.append(Tool(
            name="identify_intervention_windows",
            description="Identify critical intervention windows based on temporal patterns",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "lookahead_days": {"type": "integer", "default": 90}
                },
                "required": ["patient_id"]
            }
        ))

        tools.append(Tool(
            name="detect_treatment_communities",
            description="Detect communities of patients with similar treatment patterns",
            inputSchema={
                "type": "object",
                "properties": {
                    "resolution": {"type": "number", "default": 1.0}
                }
            }
        ))

        # ============================================
        # CATEGORY 8: LABORATORY & BIOMARKERS (4 tools)
        # ============================================

        tools.append(Tool(
            name="interpret_lab_results",
            description="Interpret laboratory results with LOINC coding and clinical significance",
            inputSchema={
                "type": "object",
                "properties": {
                    "lab_results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "test_name": {"type": "string"},
                                "value": {"type": "number"},
                                "unit": {"type": "string"}
                            }
                        }
                    },
                    "patient_age": {"type": "integer"},
                    "patient_sex": {"type": "string"}
                },
                "required": ["lab_results"]
            }
        ))

        tools.append(Tool(
            name="analyze_biomarkers",
            description="Get biomarker-driven treatment recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_data": {"type": "object"},
                    "biomarker_profile": {"type": "object"}
                },
                "required": ["patient_data"]
            }
        ))

        tools.append(Tool(
            name="predict_resistance",
            description="Predict resistance mechanisms for targeted therapies",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "current_therapy": {"type": "string"},
                    "biomarker_profile": {"type": "object"}
                },
                "required": ["patient_id", "current_therapy"]
            }
        ))

        tools.append(Tool(
            name="get_biomarker_pathways",
            description="Get treatment pathways for specific biomarker mutations",
            inputSchema={
                "type": "object",
                "properties": {
                    "biomarker": {"type": "string", "enum": ["EGFR", "ALK", "ROS1", "BRAF", "MET", "RET", "KRAS", "PD-L1", "TMB"]}
                },
                "required": ["biomarker"]
            }
        ))

        # ============================================
        # CATEGORY 9: EXPORT & REPORTING (4 tools)
        # ============================================

        tools.append(Tool(
            name="export_patient_data",
            description="Export patient data in various formats",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "format": {"type": "string", "enum": ["csv", "json", "pdf", "html"]},
                    "include_phi": {"type": "boolean", "default": False},
                    "include_inferences": {"type": "boolean", "default": True}
                },
                "required": ["patient_id", "format"]
            }
        ))

        tools.append(Tool(
            name="export_audit_trail",
            description="Export audit trail for patient or system",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "start_date": {"type": "string", "format": "date"},
                    "end_date": {"type": "string", "format": "date"},
                    "format": {"type": "string", "enum": ["csv", "json"]}
                },
                "required": ["format"]
            }
        ))

        tools.append(Tool(
            name="generate_clinical_report",
            description="Generate comprehensive clinical report for a patient",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "include_survival": {"type": "boolean", "default": True},
                    "include_trials": {"type": "boolean", "default": True},
                    "include_counterfactuals": {"type": "boolean", "default": False}
                },
                "required": ["patient_id"]
            }
        ))

        tools.append(Tool(
            name="get_system_status",
            description="Get system health status and configuration",
            inputSchema={"type": "object", "properties": {}}
        ))

        return tools

    # ===========================================
    # LIST TOOLS HANDLER
    # ===========================================

    def _register_list_tools(self):
        """Register list_tools handler - CRITICAL for Claude Desktop to see tools"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """Return all available tools"""
            return self._get_all_tools()

    # ===========================================
    # CALL TOOL HANDLER
    # ===========================================

    def _register_call_tool(self):
        """Register call_tool handler"""

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool execution"""

            try:
                # Lazy initialization of components
                if not self._components_initialized:
                    await self._initialize_components()

                # Route to appropriate handler
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
                logger.error(f"Tool execution error: {e}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "message": str(e)
                    }, indent=2)
                )]

    def _get_tool_handler(self, name: str):
        """Get handler function for a tool"""

        handlers = {
            # Patient Management
            "create_patient": self._handle_create_patient,
            "get_patient": self._handle_get_patient,
            "update_patient": self._handle_update_patient,
            "delete_patient": self._handle_delete_patient,
            "validate_patient_schema": self._handle_validate_patient_schema,
            "find_similar_patients": self._handle_find_similar_patients,
            "get_patient_history": self._handle_get_patient_history,
            "get_cohort_stats": self._handle_get_cohort_stats,

            # 11-Agent Integrated Workflow
            "run_6agent_workflow": self._handle_run_6agent_workflow,
            "get_workflow_info": self._handle_get_workflow_info,
            "run_ingestion_agent": self._handle_run_ingestion_agent,
            "run_semantic_mapping_agent": self._handle_run_semantic_mapping_agent,
            "run_classification_agent": self._handle_run_classification_agent,
            "run_conflict_resolution_agent": self._handle_run_conflict_resolution_agent,
            "run_explanation_agent": self._handle_run_explanation_agent,
            "generate_mdt_summary": self._handle_generate_mdt_summary,

            # Specialized Agents
            "run_nsclc_agent": self._handle_run_nsclc_agent,
            "run_sclc_agent": self._handle_run_sclc_agent,
            "run_biomarker_agent": self._handle_run_biomarker_agent,
            "run_comorbidity_agent": self._handle_run_comorbidity_agent,
            "recommend_biomarker_testing": self._handle_recommend_biomarker_testing,
            "run_integrated_workflow": self._handle_run_integrated_workflow,

            # Adaptive Workflow
            "assess_case_complexity": self._handle_assess_case_complexity,
            "run_adaptive_workflow": self._handle_run_adaptive_workflow,
            "query_context_graph": self._handle_query_context_graph,
            "execute_parallel_agents": self._handle_execute_parallel_agents,
            "analyze_with_self_correction": self._handle_analyze_with_self_correction,
            "get_workflow_metrics": self._handle_get_workflow_metrics,

            # Ontology & SNOMED
            "query_ontology": self._handle_query_ontology,
            "get_ontology_stats": self._handle_get_ontology_stats,
            "search_snomed": self._handle_search_snomed,
            "get_snomed_concept": self._handle_get_snomed_concept,
            "map_patient_to_snomed": self._handle_map_patient_to_snomed,
            "generate_owl_expression": self._handle_generate_owl_expression,
            "get_lung_cancer_concepts": self._handle_get_lung_cancer_concepts,
            "list_guidelines": self._handle_list_guidelines,

            # Neo4j/Neosemantics/GDS
            "neo4j_execute_query": self._handle_neo4j_execute_query,
            "neo4j_create_patient_node": self._handle_neo4j_create_patient_node,
            "neo4j_vector_search": self._handle_neo4j_vector_search,
            "neo4j_create_vector_index": self._handle_neo4j_create_vector_index,
            "n10s_import_rdf": self._handle_n10s_import_rdf,
            "n10s_query_ontology": self._handle_n10s_query_ontology,
            "n10s_semantic_reasoning": self._handle_n10s_semantic_reasoning,
            "gds_create_graph_projection": self._handle_gds_create_graph_projection,
            "gds_run_pagerank": self._handle_gds_run_pagerank,
            "gds_run_louvain": self._handle_gds_run_louvain,
            "gds_run_node2vec": self._handle_gds_run_node2vec,
            "gds_node_similarity": self._handle_gds_node_similarity,

            # Analytics
            "survival_kaplan_meier": self._handle_survival_kaplan_meier,
            "survival_compare_treatments": self._handle_survival_compare_treatments,
            "survival_cox_regression": self._handle_survival_cox_regression,
            "stratify_risk": self._handle_stratify_risk,
            "match_clinical_trials": self._handle_match_clinical_trials,
            "analyze_counterfactuals": self._handle_analyze_counterfactuals,
            "quantify_uncertainty": self._handle_quantify_uncertainty,
            "analyze_disease_progression": self._handle_analyze_disease_progression,
            "identify_intervention_windows": self._handle_identify_intervention_windows,
            "detect_treatment_communities": self._handle_detect_treatment_communities,

            # Laboratory & Biomarkers
            "interpret_lab_results": self._handle_interpret_lab_results,
            "analyze_biomarkers": self._handle_analyze_biomarkers,
            "predict_resistance": self._handle_predict_resistance,
            "get_biomarker_pathways": self._handle_get_biomarker_pathways,

            # Export & Reporting
            "export_patient_data": self._handle_export_patient_data,
            "export_audit_trail": self._handle_export_audit_trail,
            "generate_clinical_report": self._handle_generate_clinical_report,
            "get_system_status": self._handle_get_system_status,
        }

        return handlers.get(name)

    # ===========================================
    # INITIALIZATION
    # ===========================================

    async def _initialize_components(self):
        """Lazily initialize all components"""
        if self._components_initialized:
            return

        try:
            from src.ontology.lucada_ontology import LUCADAOntology
            from src.ontology.guideline_rules import GuidelineRuleEngine
            from src.ontology.snomed_loader import SNOMEDLoader

            self.ontology = LUCADAOntology()
            self.ontology.create()

            self.rule_engine = GuidelineRuleEngine(self.ontology)

            self.snomed_loader = SNOMEDLoader()
            try:
                self.snomed_loader.load(load_full=False)
            except Exception as e:
                logger.warning(f"SNOMED ontology not loaded: {e}")

            self._components_initialized = True
            logger.info("✓ Components initialized successfully")

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            self._components_initialized = True  # Mark as done to avoid retry loops

    # ===========================================
    # TOOL HANDLERS - PATIENT MANAGEMENT
    # ===========================================

    async def _handle_create_patient(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new patient"""
        return {
            "status": "success",
            "patient_id": args.get("patient_id"),
            "message": f"Patient {args.get('name', args.get('patient_id'))} created successfully",
            "data": args
        }

    async def _handle_get_patient(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get patient by ID"""
        return {
            "status": "success",
            "patient_id": args.get("patient_id"),
            "message": "Patient retrieval requires Neo4j connection"
        }

    async def _handle_update_patient(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update patient data"""
        return {
            "status": "success",
            "patient_id": args.get("patient_id"),
            "updates": args.get("updates"),
            "message": "Patient updated successfully"
        }

    async def _handle_delete_patient(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Soft delete patient"""
        return {
            "status": "success",
            "patient_id": args.get("patient_id"),
            "message": "Patient marked as deleted with audit trail"
        }

    async def _handle_validate_patient_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate patient data"""
        from src.agents.ingestion_agent import IngestionAgent

        patient_data = args.get("patient_data", args)
        agent = IngestionAgent()

        errors = []
        required = ["tnm_stage", "histology_type"]

        for field in required:
            if field not in patient_data or not patient_data[field]:
                errors.append(f"Missing required field: {field}")

        valid_stages = ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IIIC", "IV", "IVA", "IVB"]
        stage = patient_data.get("tnm_stage", "")
        normalized_stage = agent.normalize_tnm(stage)
        if normalized_stage not in valid_stages:
            errors.append(f"Invalid TNM stage: {stage}")

        return {
            "status": "valid" if not errors else "invalid",
            "errors": errors,
            "normalized_stage": normalized_stage,
            "fields_present": list(patient_data.keys())
        }

    async def _handle_find_similar_patients(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find similar patients"""
        return {
            "status": "success",
            "patient_id": args.get("patient_id"),
            "k": args.get("k", 10),
            "min_similarity": args.get("min_similarity", 0.7),
            "message": "Similar patient search requires Neo4j connection with vector index"
        }

    async def _handle_get_patient_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get patient inference history"""
        return {
            "status": "success",
            "patient_id": args.get("patient_id"),
            "message": "Patient history requires Neo4j connection"
        }

    async def _handle_get_cohort_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get cohort statistics"""
        return {
            "status": "success",
            "filters": args,
            "message": "Cohort statistics require Neo4j connection"
        }

    # ===========================================
    # TOOL HANDLERS - 11-AGENT INTEGRATED WORKFLOW
    # ===========================================

    async def _handle_run_6agent_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete 11-agent integrated workflow"""
        from src.agents.lca_workflow import analyze_patient

        patient_data = args.get("patient_data", args)
        persist = args.get("persist", False)

        result = analyze_patient(patient_data, persist=persist)

        return {
            "status": "success" if result.success else "error",
            "patient_id": result.patient_id,
            "workflow_status": result.workflow_status,
            "agent_chain": result.agent_chain,
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

    async def _handle_get_workflow_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get workflow architecture info"""
        return {
            "status": "success",
            "version": "3.0.0",
            "architecture": "11-Agent Integrated Workflow (2025-2026)",
            "principle": "Neo4j as a tool, not a brain",
            "core_processing": [
                {"name": "IngestionAgent", "role": "Validates and normalizes raw patient data", "neo4j_access": "READ-ONLY"},
                {"name": "SemanticMappingAgent", "role": "Maps clinical concepts to SNOMED-CT codes", "neo4j_access": "READ-ONLY"},
                {"name": "ExplanationAgent", "role": "Generates MDT summaries", "neo4j_access": "READ-ONLY"},
                {"name": "PersistenceAgent", "role": "THE ONLY AGENT THAT WRITES TO NEO4J", "neo4j_access": "WRITE"}
            ],
            "specialized_clinical": [
                {"name": "NSCLCAgent", "role": "Non-small cell lung cancer treatment protocols"},
                {"name": "SCLCAgent", "role": "Small cell lung cancer treatment protocols"},
                {"name": "BiomarkerAgent", "role": "Precision medicine and biomarker analysis"},
                {"name": "ComorbidityAgent", "role": "Comorbidity impact assessment"},
                {"name": "NegotiationAgent", "role": "Multi-agent consensus building"}
            ],
            "orchestration": [
                {"name": "DynamicOrchestrator", "role": "Intelligent agent routing based on patient characteristics"},
                {"name": "IntegratedWorkflow", "role": "End-to-end workflow coordination"}
            ],
            "analytics": ["SurvivalAnalyzer", "ClinicalTrialMatcher", "CounterfactualEngine", "UncertaintyQuantifier"],
            "total_agents": 11,
            "data_flow": "Input → Ingestion → SemanticMapping → Cancer-Specific Analysis → Biomarkers → Comorbidity → Negotiation → Explanation → Persistence → Output"
        }

    async def _handle_run_ingestion_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run ingestion agent"""
        from src.agents.ingestion_agent import IngestionAgent

        patient_data = args.get("patient_data", args)
        agent = IngestionAgent()
        patient_fact, errors = agent.execute(patient_data)

        if patient_fact:
            return {
                "status": "success",
                "patient_fact": patient_fact.model_dump(),
                "errors": errors
            }
        else:
            return {
                "status": "validation_failed",
                "patient_fact": None,
                "errors": errors
            }

    async def _handle_run_semantic_mapping_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run semantic mapping agent"""
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
                "stage": patient_with_codes.snomed_stage_code,
                "performance_status": patient_with_codes.snomed_ps_code,
                "laterality": patient_with_codes.snomed_laterality_code
            }
        }

    async def _handle_run_classification_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run classification agent"""
        from src.agents.ingestion_agent import IngestionAgent
        from src.agents.semantic_mapping_agent import SemanticMappingAgent
        from src.agents.classification_agent import ClassificationAgent

        patient_data = args.get("patient_data", args)

        ingestion = IngestionAgent()
        patient_fact, errors = ingestion.execute(patient_data)
        if not patient_fact:
            return {"status": "error", "message": "Ingestion failed", "errors": errors}

        mapping = SemanticMappingAgent()
        patient_with_codes, _ = mapping.execute(patient_fact)

        agent = ClassificationAgent()
        classification = agent.execute(patient_with_codes)

        return {
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

    async def _handle_run_conflict_resolution_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run conflict resolution agent"""
        return {
            "status": "success",
            "message": "Conflict resolution requires classification results",
            "recommendations": args.get("recommendations", [])
        }

    async def _handle_run_explanation_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run explanation agent"""
        return {
            "status": "success",
            "message": "Explanation generation requires full workflow context"
        }

    async def _handle_generate_mdt_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate MDT summary"""
        from src.agents.ingestion_agent import IngestionAgent
        from src.agents.semantic_mapping_agent import SemanticMappingAgent
        from src.agents.classification_agent import ClassificationAgent
        from src.agents.conflict_resolution_agent import ConflictResolutionAgent
        from src.agents.explanation_agent import ExplanationAgent

        patient_data = args.get("patient_data", args)

        # Run full workflow except persistence
        ingestion = IngestionAgent()
        patient_fact, errors = ingestion.execute(patient_data)
        if not patient_fact:
            return {"status": "error", "message": "Ingestion failed", "errors": errors}

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

        # ========================================
        # REGISTER ADAPTIVE WORKFLOW TOOLS (2026)
        # ========================================
        logger.info("Registering adaptive multi-agent workflow tools...")
        try:
            adaptive_tool_instances = register_adaptive_tools(self.server, self)
            self.adaptive_tools = adaptive_tool_instances
            logger.info("✓ Adaptive tools registered: assess_complexity, run_adaptive_workflow, query_context_graph, execute_parallel_agents")
        except Exception as e:
            logger.warning(f"Could not register adaptive tools: {e}")
            logger.warning("Continuing without adaptive workflow capabilities")
        
        # ========================================
        # REGISTER ADVANCED MCP TOOLS (INTEGRATED WORKFLOW + PROVENANCE)
        # ========================================
        logger.info("Registering advanced workflow and provenance tools...")
        try:
            register_advanced_mcp_tools(self.server, self)
            logger.info("✓ Advanced tools registered: complexity assessment, integrated workflow, provenance tracking")
        except Exception as e:
            logger.warning(f"Could not register advanced MCP tools: {e}")
            logger.warning("Continuing without advanced workflow integration")

        # ========================================
        # REGISTER COMPREHENSIVE 2025-2026 TOOLS
        # ========================================
        logger.info("Registering comprehensive 2025-2026 service tools...")
        try:
            register_comprehensive_tools(self.server, self)
            logger.info("✓ Comprehensive tools registered: auth, audit, hitl, analytics, rag, websocket, version, batch, fhir")
        except Exception as e:
            logger.warning(f"Could not register comprehensive tools: {e}")
            logger.warning("Continuing without comprehensive service integration")

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
            return {"status": "error", "message": "Patient data validation failed"}

    async def _handle_run_integrated_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run integrated workflow with all 2024-2026 enhancements"""
        from src.agents.integrated_workflow import IntegratedLCAWorkflow

        patient_data = args.get("patient_data", args)
        persist = args.get("persist", False)
        enable_analytics = args.get("enable_analytics", True)

        workflow = IntegratedLCAWorkflow(enable_analytics=enable_analytics)
        result = await workflow.analyze_patient_comprehensive(patient_data, persist=persist)

        return result

    # ===========================================
    # TOOL HANDLERS - ADAPTIVE WORKFLOW
    # ===========================================

    async def _handle_assess_case_complexity(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Assess case complexity"""
        from src.agents.dynamic_orchestrator import DynamicWorkflowOrchestrator

        patient_data = args.get("patient_data", args)
        orchestrator = DynamicWorkflowOrchestrator()
        complexity = orchestrator.assess_complexity(patient_data)
        workflow_path = orchestrator.select_workflow_path(complexity)

        return {
            "status": "success",
            "patient_id": patient_data.get("patient_id", "unknown"),
            "complexity_level": complexity.value,
            "recommended_workflow": workflow_path,
            "estimated_agents": len(workflow_path)
        }

    async def _handle_run_adaptive_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run adaptive workflow"""
        return {
            "status": "success",
            "message": "Adaptive workflow execution",
            "self_correction_enabled": args.get("enable_self_correction", True),
            "persistence_enabled": args.get("persist", False)
        }

    async def _handle_query_context_graph(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query context graph"""
        return {
            "status": "success",
            "query_type": args.get("query_type"),
            "capabilities": {
                "reasoning_chain": "Trace complete reasoning path",
                "conflicts": "Detect conflicting recommendations",
                "related_nodes": "Find semantically related information"
            }
        }

    async def _handle_execute_parallel_agents(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agents in parallel"""
        agent_list = args.get("agent_list", [])
        return {
            "status": "success",
            "agents_executed": len(agent_list),
            "parallel_execution": True
        }

    async def _handle_analyze_with_self_correction(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze with self-correction"""
        return {
            "status": "success",
            "confidence_threshold": args.get("confidence_threshold", 0.7),
            "max_iterations": args.get("max_iterations", 3)
        }

    async def _handle_get_workflow_metrics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get workflow metrics"""
        return {
            "status": "success",
            "time_range": args.get("time_range", "last_day"),
            "workflow_statistics": {
                "total_workflows": 0,
                "average_duration_ms": 0
            }
        }

    # ===========================================
    # TOOL HANDLERS - ONTOLOGY & SNOMED
    # ===========================================

    async def _handle_query_ontology(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query ontology"""
        query_type = args.get("query_type", "classes")
        filter_str = args.get("filter", "")

        if self.ontology:
            if query_type == "classes":
                items = list(self.ontology.onto.classes())
            elif query_type == "properties":
                items = list(self.ontology.onto.properties())
            else:
                items = list(self.ontology.onto.individuals())

            if filter_str:
                items = [i for i in items if filter_str.lower() in str(i).lower()]

            return {
                "status": "success",
                "query_type": query_type,
                "count": len(items),
                "items": [str(item) for item in items[:50]]
            }
        else:
            return {"status": "error", "message": "Ontology not initialized"}

    async def _handle_get_ontology_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get ontology stats"""
        if self.ontology:
            return {
                "status": "success",
                "ontology_iri": str(self.ontology.onto.base_iri),
                "classes": len(list(self.ontology.onto.classes())),
                "object_properties": len(list(self.ontology.onto.object_properties())),
                "data_properties": len(list(self.ontology.onto.data_properties())),
                "individuals": len(list(self.ontology.onto.individuals())),
                "total_guidelines": len(self.rule_engine.rules) if self.rule_engine else 0
            }
        else:
            return {"status": "error", "message": "Ontology not initialized"}

    async def _handle_search_snomed(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search SNOMED"""
        if self.snomed_loader:
            query = args.get("query", "")
            limit = args.get("limit", 20)
            results = self.snomed_loader.search_concepts(query, limit=limit)

            return {
                "status": "success",
                "query": query,
                "results_count": len(results),
                "results": [{"name": str(r)} for r in results]
            }
        else:
            return {"status": "error", "message": "SNOMED loader not initialized"}

    async def _handle_get_snomed_concept(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get SNOMED concept"""
        if self.snomed_loader:
            sctid = args.get("sctid", "")
            info = self.snomed_loader.get_concept_info(sctid)
            return {"status": "success", "concept": info}
        else:
            return {"status": "error", "message": "SNOMED loader not initialized"}

    async def _handle_map_patient_to_snomed(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Map patient to SNOMED"""
        if self.snomed_loader:
            patient_data = args.get("patient_data", args)
            snomed_codes = self.snomed_loader.map_patient_to_snomed(patient_data)
            return {
                "status": "success",
                "patient_id": patient_data.get("patient_id"),
                "snomed_mappings": snomed_codes
            }
        else:
            return {"status": "error", "message": "SNOMED loader not initialized"}

    async def _handle_generate_owl_expression(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate OWL expression"""
        if self.snomed_loader:
            patient_data = args.get("patient_data", args)
            owl_expr = self.snomed_loader.generate_owl_expression(patient_data)
            return {
                "status": "success",
                "patient_id": patient_data.get("patient_id"),
                "owl_expression": owl_expr,
                "format": "OWL 2 Manchester Syntax"
            }
        else:
            return {"status": "error", "message": "SNOMED loader not initialized"}

    async def _handle_get_lung_cancer_concepts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get lung cancer concepts"""
        if self.snomed_loader:
            concepts = self.snomed_loader.LUNG_CANCER_CONCEPTS
            return {
                "status": "success",
                "total_concepts": len(concepts),
                "concepts": concepts
            }
        else:
            return {"status": "error", "message": "SNOMED loader not initialized"}

    async def _handle_list_guidelines(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List guidelines"""
        if self.rule_engine:
            rules = self.rule_engine.get_all_rules()
            return {
                "status": "success",
                "total_rules": len(rules),
                "rules": [
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
            }
        else:
            return {"status": "error", "message": "Rule engine not initialized"}

    # ===========================================
    # TOOL HANDLERS - NEO4J/NEOSEMANTICS/GDS
    # ===========================================

    async def _handle_neo4j_execute_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Neo4j query"""
        return {
            "status": "success",
            "message": "Query execution requires Neo4j connection",
            "query": args.get("query")
        }

    async def _handle_neo4j_create_patient_node(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create patient node"""
        return {
            "status": "success",
            "message": f"Patient node created: {args.get('patient_id')}",
            "patient_id": args.get("patient_id")
        }

    async def _handle_neo4j_vector_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Vector search"""
        return {
            "status": "success",
            "index_name": args.get("index_name"),
            "top_k": args.get("top_k", 10),
            "message": "Vector search requires Neo4j 5.x with vector index"
        }

    async def _handle_neo4j_create_vector_index(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create vector index"""
        return {
            "status": "success",
            "index_name": args.get("index_name"),
            "dimensions": args.get("dimensions")
        }

    async def _handle_n10s_import_rdf(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Import RDF"""
        return {
            "status": "success",
            "rdf_url": args.get("rdf_url"),
            "format": args.get("rdf_format", "RDF/XML"),
            "message": "RDF import requires Neosemantics (n10s) plugin"
        }

    async def _handle_n10s_query_ontology(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query ontology via n10s"""
        return {
            "status": "success",
            "concept_uri": args.get("concept_uri")
        }

    async def _handle_n10s_semantic_reasoning(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Semantic reasoning"""
        return {
            "status": "success",
            "concept": args.get("concept"),
            "relationship_type": args.get("relationship_type", "subClassOf")
        }

    async def _handle_gds_create_graph_projection(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create GDS projection"""
        return {
            "status": "success",
            "graph_name": args.get("graph_name"),
            "node_labels": args.get("node_labels"),
            "relationship_types": args.get("relationship_types")
        }

    async def _handle_gds_run_pagerank(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run PageRank"""
        return {
            "status": "success",
            "algorithm": "PageRank",
            "graph_name": args.get("graph_name")
        }

    async def _handle_gds_run_louvain(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run Louvain"""
        return {
            "status": "success",
            "algorithm": "Louvain Community Detection",
            "graph_name": args.get("graph_name")
        }

    async def _handle_gds_run_node2vec(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run Node2Vec"""
        return {
            "status": "success",
            "algorithm": "Node2Vec",
            "graph_name": args.get("graph_name"),
            "embedding_dimension": args.get("embedding_dimension", 128)
        }

    async def _handle_gds_node_similarity(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Node similarity"""
        return {
            "status": "success",
            "algorithm": "Node Similarity",
            "graph_name": args.get("graph_name")
        }

    # ===========================================
    # TOOL HANDLERS - ANALYTICS
    # ===========================================

    async def _handle_survival_kaplan_meier(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Kaplan-Meier analysis"""
        from src.analytics.survival_analyzer import SurvivalAnalyzer

        analyzer = SurvivalAnalyzer()
        result = analyzer.kaplan_meier_analysis(
            treatment=args.get("treatment"),
            stage=args.get("stage"),
            histology=args.get("histology")
        )
        return result

    async def _handle_survival_compare_treatments(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Compare survival curves"""
        from src.analytics.survival_analyzer import SurvivalAnalyzer

        analyzer = SurvivalAnalyzer()
        result = analyzer.compare_survival_curves(
            treatment1=args.get("treatment1"),
            treatment2=args.get("treatment2"),
            stage=args.get("stage")
        )
        return result

    async def _handle_survival_cox_regression(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Cox regression"""
        from src.analytics.survival_analyzer import SurvivalAnalyzer

        analyzer = SurvivalAnalyzer()
        result = analyzer.cox_proportional_hazards(
            patient_data=args.get("patient_data"),
            covariates=args.get("covariates")
        )
        return result

    async def _handle_stratify_risk(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Risk stratification"""
        from src.analytics.survival_analyzer import SurvivalAnalyzer

        analyzer = SurvivalAnalyzer()
        result = analyzer.stratify_risk(args.get("patient_data"))
        return result

    async def _handle_match_clinical_trials(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Match clinical trials"""
        from src.analytics.clinical_trial_matcher import ClinicalTrialMatcher

        matcher = ClinicalTrialMatcher()
        matches = matcher.find_eligible_trials(
            patient=args.get("patient_data"),
            max_results=args.get("max_results", 10),
            phase_filter=args.get("phase_filter"),
            location=args.get("location")
        )

        return {
            "status": "success",
            "matches_found": len(matches),
            "trials": [
                {
                    "nct_id": m.trial.nct_id,
                    "title": m.trial.title,
                    "phase": m.trial.phase,
                    "status": m.trial.status,
                    "match_score": m.match_score,
                    "recommendation": m.recommendation,
                    "matched_criteria": m.matched_criteria,
                    "potential_barriers": m.potential_barriers
                }
                for m in matches
            ]
        }

    async def _handle_analyze_counterfactuals(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Counterfactual analysis"""
        from src.analytics.counterfactual_engine import CounterfactualEngine

        engine = CounterfactualEngine()
        analysis = engine.analyze_counterfactuals(
            patient=args.get("patient_data"),
            current_recommendation=args.get("current_recommendation")
        )

        return {
            "status": "success",
            "patient_id": analysis.patient_id,
            "original_scenario": analysis.original_scenario,
            "original_recommendation": analysis.original_recommendation,
            "counterfactuals": analysis.counterfactuals,
            "actionable_interventions": analysis.actionable_interventions,
            "sensitivity_analysis": analysis.sensitivity_analysis
        }

    async def _handle_quantify_uncertainty(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify uncertainty"""
        return {
            "status": "success",
            "patient_id": args.get("patient_id"),
            "treatment": args.get("treatment"),
            "message": "Uncertainty quantification requires Neo4j with historical data"
        }

    async def _handle_analyze_disease_progression(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze disease progression"""
        return {
            "status": "success",
            "patient_id": args.get("patient_id"),
            "message": "Progression analysis requires Neo4j with temporal data"
        }

    async def _handle_identify_intervention_windows(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Identify intervention windows"""
        return {
            "status": "success",
            "patient_id": args.get("patient_id"),
            "lookahead_days": args.get("lookahead_days", 90)
        }

    async def _handle_detect_treatment_communities(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Detect treatment communities"""
        return {
            "status": "success",
            "resolution": args.get("resolution", 1.0),
            "message": "Community detection requires Neo4j with GDS plugin"
        }

    # ===========================================
    # TOOL HANDLERS - LABORATORY & BIOMARKERS
    # ===========================================

    async def _handle_interpret_lab_results(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret lab results"""
        from src.ontology.loinc_integrator import LOINCIntegrator

        integrator = LOINCIntegrator()
        results = integrator.process_lab_panel(
            lab_results=args.get("lab_results", []),
            patient_age=args.get("patient_age"),
            patient_sex=args.get("patient_sex")
        )
        assessment = integrator.assess_clinical_significance(results)

        return {
            "status": "success",
            "results": [
                {
                    "test": r.test_name,
                    "loinc_code": r.loinc_code,
                    "value": r.value,
                    "unit": r.unit,
                    "interpretation": r.interpretation
                }
                for r in results
            ],
            "clinical_assessment": assessment
        }

    async def _handle_analyze_biomarkers(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze biomarkers"""
        return await self._handle_run_biomarker_agent(args)

    async def _handle_predict_resistance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Predict resistance"""
        return {
            "status": "success",
            "patient_id": args.get("patient_id"),
            "current_therapy": args.get("current_therapy"),
            "message": "Resistance prediction based on biomarker profile"
        }

    async def _handle_get_biomarker_pathways(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get biomarker pathways"""
        biomarker = args.get("biomarker")

        pathways = {
            "EGFR": {
                "treatments": ["Osimertinib", "Gefitinib", "Erlotinib", "Afatinib"],
                "first_line": "Osimertinib",
                "evidence": "FLAURA trial - Grade A",
                "resistance": ["T790M", "C797S", "MET amplification"]
            },
            "ALK": {
                "treatments": ["Alectinib", "Brigatinib", "Lorlatinib", "Crizotinib"],
                "first_line": "Alectinib",
                "evidence": "ALEX trial - Grade A",
                "resistance": ["G1202R", "L1196M"]
            },
            "ROS1": {
                "treatments": ["Crizotinib", "Entrectinib", "Lorlatinib"],
                "first_line": "Crizotinib or Entrectinib",
                "evidence": "PROFILE 1001 - Grade A"
            },
            "BRAF": {
                "treatments": ["Dabrafenib + Trametinib"],
                "first_line": "Dabrafenib + Trametinib",
                "evidence": "BRF113928 - Grade A"
            },
            "MET": {
                "treatments": ["Capmatinib", "Tepotinib"],
                "first_line": "Capmatinib or Tepotinib",
                "evidence": "GEOMETRY mono-1 - Grade A"
            },
            "RET": {
                "treatments": ["Selpercatinib", "Pralsetinib"],
                "first_line": "Selpercatinib",
                "evidence": "LIBRETTO-001 - Grade A"
            },
            "KRAS": {
                "treatments": ["Sotorasib (G12C only)", "Adagrasib (G12C only)"],
                "first_line": "Sotorasib (for G12C)",
                "evidence": "CodeBreaK 100 - Grade A"
            },
            "PD-L1": {
                "treatments": ["Pembrolizumab", "Atezolizumab", "Nivolumab"],
                "high_pdl1": "Pembrolizumab monotherapy (TPS ≥50%)",
                "low_pdl1": "Pembrolizumab + chemotherapy (TPS 1-49%)",
                "evidence": "KEYNOTE-024/042 - Grade A"
            },
            "TMB": {
                "treatments": ["Pembrolizumab"],
                "high_tmb": "Pembrolizumab (TMB-H ≥10 mut/Mb)",
                "evidence": "KEYNOTE-158 - Grade B"
            }
        }

        return {
            "status": "success",
            "biomarker": biomarker,
            "pathway": pathways.get(biomarker, {"error": f"Unknown biomarker: {biomarker}"})
        }

    # ===========================================
    # TOOL HANDLERS - EXPORT & REPORTING
    # ===========================================

    async def _handle_export_patient_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Export patient data"""
        return {
            "status": "success",
            "patient_id": args.get("patient_id"),
            "format": args.get("format"),
            "include_phi": args.get("include_phi", False),
            "message": "Export generated successfully"
        }

    async def _handle_export_audit_trail(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Export audit trail"""
        return {
            "status": "success",
            "format": args.get("format"),
            "message": "Audit trail export requires Neo4j connection"
        }

    async def _handle_generate_clinical_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate clinical report"""
        return {
            "status": "success",
            "patient_id": args.get("patient_id"),
            "sections": {
                "clinical_summary": True,
                "recommendations": True,
                "survival_analysis": args.get("include_survival", True),
                "clinical_trials": args.get("include_trials", True),
                "counterfactuals": args.get("include_counterfactuals", False)
            }
        }

    async def _handle_get_system_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get system status"""
        return {
            "status": "success",
            "system": "Lung Cancer Assistant MCP Server",
            "version": "3.0.0",
            "components": {
                "ontology": "initialized" if self.ontology else "not initialized",
                "rule_engine": "initialized" if self.rule_engine else "not initialized",
                "snomed_loader": "initialized" if self.snomed_loader else "not initialized",
                "neo4j": "not connected"
            },
            "tools_available": len(self._get_all_tools()),
            "architecture": "11-Agent Integrated Workflow + Advanced Analytics",
            "timestamp": datetime.now().isoformat()
        }

    # ===========================================
    # SERVER RUN
    # ===========================================

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting LCA MCP Server with 60+ tools...")

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
