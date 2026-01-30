"""
LangChain Graph Tools for Context Graph Operations
===================================================

LangChain tools that wrap context graph client operations for use in agent workflows.
Each tool returns graph_data along with the query results for visualization.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from ..db.context_graph_client import (
    get_context_graph_client,
    GraphData,
    GraphNode,
    GraphRelationship,
)

logger = get_logger(__name__)


# ============================================
# PYDANTIC MODELS FOR TOOL INPUTS
# ============================================

class SearchPatientInput(BaseModel):
    """Input for patient search."""
    query: str = Field(description="Search query (name, ID, stage, or histology)")
    limit: int = Field(default=5, description="Maximum number of results")


class PatientDecisionsInput(BaseModel):
    """Input for getting patient decisions."""
    patient_id: str = Field(description="Patient ID")
    decision_type: Optional[str] = Field(default=None, description="Filter by decision type")
    limit: int = Field(default=10, description="Maximum number of decisions")


class SimilarDecisionsInput(BaseModel):
    """Input for finding similar decisions."""
    decision_id: str = Field(description="Decision ID to find similar decisions for")
    limit: int = Field(default=5, description="Maximum number of similar decisions")


class CausalChainInput(BaseModel):
    """Input for getting causal chain."""
    decision_id: str = Field(description="Decision ID")
    direction: str = Field(default="both", description="Direction: 'causes', 'effects', or 'both'")
    depth: int = Field(default=3, description="Maximum depth to traverse")


class PatientGraphInput(BaseModel):
    """Input for getting patient-centered graph."""
    patient_id: str = Field(description="Patient ID")
    depth: int = Field(default=2, description="Graph traversal depth")


class RecordDecisionInput(BaseModel):
    """Input for recording a new decision."""
    decision_type: str = Field(description="Type of decision (e.g., 'treatment_recommendation')")
    category: str = Field(description="Category (e.g., 'NSCLC', 'immunotherapy')")
    reasoning: str = Field(description="Full reasoning for the decision")
    patient_id: Optional[str] = Field(default=None, description="Patient ID if decision is about a patient")
    treatment: Optional[str] = Field(default=None, description="Recommended treatment")
    confidence_score: float = Field(default=0.8, description="Confidence score 0-1")
    risk_factors: List[str] = Field(default=[], description="List of risk factors")
    guideline_ids: List[str] = Field(default=[], description="Applied guideline IDs")
    biomarker_ids: List[str] = Field(default=[], description="Related biomarker IDs")


class GuidelinesInput(BaseModel):
    """Input for getting guidelines."""
    category: Optional[str] = Field(default=None, description="Filter by category (e.g., 'NSCLC', 'SCLC')")


# ============================================
# TOOL IMPLEMENTATIONS
# ============================================

def search_patient_tool(query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Search for patients by name, ID, stage, or histology type.
    Returns matching patients with graph data for visualization.
    """
    try:
        client = get_context_graph_client()
        result = client.search_patients_with_graph(query, limit=limit, graph_depth=1)

        logger.info(f"[GraphTools] search_patient: Found {len(result['patients'])} patients for '{query}'")

        return {
            "patients": result["patients"],
            "graph_data": result["graph_data"].model_dump() if result["graph_data"] else None,
            "query": query,
            "count": len(result["patients"]),
        }
    except Exception as e:
        logger.error(f"[GraphTools] search_patient error: {e}")
        return {"error": str(e), "patients": [], "graph_data": None}


def get_patient_decisions_tool(
    patient_id: str,
    decision_type: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Get treatment decisions made for a patient.
    Returns decisions with graph data showing relationships.
    """
    try:
        client = get_context_graph_client()
        decisions = client.get_patient_decisions(patient_id, decision_type=decision_type, limit=limit)

        # Get graph data centered on patient
        graph_data = client.get_graph_data(
            center_node_id=patient_id,
            center_node_type="Patient",
            depth=2,
            limit=50,
        )

        logger.info(f"[GraphTools] get_patient_decisions: Found {len(decisions)} decisions for patient {patient_id}")

        return {
            "patient_id": patient_id,
            "decisions": decisions,
            "decision_count": len(decisions),
            "graph_data": graph_data.model_dump(),
        }
    except Exception as e:
        logger.error(f"[GraphTools] get_patient_decisions error: {e}")
        return {"error": str(e), "decisions": [], "graph_data": None}


def find_similar_decisions_tool(decision_id: str, limit: int = 5) -> Dict[str, Any]:
    """
    Find treatment decisions similar to the given decision.
    Uses patient characteristics and treatment patterns for matching.
    """
    try:
        client = get_context_graph_client()
        similar = client.find_similar_decisions(decision_id, limit=limit)

        # Get graph data centered on the decision
        graph_data = client.get_graph_data(
            center_node_id=decision_id,
            center_node_type="TreatmentDecision",
            depth=2,
            limit=50,
        )

        logger.info(f"[GraphTools] find_similar_decisions: Found {len(similar)} similar decisions")

        return {
            "decision_id": decision_id,
            "similar_decisions": similar,
            "count": len(similar),
            "graph_data": graph_data.model_dump(),
        }
    except Exception as e:
        logger.error(f"[GraphTools] find_similar_decisions error: {e}")
        return {"error": str(e), "similar_decisions": [], "graph_data": None}


def get_causal_chain_tool(
    decision_id: str,
    direction: str = "both",
    depth: int = 3,
) -> Dict[str, Any]:
    """
    Trace the causal chain of a treatment decision.
    Shows what caused this decision and what effects it had.
    """
    try:
        client = get_context_graph_client()
        chain = client.get_causal_chain(decision_id, direction=direction, depth=depth)

        # Get graph data including the causal relationships
        graph_data = client.get_graph_data(
            center_node_id=decision_id,
            center_node_type="TreatmentDecision",
            depth=depth + 1,
            limit=75,
        )

        logger.info(
            f"[GraphTools] get_causal_chain: {len(chain.get('causes', []))} causes, "
            f"{len(chain.get('effects', []))} effects"
        )

        return {
            "decision_id": decision_id,
            "causal_chain": chain,
            "graph_data": graph_data.model_dump(),
        }
    except Exception as e:
        logger.error(f"[GraphTools] get_causal_chain error: {e}")
        return {"error": str(e), "causal_chain": {}, "graph_data": None}


def get_patient_graph_tool(patient_id: str, depth: int = 2) -> Dict[str, Any]:
    """
    Get a patient-centered subgraph for visualization.
    Includes all related entities (decisions, biomarkers, guidelines).
    """
    try:
        client = get_context_graph_client()

        # Get patient details
        patient = client.get_patient(patient_id)

        # Get graph data
        graph_data = client.get_graph_data(
            center_node_id=patient_id,
            center_node_type="Patient",
            depth=depth,
            limit=100,
        )

        logger.info(
            f"[GraphTools] get_patient_graph: {len(graph_data.nodes)} nodes, "
            f"{len(graph_data.relationships)} relationships"
        )

        return {
            "patient_id": patient_id,
            "patient": patient,
            "graph_data": graph_data.model_dump(),
            "node_count": len(graph_data.nodes),
            "relationship_count": len(graph_data.relationships),
        }
    except Exception as e:
        logger.error(f"[GraphTools] get_patient_graph error: {e}")
        return {"error": str(e), "patient": None, "graph_data": None}


def record_decision_tool(
    decision_type: str,
    category: str,
    reasoning: str,
    patient_id: Optional[str] = None,
    treatment: Optional[str] = None,
    confidence_score: float = 0.8,
    risk_factors: List[str] = None,
    guideline_ids: List[str] = None,
    biomarker_ids: List[str] = None,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record a new treatment decision with full context.
    Creates a decision trace in the graph for auditability.
    """
    try:
        client = get_context_graph_client()

        decision_id = client.record_decision(
            decision_type=decision_type,
            category=category,
            reasoning=reasoning,
            patient_id=patient_id,
            treatment=treatment,
            confidence_score=confidence_score,
            risk_factors=risk_factors or [],
            guideline_ids=guideline_ids or [],
            biomarker_ids=biomarker_ids or [],
            session_id=session_id,
            agent_name=agent_name,
        )

        # Get graph data for the new decision
        graph_data = None
        if decision_id:
            graph_data = client.get_graph_data(
                center_node_id=decision_id,
                center_node_type="TreatmentDecision",
                depth=2,
                limit=30,
            )

        logger.info(f"[GraphTools] record_decision: Created decision {decision_id}")

        return {
            "success": bool(decision_id),
            "decision_id": decision_id,
            "decision_type": decision_type,
            "treatment": treatment,
            "graph_data": graph_data.model_dump() if graph_data else None,
        }
    except Exception as e:
        logger.error(f"[GraphTools] record_decision error: {e}")
        return {"error": str(e), "success": False, "decision_id": None}


def get_treatment_guidelines_tool(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Get clinical treatment guidelines.
    Optionally filtered by cancer type/category.
    """
    try:
        client = get_context_graph_client()
        guidelines = client.get_guidelines(category=category)

        logger.info(f"[GraphTools] get_treatment_guidelines: Found {len(guidelines)} guidelines")

        return {
            "guidelines": guidelines,
            "count": len(guidelines),
            "category_filter": category,
        }
    except Exception as e:
        logger.error(f"[GraphTools] get_treatment_guidelines error: {e}")
        return {"error": str(e), "guidelines": []}


def get_decision_details_tool(decision_id: str) -> Dict[str, Any]:
    """
    Get full details of a treatment decision including all related entities.
    """
    try:
        client = get_context_graph_client()
        result = client.get_decision_with_graph(decision_id)

        logger.info(f"[GraphTools] get_decision_details: Retrieved decision {decision_id}")

        return {
            "decision": result["decision"],
            "graph_data": result["graph_data"].model_dump() if result["graph_data"] else None,
        }
    except Exception as e:
        logger.error(f"[GraphTools] get_decision_details error: {e}")
        return {"error": str(e), "decision": None, "graph_data": None}


# ============================================
# LANGCHAIN TOOLS FACTORY
# ============================================

def create_graph_tools() -> List[BaseTool]:
    """
    Create LangChain tools for context graph operations.

    Returns a list of tools that can be used with LangChain agents.
    Each tool returns graph_data for visualization alongside results.
    """
    tools = [
        StructuredTool.from_function(
            func=search_patient_tool,
            name="search_patient",
            description=(
                "Search for patients by name, ID, TNM stage, or histology type. "
                "Returns matching patients with a graph visualization of their relationships. "
                "Use this to find patients before analyzing their treatment history."
            ),
            args_schema=SearchPatientInput,
        ),
        StructuredTool.from_function(
            func=get_patient_decisions_tool,
            name="get_patient_decisions",
            description=(
                "Get all treatment decisions made for a specific patient. "
                "Returns decisions with their reasoning, confidence scores, and related biomarkers. "
                "Includes a graph showing the patient's treatment history."
            ),
            args_schema=PatientDecisionsInput,
        ),
        StructuredTool.from_function(
            func=find_similar_decisions_tool,
            name="find_similar_decisions",
            description=(
                "Find treatment decisions similar to a given decision. "
                "Matches based on patient characteristics (stage, histology) and treatment patterns. "
                "Useful for finding precedents and validating recommendations."
            ),
            args_schema=SimilarDecisionsInput,
        ),
        StructuredTool.from_function(
            func=get_causal_chain_tool,
            name="get_causal_chain",
            description=(
                "Trace the causal chain of a treatment decision. "
                "Shows what led to this decision (causes) and what resulted from it (effects). "
                "Essential for understanding the reasoning behind clinical decisions."
            ),
            args_schema=CausalChainInput,
        ),
        StructuredTool.from_function(
            func=get_patient_graph_tool,
            name="get_patient_graph",
            description=(
                "Get a visual graph centered on a patient. "
                "Shows all related entities: treatment decisions, biomarkers, comorbidities, and guidelines. "
                "Use this for a comprehensive view of a patient's clinical context."
            ),
            args_schema=PatientGraphInput,
        ),
        StructuredTool.from_function(
            func=record_decision_tool,
            name="record_decision",
            description=(
                "Record a new treatment decision in the context graph. "
                "Creates a full decision trace with reasoning, confidence, and links to guidelines. "
                "Use this to document AI-assisted treatment recommendations for audit trails."
            ),
            args_schema=RecordDecisionInput,
        ),
        StructuredTool.from_function(
            func=get_treatment_guidelines_tool,
            name="get_treatment_guidelines",
            description=(
                "Get clinical treatment guidelines, optionally filtered by cancer type. "
                "Returns NICE guidelines and evidence-based protocols. "
                "Use to understand what guidelines apply to a specific case."
            ),
            args_schema=GuidelinesInput,
        ),
        StructuredTool.from_function(
            func=get_decision_details_tool,
            name="get_decision_details",
            description=(
                "Get full details of a specific treatment decision. "
                "Includes the patient, applied guidelines, biomarkers, and precedents. "
                "Use this to understand the complete context of a decision."
            ),
            args_schema=SimilarDecisionsInput,  # Same schema - just needs decision_id
        ),
    ]

    return tools


# ============================================
# GRAPH TOOL EXECUTOR FOR CHAT
# ============================================

class GraphToolExecutor:
    """
    Executor for graph tools that tracks tool usage and returns graph data.

    Used by chat endpoints to execute graph tools and emit SSE events.
    """

    def __init__(self):
        self.tools = {tool.name: tool for tool in create_graph_tools()}
        self.tool_calls: List[Dict[str, Any]] = []

    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool and track the call."""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}

        tool = self.tools[tool_name]

        # Record tool call
        tool_call = {
            "name": tool_name,
            "input": kwargs,
            "output": None,
        }
        self.tool_calls.append(tool_call)

        # Execute tool
        try:
            result = tool.invoke(kwargs)
            tool_call["output"] = result
            return result
        except Exception as e:
            error_result = {"error": str(e)}
            tool_call["output"] = error_result
            return error_result

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Get all tool calls made by this executor."""
        return self.tool_calls

    def clear_history(self):
        """Clear tool call history."""
        self.tool_calls = []
