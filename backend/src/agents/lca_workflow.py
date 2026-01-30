"""
LCA Workflow Orchestration
LangGraph-based workflow orchestrating all 6 agents.

Flow: Input → Ingestion → SemanticMapping → Classification → ConflictResolution → Persistence → Explanation → Output

CRITICAL PRINCIPLE: "Neo4j as a tool, not a brain"
- All medical reasoning happens in Python/OWL
- Neo4j is only for storage and retrieval
- Only PersistenceAgent writes to Neo4j
"""

from typing import Dict, Any, Optional, TypedDict, List, Annotated
from datetime import datetime
import operator

# Centralized logging
from ..logging_config import get_logger, log_execution, log_workflow_event

logger = get_logger(__name__)

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

# Agent imports
from .ingestion_agent import IngestionAgent
from .semantic_mapping_agent import SemanticMappingAgent
from .classification_agent import ClassificationAgent
from .conflict_resolution_agent import ConflictResolutionAgent
# Lazy import to avoid loading sentence_transformers/torch
# from .persistence_agent import PersistenceAgent
from .explanation_agent import ExplanationAgent

# Model imports
from ..db.models import (
    PatientFact,
    PatientFactWithCodes,
    ClassificationResult,
    MDTSummary,
    WriteReceipt,
    DecisionSupportResponse
)
from ..db.neo4j_tools import Neo4jReadTools, Neo4jWriteTools


class WorkflowState(TypedDict):
    """State passed between agents in the workflow."""
    # Input
    raw_patient_data: Dict[str, Any]
    
    # After IngestionAgent
    patient_fact: Optional[PatientFact]
    ingestion_errors: List[str]
    
    # After SemanticMappingAgent
    patient_with_codes: Optional[PatientFactWithCodes]
    mapping_confidence: float
    
    # After ClassificationAgent
    classification: Optional[ClassificationResult]
    
    # After ConflictResolutionAgent
    resolved_classification: Optional[ClassificationResult]
    conflict_reports: List[Dict[str, Any]]
    
    # After PersistenceAgent
    write_receipt: Optional[WriteReceipt]
    inference_id: Optional[str]
    
    # After ExplanationAgent
    mdt_summary: Optional[MDTSummary]
    
    # Workflow metadata
    agent_chain: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    workflow_status: str


class LCAWorkflow:
    """
    LangGraph workflow orchestrating 6 specialized agents for lung cancer clinical decision support.
    
    Agent Chain:
    1. IngestionAgent: Validates and normalizes raw patient data
    2. SemanticMappingAgent: Maps clinical concepts to SNOMED-CT codes
    3. ClassificationAgent: Applies LUCADA ontology and NICE guidelines
    4. ConflictResolutionAgent: Resolves conflicting recommendations
    5. PersistenceAgent: Saves results to Neo4j (ONLY writer)
    6. ExplanationAgent: Generates MDT summaries
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        ontology_path: Optional[str] = None,
        persist_results: bool = True
    ):
        self.persist_results = persist_results
        
        # Initialize agents
        self.ingestion_agent = IngestionAgent()
        self.semantic_mapping_agent = SemanticMappingAgent()
        self.classification_agent = ClassificationAgent(ontology_path=ontology_path)
        self.conflict_resolution_agent = ConflictResolutionAgent()
        self.explanation_agent = ExplanationAgent()
        
        # Initialize Neo4j tools
        self.read_tools = None
        self.write_tools = None
        self.persistence_agent = None
        
        # Initialize Neo4j tools if persistence is requested
        if persist_results:
            try:
                # Use provided credentials or fall back to environment variables
                self.read_tools = Neo4jReadTools(neo4j_uri, neo4j_user, neo4j_password)
                self.write_tools = Neo4jWriteTools(neo4j_uri, neo4j_user, neo4j_password)
                
                if self.write_tools.is_available:
                    # Lazy import PersistenceAgent to avoid loading sentence_transformers
                    try:
                        from .persistence_agent import PersistenceAgent
                        self.persistence_agent = PersistenceAgent(self.write_tools)
                        logger.info("✅ Neo4j tools initialized successfully for persistence")
                    except ImportError as e:
                        logger.warning(f"⚠ PersistenceAgent not available: {e}")
                else:
                    logger.warning("⚠ Neo4j not available - persistence will be skipped")
            except Exception as e:
                logger.warning(f"Failed to initialize Neo4j tools: {e}")
        
        # Build workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """Build the LangGraph workflow."""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available. Using fallback sequential execution.")
            return None

        workflow = StateGraph(WorkflowState)

        # Add nodes for each agent
        workflow.add_node("ingestion", self._run_ingestion)
        workflow.add_node("semantic_mapping", self._run_semantic_mapping)
        workflow.add_node("classification", self._run_classification)
        workflow.add_node("conflict_resolution", self._run_conflict_resolution)
        workflow.add_node("persistence", self._run_persistence)
        workflow.add_node("explanation", self._run_explanation)

        # Define edges (linear flow)
        workflow.set_entry_point("ingestion")
        workflow.add_conditional_edges(
            "ingestion",
            self._should_continue_after_ingestion,
            {
                "continue": "semantic_mapping",
                "error": END
            }
        )
        workflow.add_edge("semantic_mapping", "classification")
        workflow.add_edge("classification", "conflict_resolution")
        workflow.add_conditional_edges(
            "conflict_resolution",
            self._should_persist,
            {
                "persist": "persistence",
                "skip": "explanation"
            }
        )
        workflow.add_edge("persistence", "explanation")
        workflow.add_edge("explanation", END)

        return workflow.compile()

    def _should_continue_after_ingestion(self, state: WorkflowState) -> str:
        """Decide whether to continue after ingestion."""
        if state["patient_fact"] is None or state["ingestion_errors"]:
            return "error"
        return "continue"

    def _should_persist(self, state: WorkflowState) -> str:
        """Decide whether to persist results."""
        if self.persist_results and self.persistence_agent:
            return "persist"
        return "skip"

    def _run_ingestion(self, state: WorkflowState) -> Dict[str, Any]:
        """Run the IngestionAgent."""
        logger.info("[Workflow] Running IngestionAgent...")
        
        patient_fact, errors = self.ingestion_agent.execute(state["raw_patient_data"])
        
        return {
            "patient_fact": patient_fact,
            "ingestion_errors": errors,
            "agent_chain": ["IngestionAgent"],
            "errors": errors,
            "workflow_status": "ingestion_complete" if patient_fact else "ingestion_failed"
        }

    def _run_semantic_mapping(self, state: WorkflowState) -> Dict[str, Any]:
        """Run the SemanticMappingAgent."""
        logger.info("[Workflow] Running SemanticMappingAgent...")
        
        patient_with_codes, confidence = self.semantic_mapping_agent.execute(
            state["patient_fact"]
        )
        
        return {
            "patient_with_codes": patient_with_codes,
            "mapping_confidence": confidence,
            "agent_chain": ["SemanticMappingAgent"],
            "workflow_status": "semantic_mapping_complete"
        }

    def _run_classification(self, state: WorkflowState) -> Dict[str, Any]:
        """Run the ClassificationAgent."""
        logger.info("[Workflow] Running ClassificationAgent...")
        
        classification = self.classification_agent.execute(state["patient_with_codes"])
        
        return {
            "classification": classification,
            "agent_chain": ["ClassificationAgent"],
            "workflow_status": "classification_complete"
        }

    def _run_conflict_resolution(self, state: WorkflowState) -> Dict[str, Any]:
        """Run the ConflictResolutionAgent."""
        logger.info("[Workflow] Running ConflictResolutionAgent...")
        
        resolved_classification, conflict_reports = self.conflict_resolution_agent.execute(
            state["classification"]
        )
        
        return {
            "resolved_classification": resolved_classification,
            "conflict_reports": [
                {"type": r.conflict_type, "resolution": r.resolution} 
                for r in conflict_reports
            ],
            "agent_chain": ["ConflictResolutionAgent"],
            "workflow_status": "conflict_resolution_complete"
        }

    def _run_persistence(self, state: WorkflowState) -> Dict[str, Any]:
        """Run the PersistenceAgent."""
        logger.info("[Workflow] Running PersistenceAgent...")
        
        if not self.persistence_agent:
            return {
                "write_receipt": None,
                "inference_id": None,
                "agent_chain": ["PersistenceAgent(skipped)"],
                "workflow_status": "persistence_skipped"
            }
        
        write_receipt = self.persistence_agent.execute(
            patient=state["patient_with_codes"],
            classification=state["resolved_classification"],
            agent_chain=state["agent_chain"]
        )
        
        return {
            "write_receipt": write_receipt,
            "inference_id": write_receipt.inference_id if write_receipt else None,
            "agent_chain": ["PersistenceAgent"],
            "workflow_status": "persistence_complete"
        }

    def _run_explanation(self, state: WorkflowState) -> Dict[str, Any]:
        """Run the ExplanationAgent."""
        logger.info("[Workflow] Running ExplanationAgent...")
        
        mdt_summary = self.explanation_agent.execute(
            patient=state["patient_with_codes"],
            classification=state["resolved_classification"],
            inference_id=state.get("inference_id")
        )
        
        return {
            "mdt_summary": mdt_summary,
            "agent_chain": ["ExplanationAgent"],
            "workflow_status": "complete"
        }

    def run(self, patient_data: Dict[str, Any]) -> DecisionSupportResponse:
        """
        Run the complete workflow for a patient.
        
        Args:
            patient_data: Raw patient data dictionary
            
        Returns:
            DecisionSupportResponse with all results
        """
        start_time = datetime.utcnow()
        
        # Initialize state
        initial_state: WorkflowState = {
            "raw_patient_data": patient_data,
            "patient_fact": None,
            "ingestion_errors": [],
            "patient_with_codes": None,
            "mapping_confidence": 0.0,
            "classification": None,
            "resolved_classification": None,
            "conflict_reports": [],
            "write_receipt": None,
            "inference_id": None,
            "mdt_summary": None,
            "agent_chain": [],
            "errors": [],
            "workflow_status": "started"
        }

        if self.workflow:
            # Use LangGraph workflow
            final_state = self.workflow.invoke(initial_state)
        else:
            # Fallback to sequential execution
            final_state = self._run_sequential(initial_state)

        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        # Build response
        return self._build_response(final_state, processing_time)

    def _run_sequential(self, state: WorkflowState) -> WorkflowState:
        """Fallback sequential execution without LangGraph."""
        # Ingestion
        state = {**state, **self._run_ingestion(state)}
        if state["patient_fact"] is None:
            return state

        # Semantic mapping
        state = {**state, **self._run_semantic_mapping(state)}

        # Classification
        state = {**state, **self._run_classification(state)}

        # Conflict resolution
        state = {**state, **self._run_conflict_resolution(state)}

        # Persistence (if enabled)
        if self.persist_results and self.persistence_agent:
            state = {**state, **self._run_persistence(state)}

        # Explanation
        state = {**state, **self._run_explanation(state)}

        return state

    def _build_response(
        self, 
        state: WorkflowState, 
        processing_time: float
    ) -> DecisionSupportResponse:
        """Build the final response from workflow state."""
        
        # Get classification result
        classification = state.get("resolved_classification") or state.get("classification")
        
        # Build recommendations list
        recommendations = []
        if classification and classification.recommendations:
            recommendations = [
                {
                    "rank": r.get("rank") if isinstance(r, dict) else r.rank,
                    "treatment": r.get("treatment") if isinstance(r, dict) else r.treatment,
                    "evidence_level": r.get("evidence_level") if isinstance(r, dict) else r.evidence_level.value,
                    "intent": (r.get("intent") if isinstance(r, dict) else (r.intent.value if r.intent else None)),
                    "guideline_reference": r.get("guideline_reference") if isinstance(r, dict) else r.guideline_reference,
                    "rationale": r.get("rationale") if isinstance(r, dict) else r.rationale
                }
                for r in classification.recommendations
            ]

        return DecisionSupportResponse(
            patient_id=state["patient_with_codes"].patient_id if state.get("patient_with_codes") else None,
            success=state["workflow_status"] == "complete",
            workflow_status=state["workflow_status"],
            agent_chain=state["agent_chain"],
            
            # Classification results
            scenario=classification.scenario if classification else None,
            scenario_confidence=classification.scenario_confidence if classification else None,
            recommendations=recommendations,
            reasoning_chain=classification.reasoning_chain if classification else [],
            
            # SNOMED mappings
            snomed_mappings={
                "diagnosis": state["patient_with_codes"].snomed_diagnosis_code if state.get("patient_with_codes") else None,
                "histology": state["patient_with_codes"].snomed_histology_code if state.get("patient_with_codes") else None,
                "stage": state["patient_with_codes"].snomed_stage_code if state.get("patient_with_codes") else None,
            } if state.get("patient_with_codes") else {},
            mapping_confidence=state.get("mapping_confidence", 0.0),
            
            # Persistence results
            inference_id=state.get("inference_id"),
            persisted=state.get("write_receipt") is not None and state["write_receipt"].success if state.get("write_receipt") else False,
            
            # MDT summary
            mdt_summary=state["mdt_summary"].clinical_summary if state.get("mdt_summary") else None,
            key_considerations=state["mdt_summary"].key_considerations if state.get("mdt_summary") else [],
            discussion_points=state["mdt_summary"].discussion_points if state.get("mdt_summary") else [],
            
            # Metadata
            processing_time_seconds=processing_time,
            errors=state.get("errors", []),
            guideline_refs=classification.guideline_refs if classification else []
        )

    def close(self):
        """Clean up resources."""
        if self.read_tools:
            self.read_tools.close()
        if self.write_tools:
            self.write_tools.close()


# Convenience function for quick analysis
def analyze_patient(
    patient_data: Dict[str, Any],
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    persist: bool = False
) -> DecisionSupportResponse:
    """
    Analyze a patient using the LCA workflow.
    
    Args:
        patient_data: Raw patient data
        neo4j_uri: Optional Neo4j connection URI
        neo4j_user: Optional Neo4j username
        neo4j_password: Optional Neo4j password
        persist: Whether to save results to Neo4j
        
    Returns:
        DecisionSupportResponse with analysis results
    """
    workflow = LCAWorkflow(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        persist_results=persist
    )
    
    try:
        return workflow.run(patient_data)
    finally:
        workflow.close()
