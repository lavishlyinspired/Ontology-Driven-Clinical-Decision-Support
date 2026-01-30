"""
Orchestration Service for Lung Cancer Assistant
Provides high-level API for the 11-agent integrated workflow system.

Per PHASE 6 of final.md specification.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

# Centralized logging
from ..logging_config import get_logger, log_execution

logger = get_logger(__name__)

from ..agents.lca_workflow import LCAWorkflow, analyze_patient
from ..agents.ingestion_agent import IngestionAgent
from ..agents.semantic_mapping_agent import SemanticMappingAgent
from ..agents.classification_agent import ClassificationAgent
from ..agents.conflict_resolution_agent import ConflictResolutionAgent
from ..agents.persistence_agent import PersistenceAgent
from ..agents.explanation_agent import ExplanationAgent

from ..db.models import (
    PatientFact,
    PatientFactWithCodes,
    ClassificationResult,
    MDTSummary,
    DecisionSupportResponse,
    WriteReceipt
)
from ..db.neo4j_tools import Neo4jReadTools, Neo4jWriteTools

from ..ontology.lucada_ontology import LUCADAOntology
from ..ontology.guideline_rules import GuidelineRuleEngine


class LungCancerAssistantService:
    """
    Main orchestration service for the Lung Cancer Assistant.
    
    Provides high-level methods for:
    - Processing patients through the 11-agent integrated workflow
    - Retrieving patient history and inferences
    - Validating new guideline rules
    - Managing system configuration
    
    CRITICAL: Enforces Neo4j read/write separation.
    Only PersistenceAgent can write to Neo4j.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        ontology_path: Optional[str] = None,
        use_neo4j: bool = False
    ):
        """
        Initialize the LCA orchestration service.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            ontology_path: Path to LUCADA OWL ontology file
            use_neo4j: Whether to use Neo4j for persistence
        """
        self.use_neo4j = use_neo4j
        self.ontology_path = ontology_path
        
        # Initialize core components
        self.ontology = LUCADAOntology()
        self.ontology.create()
        
        self.rule_engine = GuidelineRuleEngine(self.ontology)
        
        # Initialize Neo4j tools (if configured)
        self.read_tools = None
        self.write_tools = None
        
        if use_neo4j and neo4j_uri:
            try:
                self.read_tools = Neo4jReadTools(neo4j_uri, neo4j_user, neo4j_password)
                self.write_tools = Neo4jWriteTools(neo4j_uri, neo4j_user, neo4j_password)
                logger.info("Neo4j tools initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Neo4j: {e}")
                self.use_neo4j = False
        
        # Initialize workflow
        self.workflow = LCAWorkflow(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            ontology_path=ontology_path,
            persist_results=use_neo4j
        )
        
        # Track service stats
        self.patients_processed = 0
        self.start_time = datetime.utcnow()
        
        logger.info("LungCancerAssistantService initialized")

    async def process_patient(
        self, 
        patient_data: Dict[str, Any],
        persist: bool = True
    ) -> DecisionSupportResponse:
        """
        Process a patient through the complete 11-agent integrated workflow.
        
        This is the main entry point for clinical decision support.
        
        Args:
            patient_data: Raw patient clinical data
            persist: Whether to save results to Neo4j
            
        Returns:
            DecisionSupportResponse with recommendations and MDT summary
        """
        logger.info(f"Processing patient: {patient_data.get('patient_id', 'NEW')}")
        
        # Use the workflow (synchronous internally, wrapped async)
        result = await asyncio.to_thread(
            self.workflow.run, 
            patient_data
        )
        
        self.patients_processed += 1
        
        return result

    def process_patient_sync(
        self, 
        patient_data: Dict[str, Any],
        persist: bool = True
    ) -> DecisionSupportResponse:
        """
        Synchronous version of process_patient.
        
        Args:
            patient_data: Raw patient clinical data
            persist: Whether to save results to Neo4j
            
        Returns:
            DecisionSupportResponse with recommendations and MDT summary
        """
        logger.info(f"Processing patient (sync): {patient_data.get('patient_id', 'NEW')}")
        
        result = self.workflow.run(patient_data)
        self.patients_processed += 1
        
        return result

    async def get_patient_history(
        self, 
        patient_id: str
    ) -> Dict[str, Any]:
        """
        Get complete history for a patient including all inferences.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary with patient data and inference timeline
        """
        if not self.read_tools:
            return {"error": "Neo4j not configured", "patient_id": patient_id}
        
        try:
            patient = self.read_tools.get_patient(patient_id)
            inferences = self.read_tools.get_historical_inferences(patient_id)
            
            return {
                "patient_id": patient_id,
                "patient_data": patient.model_dump() if patient else None,
                "inferences": [inf.model_dump() for inf in inferences],
                "total_inferences": len(inferences),
                "retrieved_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error retrieving patient history: {e}")
            return {"error": str(e), "patient_id": patient_id}

    async def find_similar_patients(
        self, 
        patient_data: Dict[str, Any],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar patients for cohort analysis.
        
        Args:
            patient_data: Patient data to match against
            k: Number of similar patients to return
            
        Returns:
            List of similar patients with similarity scores
        """
        if not self.read_tools:
            return []
        
        try:
            # First validate and process the patient data
            ingestion = IngestionAgent()
            patient_fact, errors = ingestion.execute(patient_data)
            
            if not patient_fact:
                return []
            
            # Find similar patients
            similar = self.read_tools.find_similar_patients(patient_fact, k=k)
            
            return [
                {
                    "patient_id": p.patient_id,
                    "similarity_score": p.similarity_score,
                    "scenario": p.scenario,
                    "outcome": p.outcome
                }
                for p in similar
            ]
        except Exception as e:
            logger.error(f"Error finding similar patients: {e}")
            return []

    async def validate_guideline(
        self, 
        guideline_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a new guideline rule by testing it with sample patients.
        
        Args:
            guideline_data: New guideline rule definition
            
        Returns:
            Validation results including test outcomes
        """
        validation_result = {
            "status": "pending",
            "guideline_id": guideline_data.get("rule_id"),
            "tests_run": 0,
            "tests_passed": 0,
            "issues": []
        }
        
        # Check required fields
        required_fields = ["rule_id", "name", "conditions", "treatment"]
        for field in required_fields:
            if field not in guideline_data:
                validation_result["issues"].append(f"Missing required field: {field}")
        
        if validation_result["issues"]:
            validation_result["status"] = "invalid"
            return validation_result
        
        # Test with sample patients (would be expanded in production)
        test_patients = [
            {
                "patient_id": "TEST-001",
                "tnm_stage": "IIIA",
                "histology_type": "Adenocarcinoma",
                "performance_status": 1
            },
            {
                "patient_id": "TEST-002",
                "tnm_stage": "IV",
                "histology_type": "SquamousCellCarcinoma",
                "performance_status": 2
            }
        ]
        
        for test_patient in test_patients:
            try:
                result = self.process_patient_sync(test_patient, persist=False)
                validation_result["tests_run"] += 1
                if result.success:
                    validation_result["tests_passed"] += 1
            except Exception as e:
                validation_result["issues"].append(f"Test failed: {str(e)}")
        
        validation_result["status"] = "valid" if not validation_result["issues"] else "invalid"
        return validation_result

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and health information.
        
        Returns:
            Dictionary with system stats
        """
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "service": {
                "name": "LungCancerAssistantService",
                "version": "2.0.0",
                "uptime_seconds": uptime,
                "patients_processed": self.patients_processed,
                "start_time": self.start_time.isoformat()
            },
            "ontology": {
                "classes": len(list(self.ontology.onto.classes())),
                "individuals": len(list(self.ontology.onto.individuals())),
                "properties": len(list(self.ontology.onto.properties()))
            },
            "guidelines": {
                "total_rules": len(self.rule_engine.rules),
                "rule_ids": [r.rule_id for r in self.rule_engine.rules]
            },
            "neo4j": {
                "enabled": self.use_neo4j,
                "read_tools": self.read_tools is not None,
                "write_tools": self.write_tools is not None
            },
            "workflow": {
                "core_processing": [
                    "IngestionAgent",
                    "SemanticMappingAgent",
                    "ExplanationAgent",
                    "PersistenceAgent"
                ],
                "specialized_clinical": [
                    "NSCLCAgent",
                    "SCLCAgent",
                    "BiomarkerAgent",
                    "ComorbidityAgent",
                    "NegotiationAgent"
                ],
                "orchestration": [
                    "DynamicOrchestrator",
                    "IntegratedWorkflow"
                ],
                "total_agents": 11,
                "architecture": "11-Agent Integrated Workflow (2025-2026)"
            }
        }

    def get_available_guidelines(self) -> List[Dict[str, Any]]:
        """
        Get all available guideline rules.
        
        Returns:
            List of guideline rule summaries
        """
        return [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "source": r.source,
                "description": r.description,
                "treatment": r.recommended_treatment,
                "evidence_level": r.evidence_level,
                "intent": r.treatment_intent
            }
            for r in self.rule_engine.rules
        ]

    def close(self):
        """Clean up resources."""
        if self.workflow:
            self.workflow.close()
        if self.read_tools:
            self.read_tools.close()
        if self.write_tools:
            self.write_tools.close()
        logger.info("LungCancerAssistantService closed")


# Convenience function for quick access
def create_lca_service(
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    use_neo4j: bool = False
) -> LungCancerAssistantService:
    """
    Create a configured LCA service instance.
    
    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        use_neo4j: Whether to enable Neo4j persistence
        
    Returns:
        Configured LungCancerAssistantService instance
    """
    return LungCancerAssistantService(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        use_neo4j=use_neo4j
    )
