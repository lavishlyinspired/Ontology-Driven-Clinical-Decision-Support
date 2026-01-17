"""
Persistence Agent (Agent 5 of 6)
THE ONLY AGENT THAT WRITES TO NEO4J.

Responsibilities:
- Save patient facts to Neo4j
- Save inference results with full audit trail
- Generate and save patient embeddings for semantic search
- Maintain provenance and versioning
- Return write receipts for confirmation

Tools: Neo4jWriteTools ONLY (save_patient_facts, save_inference_result, etc.)
Data Sources: Classification results from upstream agents
EXCLUSIVE: Only agent with Neo4j write access
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import hashlib
import json

from ..db.models import (
    PatientFactWithCodes,
    ClassificationResult,
    InferenceRecord,
    WriteReceipt,
    InferenceStatus
)
from ..db.neo4j_tools import Neo4jWriteTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import sentence_transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence_transformers not available - patient embeddings disabled")


class PersistenceAgent:
    """
    Agent 5: Persistence Agent
    THE ONLY AGENT WITH NEO4J WRITE ACCESS.
    
    Maintains strict audit trail and provenance for all writes.
    All other agents are READ-ONLY.
    Generates patient embeddings for semantic similarity search.
    """

    def __init__(self, neo4j_write_tools: Neo4jWriteTools, enable_embeddings: bool = True):
        self.name = "PersistenceAgent"
        self.version = "1.0.0"
        self.write_tools = neo4j_write_tools
        self.write_count = 0
        
        # Initialize embedding model if available
        self.embedding_model = None
        if enable_embeddings and EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info(f"[{self.name}] ✓ Embedding model loaded (all-MiniLM-L6-v2)")
            except Exception as e:
                logger.warning(f"[{self.name}] Could not load embedding model: {e}")

    def _generate_patient_embedding(self, patient: PatientFactWithCodes, classification: ClassificationResult) -> Optional[List[float]]:
        """
        Generate embedding for patient to enable semantic similarity search.
        
        The embedding captures clinical characteristics for finding similar patients.
        """
        if not self.embedding_model:
            return None
        
        try:
            # Create descriptive text for embedding
            text = f"""
            Lung cancer patient with {patient.tnm_stage} stage {patient.histology_type}.
            Age: {patient.age_at_diagnosis}. Sex: {patient.sex}.
            WHO Performance Status: {patient.performance_status}.
            Laterality: {patient.laterality or 'Unknown'}.
            Clinical scenario: {classification.scenario}.
            Recommended treatments: {', '.join([r.get('treatment', str(r)) if isinstance(r, dict) else r.treatment for r in classification.recommendations[:3]])}.
            """
            
            embedding = self.embedding_model.encode(text.strip()).tolist()
            logger.info(f"[{self.name}] ✓ Generated {len(embedding)}-dim patient embedding")
            return embedding
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to generate embedding: {e}")
            return None

    def execute(
        self, 
        patient: PatientFactWithCodes,
        classification: ClassificationResult,
        agent_chain: List[str],
        llm_model: str = "gpt-4"
    ) -> WriteReceipt:
        """
        Execute persistence: save all results to Neo4j with full audit trail.
        
        Args:
            patient: Patient data with SNOMED codes
            classification: Classification results with recommendations
            agent_chain: List of agents that processed this data
            llm_model: LLM model used for inference
            
        Returns:
            WriteReceipt confirming all writes
        """
        logger.info(f"[{self.name}] Persisting results for patient {patient.patient_id}...")

        timestamp = datetime.utcnow().isoformat()
        entities_written = []
        relationships_written = []

        try:
            # Step 0: Generate patient embedding for semantic search
            embedding = self._generate_patient_embedding(patient, classification)
            
            # Step 1: Save patient facts (with optional embedding)
            patient_node_id = self.write_tools.save_patient_facts(patient, embedding=embedding)
            entities_written.append(f"Patient:{patient.patient_id}")
            logger.info(f"[{self.name}] ✓ Saved patient facts")

            # Step 2: Create inference record
            inference_record = self._create_inference_record(
                patient=patient,
                classification=classification,
                agent_chain=agent_chain,
                llm_model=llm_model
            )
            
            inference_id = self.write_tools.save_inference_result(inference_record)
            entities_written.append(f"Inference:{inference_id}")
            relationships_written.append(f"Patient->Inference:{inference_id}")
            logger.info(f"[{self.name}] ✓ Saved inference record")

            # Step 3: Save treatment recommendations
            for rec in classification.recommendations:
                rec_id = self.write_tools.save_treatment_recommendation(
                    patient_id=patient.patient_id,
                    inference_id=inference_id,
                    recommendation=rec
                )
                entities_written.append(f"Recommendation:{rec_id}")
                relationships_written.append(f"Inference->Recommendation:{rec_id}")
            
            logger.info(f"[{self.name}] ✓ Saved {len(classification.recommendations)} recommendations")

            # Step 4: Mark previous inferences as superseded
            obsolete_count = self.write_tools.mark_inference_obsolete(
                patient_id=patient.patient_id,
                current_inference_id=inference_id
            )
            if obsolete_count > 0:
                logger.info(f"[{self.name}] ✓ Marked {obsolete_count} previous inferences as superseded")

            self.write_count += 1

            return WriteReceipt(
                success=True,
                timestamp=timestamp,
                entities_written=entities_written,
                relationships_written=relationships_written,
                inference_id=inference_id,
                agent_version=self.version,
                write_sequence=self.write_count
            )

        except Exception as e:
            logger.error(f"[{self.name}] ✗ Persistence failed: {e}")
            return WriteReceipt(
                success=False,
                timestamp=timestamp,
                entities_written=entities_written,
                relationships_written=relationships_written,
                error_message=str(e),
                agent_version=self.version,
                write_sequence=self.write_count
            )

    def _create_inference_record(
        self,
        patient: PatientFactWithCodes,
        classification: ClassificationResult,
        agent_chain: List[str],
        llm_model: str
    ) -> InferenceRecord:
        """Create an inference record with full provenance."""
        
        # Handle recommendations that can be either dicts or objects
        def get_treatment(r):
            if isinstance(r, dict):
                return r.get('treatment', 'Unknown')
            return getattr(r, 'treatment', 'Unknown')
        
        def to_dict(r):
            if isinstance(r, dict):
                return r
            return r.model_dump() if hasattr(r, 'model_dump') else vars(r)
        
        # Generate content hash for deduplication
        content_to_hash = {
            "patient_id": patient.patient_id,
            "scenario": classification.scenario,
            "recommendations": [get_treatment(r) for r in classification.recommendations],
            "timestamp": datetime.utcnow().isoformat()
        }
        content_hash = hashlib.sha256(
            json.dumps(content_to_hash, sort_keys=True).encode()
        ).hexdigest()[:16]

        return InferenceRecord(
            inference_id=f"INF-{patient.patient_id}-{content_hash}",
            patient_id=patient.patient_id,
            scenario=classification.scenario,
            scenario_confidence=classification.scenario_confidence,
            recommendations=[to_dict(r) for r in classification.recommendations],
            reasoning_chain=classification.reasoning_chain,
            ontology_version="LUCADA-2026-01",
            guideline_refs=classification.guideline_refs,
            snomed_codes_used=[
                patient.snomed_diagnosis_code,
                patient.snomed_histology_code,
                patient.snomed_stage_code,
                patient.snomed_ps_code
            ],
            agent_chain=agent_chain,
            llm_model=llm_model,
            status=InferenceStatus.COMPLETED,
            created_at=datetime.utcnow(),
            created_by=self.name
        )

    def save_patient_only(self, patient: PatientFactWithCodes) -> WriteReceipt:
        """
        Save only patient facts without inference results.
        Useful for initial patient registration.
        """
        timestamp = datetime.utcnow().isoformat()
        
        try:
            patient_node_id = self.write_tools.save_patient_facts(patient)
            self.write_count += 1
            
            return WriteReceipt(
                success=True,
                timestamp=timestamp,
                entities_written=[f"Patient:{patient.patient_id}"],
                relationships_written=[],
                agent_version=self.version,
                write_sequence=self.write_count
            )
        except Exception as e:
            return WriteReceipt(
                success=False,
                timestamp=timestamp,
                entities_written=[],
                relationships_written=[],
                error_message=str(e),
                agent_version=self.version,
                write_sequence=self.write_count
            )

    def update_inference_status(
        self, 
        inference_id: str, 
        new_status: InferenceStatus,
        reason: str
    ) -> WriteReceipt:
        """
        Update the status of an existing inference.
        Used for MDT review outcomes.
        """
        timestamp = datetime.utcnow().isoformat()
        
        try:
            success = self.write_tools.update_inference_status(
                inference_id=inference_id,
                status=new_status,
                reason=reason
            )
            self.write_count += 1
            
            return WriteReceipt(
                success=success,
                timestamp=timestamp,
                entities_written=[f"Inference:{inference_id}"],
                relationships_written=[],
                inference_id=inference_id,
                agent_version=self.version,
                write_sequence=self.write_count
            )
        except Exception as e:
            return WriteReceipt(
                success=False,
                timestamp=timestamp,
                entities_written=[],
                relationships_written=[],
                error_message=str(e),
                agent_version=self.version,
                write_sequence=self.write_count
            )

    def get_write_stats(self) -> Dict[str, Any]:
        """Get statistics about writes performed."""
        return {
            "agent": self.name,
            "version": self.version,
            "total_writes": self.write_count,
            "timestamp": datetime.utcnow().isoformat()
        }
