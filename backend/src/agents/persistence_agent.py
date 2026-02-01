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
import hashlib
import json

# Centralized logging
from ..logging_config import get_logger, log_agent_action

logger = get_logger(__name__)

from ..db.models import (
    PatientFactWithCodes,
    ClassificationResult,
    InferenceRecord,
    WriteReceipt,
    InferenceStatus
)
from ..db.neo4j_tools import Neo4jWriteTools

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
        patient,
        classification=None,
        agent_chain: List[str] = None,
        llm_model: str = "gpt-4"
    ) -> WriteReceipt:
        """
        Execute persistence: save all results to Neo4j with full audit trail.

        Args:
            patient: Patient data with SNOMED codes (PatientFactWithCodes or dict)
            classification: Classification results with recommendations (optional)
            agent_chain: List of agents that processed this data (optional)
            llm_model: LLM model used for inference

        Returns:
            WriteReceipt confirming all writes
        """
        # Handle dict input from dynamic orchestrator
        if isinstance(patient, dict):
            patient_id = patient.get('patient_id', 'unknown')
            # Extract nested data if available
            if classification is None:
                classification = patient.get('classification')
            if agent_chain is None:
                agent_chain = patient.get('agent_chain', [])
            # Extract new data types
            lab_results = patient.get('lab_results', [])
            medications = patient.get('medications', [])
            monitoring_protocol = patient.get('monitoring_protocol')
            eligible_trials = patient.get('eligible_trials', [])
        else:
            patient_id = patient.patient_id
            lab_results = []
            medications = []
            monitoring_protocol = None
            eligible_trials = []

        # Set defaults
        if agent_chain is None:
            agent_chain = []

        logger.info(f"[{self.name}] Persisting results for patient {patient_id}...")

        timestamp = datetime.utcnow().isoformat()
        entities_written = []
        relationships_written = []

        try:
            # Step 0: Generate patient embedding for semantic search
            embedding = None
            if not isinstance(patient, dict) and classification:
                embedding = self._generate_patient_embedding(patient, classification)

            # Step 1: Save patient facts (with optional embedding)
            patient_node_id = self.write_tools.save_patient_facts(patient, embedding=embedding)
            entities_written.append(f"Patient:{patient_id}")
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
                    patient_id=patient_id,
                    inference_id=inference_id,
                    recommendation=rec
                )
                entities_written.append(f"Recommendation:{rec_id}")
                relationships_written.append(f"Inference->Recommendation:{rec_id}")

            logger.info(f"[{self.name}] ✓ Saved {len(classification.recommendations)} recommendations")

            # Step 4: Save lab results if present
            if lab_results:
                lab_ids = self._store_lab_results(patient_id, lab_results, inference_id)
                for lab_id in lab_ids:
                    entities_written.append(f"LabResult:{lab_id}")
                    relationships_written.append(f"Patient->LabResult:{lab_id}")

            # Step 5: Save medications if present
            if medications:
                med_ids = self._store_medications(patient_id, medications, inference_id)
                for med_id in med_ids:
                    entities_written.append(f"Medication:{med_id}")
                    relationships_written.append(f"Patient->Medication:{med_id}")

            # Step 6: Save monitoring protocol if present
            if monitoring_protocol:
                protocol_id = self._store_monitoring_protocol(patient_id, monitoring_protocol, inference_id)
                if protocol_id:
                    entities_written.append(f"MonitoringProtocol:{protocol_id}")
                    relationships_written.append(f"Patient->MonitoringProtocol:{protocol_id}")

            # Step 7: Save eligible clinical trials if present
            if eligible_trials:
                trial_ids = self._store_clinical_trials(patient_id, eligible_trials, inference_id)
                for trial_id in trial_ids:
                    entities_written.append(f"ClinicalTrial:{trial_id}")
                    relationships_written.append(f"Patient->ClinicalTrial:{trial_id}")

            # Step 8: Mark previous inferences as superseded
            obsolete_count = self.write_tools.mark_inference_obsolete(
                patient_id=patient_id,
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

    def _store_lab_results(
        self,
        patient_id: str,
        lab_results: List[Dict[str, Any]],
        inference_id: Optional[str] = None
    ) -> List[str]:
        """
        Store lab results to Neo4j.

        Args:
            patient_id: Patient identifier
            lab_results: List of lab result dictionaries
            inference_id: Optional inference ID to link to

        Returns:
            List of created lab result node IDs
        """
        created_ids = []
        try:
            for lab in lab_results:
                lab_id = self.write_tools.save_lab_result(
                    patient_id=patient_id,
                    loinc_code=lab.get('loinc_code'),
                    test_name=lab.get('test_name', lab.get('loinc_name', 'Unknown')),
                    value=lab.get('value', 0.0),
                    unit=lab.get('unit', lab.get('units', '')),
                    reference_range=lab.get('reference_range'),
                    interpretation=lab.get('interpretation'),
                    severity=lab.get('severity'),
                    test_date=lab.get('test_date'),
                    inference_id=inference_id
                )
                if lab_id:
                    created_ids.append(lab_id)
            if created_ids:
                logger.info(f"[{self.name}] ✓ Stored {len(created_ids)} lab results")
        except Exception as e:
            logger.error(f"[{self.name}] ✗ Failed to store lab results: {e}")

        return created_ids

    def _store_medications(
        self,
        patient_id: str,
        medications: List[Dict[str, Any]],
        inference_id: Optional[str] = None
    ) -> List[str]:
        """
        Store medications to Neo4j.

        Args:
            patient_id: Patient identifier
            medications: List of medication dictionaries
            inference_id: Optional inference ID to link to

        Returns:
            List of created medication node IDs
        """
        created_ids = []
        try:
            for med in medications:
                med_id = self.write_tools.save_medication(
                    patient_id=patient_id,
                    rxcui=med.get('rxcui'),
                    drug_name=med.get('drug_name', med.get('name', 'Unknown')),
                    dose=med.get('dose'),
                    frequency=med.get('frequency'),
                    route=med.get('route'),
                    start_date=med.get('start_date'),
                    end_date=med.get('end_date'),
                    status=med.get('status', 'active'),
                    inference_id=inference_id
                )
                if med_id:
                    created_ids.append(med_id)
            if created_ids:
                logger.info(f"[{self.name}] ✓ Stored {len(created_ids)} medications")
        except Exception as e:
            logger.error(f"[{self.name}] ✗ Failed to store medications: {e}")

        return created_ids

    def _store_monitoring_protocol(
        self,
        patient_id: str,
        monitoring_protocol: Dict[str, Any],
        inference_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Store monitoring protocol to Neo4j.

        Args:
            patient_id: Patient identifier
            monitoring_protocol: Monitoring protocol dictionary
            inference_id: Optional inference ID to link to

        Returns:
            Created protocol node ID or None
        """
        try:
            protocol_id = self.write_tools.save_monitoring_protocol(
                patient_id=patient_id,
                regimen=monitoring_protocol.get('regimen', 'Unknown'),
                frequency=monitoring_protocol.get('frequency'),
                tests_to_monitor=monitoring_protocol.get('tests_to_monitor', []),
                schedule=monitoring_protocol.get('schedule', []),
                dose_adjustments=monitoring_protocol.get('dose_adjustments', []),
                created_at=monitoring_protocol.get('created_at'),
                inference_id=inference_id
            )
            if protocol_id:
                logger.info(f"[{self.name}] ✓ Stored monitoring protocol")
            return protocol_id
        except Exception as e:
            logger.error(f"[{self.name}] ✗ Failed to store monitoring protocol: {e}")
            return None

    def _store_clinical_trials(
        self,
        patient_id: str,
        clinical_trials: List[Dict[str, Any]],
        inference_id: Optional[str] = None
    ) -> List[str]:
        """
        Store clinical trial matches to Neo4j.

        Args:
            patient_id: Patient identifier
            clinical_trials: List of clinical trial match dictionaries
            inference_id: Optional inference ID to link to

        Returns:
            List of created trial match node IDs
        """
        created_ids = []
        try:
            for trial in clinical_trials:
                trial_id = self.write_tools.save_clinical_trial_match(
                    patient_id=patient_id,
                    nct_id=trial.get('nct_id'),
                    title=trial.get('title', trial.get('brief_title', 'Unknown')),
                    phase=trial.get('phase'),
                    status=trial.get('status'),
                    match_score=trial.get('match_score', trial.get('eligibility_score', 0.0)),
                    eligibility_criteria=trial.get('eligibility_criteria', []),
                    matched_criteria=trial.get('matched_criteria', []),
                    inference_id=inference_id
                )
                if trial_id:
                    created_ids.append(trial_id)
            if created_ids:
                logger.info(f"[{self.name}] ✓ Stored {len(created_ids)} clinical trial matches")
        except Exception as e:
            logger.error(f"[{self.name}] ✗ Failed to store clinical trials: {e}")

        return created_ids

    def get_write_stats(self) -> Dict[str, Any]:
        """Get statistics about writes performed."""
        return {
            "agent": self.name,
            "version": self.version,
            "total_writes": self.write_count,
            "timestamp": datetime.utcnow().isoformat()
        }
