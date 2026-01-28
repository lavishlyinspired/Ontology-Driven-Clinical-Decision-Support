"""
Neo4j Tools with Strict Read/Write Separation
Implements the exact contract from final.md PHASE 3

CRITICAL: 
- READ methods: All 6 agents can use
- WRITE methods: ONLY PersistenceAgent can use
- NO medical reasoning in Cypher queries
- Complete audit trail for all operations
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import os
import uuid

from .models import (
    PatientFact, PatientFactWithCodes, InferenceRecord, WriteReceipt,
    SimilarPatient, CohortStats, InferenceStatus
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jReadTools:
    """
    READ-ONLY Neo4j operations.
    All 6 agents can use these methods.
    NO medical reasoning in Cypher - only data retrieval.
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            self._available = True
            logger.info(f"✓ Neo4j READ connection established: {self.uri}")
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}")
            self._available = False
            self.driver = None

    @property
    def is_available(self) -> bool:
        return self._available

    def get_patient(self, patient_id: str) -> Optional[PatientFact]:
        """
        Retrieve patient by ID.
        READ-ONLY: No medical reasoning, just data retrieval.
        """
        if not self._available:
            return None

        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        OPTIONAL MATCH (p)-[:HAS_CLINICAL_FINDING]->(cf:ClinicalFinding)
        OPTIONAL MATCH (cf)-[:HAS_HISTOLOGY]->(h:Histology)
        RETURN p, cf, h
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, patient_id=patient_id)
                record = result.single()
                
                if not record:
                    return None
                
                patient_node = record["p"]
                return PatientFact(
                    patient_id=patient_node.get("patient_id"),
                    name=patient_node.get("name", "Unknown"),
                    sex=patient_node.get("sex", "U"),
                    age_at_diagnosis=patient_node.get("age_at_diagnosis", 0),
                    tnm_stage=patient_node.get("tnm_stage", "IV"),
                    histology_type=patient_node.get("histology_type", "NonSmallCellCarcinoma_NOS"),
                    performance_status=patient_node.get("performance_status", 0),
                    laterality=patient_node.get("laterality", "Right"),
                    fev1_percent=patient_node.get("fev1_percent"),
                    comorbidities=patient_node.get("comorbidities", []),
                    notes=patient_node.get("notes")
                )
        except Exception as e:
            logger.error(f"Failed to get patient {patient_id}: {e}")
            return None

    def get_historical_inferences(self, patient_id: str) -> List[InferenceRecord]:
        """
        Retrieve all inference records for a patient.
        READ-ONLY: Historical data retrieval only.
        """
        if not self._available:
            return []

        query = """
        MATCH (p:Patient {patient_id: $patient_id})-[:HAS_INFERENCE]->(i:Inference)
        RETURN i
        ORDER BY i.created_at DESC
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, patient_id=patient_id)
                records = []
                for record in result:
                    inf_node = record["i"]
                    records.append(InferenceRecord(
                        inference_id=inf_node.get("inference_id"),
                        patient_id=patient_id,
                        status=inf_node.get("status", "completed"),
                        created_at=inf_node.get("created_at", datetime.now()),
                        agent_version=inf_node.get("agent_version", "1.0.0")
                    ))
                return records
        except Exception as e:
            logger.error(f"Failed to get inferences for {patient_id}: {e}")
            return []

    def get_cohort_statistics(self, criteria: Dict[str, Any]) -> CohortStats:
        """
        Get aggregated cohort statistics.
        READ-ONLY: Statistical aggregation only, NO medical reasoning.
        """
        if not self._available:
            return CohortStats(total_patients=0)

        # Build WHERE clause from criteria (simple filtering only)
        where_clauses = []
        params = {}
        
        if criteria.get("tnm_stage"):
            where_clauses.append("p.tnm_stage = $stage")
            params["stage"] = criteria["tnm_stage"]
        if criteria.get("histology_type"):
            where_clauses.append("p.histology_type = $histology")
            params["histology"] = criteria["histology_type"]
        if criteria.get("min_age"):
            where_clauses.append("p.age_at_diagnosis >= $min_age")
            params["min_age"] = criteria["min_age"]
        if criteria.get("max_age"):
            where_clauses.append("p.age_at_diagnosis <= $max_age")
            params["max_age"] = criteria["max_age"]

        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        MATCH (p:Patient)
        {where_clause}
        RETURN 
            count(p) as total,
            collect(p.tnm_stage) as stages,
            collect(p.histology_type) as histologies,
            avg(p.survival_days) as avg_survival
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **params)
                record = result.single()
                
                if not record:
                    return CohortStats(total_patients=0)
                
                # Calculate distributions (pure aggregation, no reasoning)
                stages = record["stages"]
                histologies = record["histologies"]
                
                stage_dist = {}
                for s in stages:
                    stage_dist[s] = stage_dist.get(s, 0) + 1
                
                hist_dist = {}
                for h in histologies:
                    hist_dist[h] = hist_dist.get(h, 0) + 1
                
                return CohortStats(
                    total_patients=record["total"],
                    stage_distribution=stage_dist,
                    histology_distribution=hist_dist,
                    avg_survival_days=record["avg_survival"]
                )
        except Exception as e:
            logger.error(f"Failed to get cohort statistics: {e}")
            return CohortStats(total_patients=0)

    def find_similar_patients(self, patient_fact: PatientFact, k: int = 5) -> List[SimilarPatient]:
        """
        Find similar patients based on clinical profile.
        READ-ONLY: Similarity matching only, NO treatment reasoning.
        """
        if not self._available:
            return []

        query = """
        MATCH (p:Patient)
        WHERE p.patient_id <> $patient_id
          AND p.tnm_stage = $stage
          AND abs(p.performance_status - $ps) <= 1
        OPTIONAL MATCH (p)-[:RECEIVED_TREATMENT]->(t:TreatmentPlan)
        OPTIONAL MATCH (t)-[:HAS_OUTCOME]->(o:Outcome)
        RETURN p, t, o
        ORDER BY abs(p.age_at_diagnosis - $age)
        LIMIT $limit
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    patient_id=patient_fact.patient_id,
                    stage=patient_fact.tnm_stage,
                    ps=patient_fact.performance_status,
                    age=patient_fact.age_at_diagnosis,
                    limit=k
                )
                
                similar = []
                for record in result:
                    p = record["p"]
                    t = record["t"]
                    o = record["o"]
                    
                    # Simple similarity score (no medical reasoning)
                    score = 1.0
                    if p.get("histology_type") == patient_fact.histology_type:
                        score += 0.2
                    if abs(p.get("age_at_diagnosis", 0) - patient_fact.age_at_diagnosis) <= 5:
                        score += 0.1
                    score = min(score, 1.0)
                    
                    similar.append(SimilarPatient(
                        patient_id=p.get("patient_id"),
                        name=p.get("name", "Unknown"),
                        similarity_score=score,
                        tnm_stage=p.get("tnm_stage") or "Unknown",
                        histology_type=p.get("histology_type") or "Unknown",
                        treatment_received=t.get("type") if t else None,
                        outcome=o.get("status") if o else None,
                        survival_days=o.get("survival_days") if o else None
                    ))
                
                return similar
        except Exception as e:
            logger.error(f"Failed to find similar patients: {e}")
            return []

    def get_guideline_outcomes(self, guideline_id: str) -> Dict[str, Any]:
        """
        Get outcome statistics for patients treated according to a specific guideline.
        READ-ONLY: Statistical aggregation only, NO medical reasoning.
        
        Args:
            guideline_id: Guideline rule ID (e.g., "R1", "R2")
            
        Returns:
            Dictionary with outcome statistics for the guideline
        """
        if not self._available:
            return {"guideline_id": guideline_id, "error": "Neo4j not available"}

        query = """
        MATCH (r:GuidelineRule {rule_id: $guideline_id})-[:RECOMMENDS]->(tp:TreatmentPlan)
        OPTIONAL MATCH (p:Patient)-[:RECEIVED_TREATMENT]->(tp)
        OPTIONAL MATCH (tp)-[:HAS_OUTCOME]->(o:Outcome)
        
        RETURN r.rule_id as guideline_id,
               r.name as guideline_name,
               tp.type as treatment_type,
               count(DISTINCT p) as patient_count,
               avg(o.survival_days) as avg_survival_days,
               collect(DISTINCT o.status) as outcome_statuses
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, guideline_id=guideline_id)
                record = result.single()
                
                if not record:
                    return {"guideline_id": guideline_id, "patient_count": 0}
                
                return {
                    "guideline_id": record["guideline_id"],
                    "guideline_name": record["guideline_name"],
                    "treatment_type": record["treatment_type"],
                    "patient_count": record["patient_count"] or 0,
                    "avg_survival_days": record["avg_survival_days"],
                    "outcome_statuses": record["outcome_statuses"] or []
                }
        except Exception as e:
            logger.error(f"Failed to get guideline outcomes: {e}")
            return {"guideline_id": guideline_id, "error": str(e)}

    def search_patients_by_snomed(self, snomed_codes: List[str]) -> List[Dict[str, Any]]:
        """
        Search for patients by SNOMED-CT codes.
        READ-ONLY: Data retrieval based on SNOMED codes.
        
        Args:
            snomed_codes: List of SNOMED-CT codes to search for
            
        Returns:
            List of patients with matching SNOMED codes
        """
        if not self._available:
            return []

        query = """
        MATCH (p:Patient)-[:HAS_CLINICAL_FINDING|HAS_HISTOLOGY|HAS_BIOMARKER]->(c)
        WHERE c.snomed_code IN $snomed_codes
        OPTIONAL MATCH (p)-[:HAS_CLINICAL_FINDING]->(cf:ClinicalFinding)
        OPTIONAL MATCH (cf)-[:HAS_HISTOLOGY]->(h:Histology)
        
        RETURN DISTINCT p.patient_id as patient_id,
               p.name as name,
               p.tnm_stage as tnm_stage,
               p.histology_type as histology_type,
               collect(DISTINCT c.snomed_code) as matching_codes,
               collect(DISTINCT c.snomed_term) as matching_terms
        LIMIT 50
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, snomed_codes=snomed_codes)
                patients = []
                for record in result:
                    patients.append({
                        "patient_id": record["patient_id"],
                        "name": record["name"],
                        "tnm_stage": record["tnm_stage"],
                        "histology_type": record["histology_type"],
                        "matching_codes": record["matching_codes"],
                        "matching_terms": record["matching_terms"]
                    })
                return patients
        except Exception as e:
            logger.error(f"Failed to search patients by SNOMED: {e}")
            return []

    def get_treatment_statistics(self, treatment_type: str) -> Dict[str, Any]:
        """
        Get outcome statistics for a specific treatment type.
        READ-ONLY: Statistical aggregation only, NO medical reasoning.
        
        Args:
            treatment_type: Treatment type (e.g., "Chemotherapy", "Surgery")
            
        Returns:
            Dictionary with treatment outcome statistics
        """
        if not self._available:
            return {"treatment_type": treatment_type, "error": "Neo4j not available"}

        query = """
        MATCH (tp:TreatmentPlan {type: $treatment_type})<-[:RECEIVED_TREATMENT]-(p:Patient)
        OPTIONAL MATCH (tp)-[:HAS_OUTCOME]->(o:Outcome)
        
        RETURN tp.type as treatment,
               count(DISTINCT p) as patient_count,
               avg(o.survival_days) as avg_survival_days,
               percentileDisc(o.survival_days, 0.5) as median_survival_days,
               collect(DISTINCT o.status) as outcome_types,
               avg(p.age_at_diagnosis) as avg_age,
               collect(DISTINCT p.tnm_stage) as stages_treated
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, treatment_type=treatment_type)
                record = result.single()
                
                if not record:
                    return {"treatment_type": treatment_type, "patient_count": 0}
                
                return {
                    "treatment_type": record["treatment"],
                    "patient_count": record["patient_count"] or 0,
                    "avg_survival_days": record["avg_survival_days"],
                    "median_survival_days": record["median_survival_days"],
                    "outcome_types": record["outcome_types"] or [],
                    "avg_age": record["avg_age"],
                    "stages_treated": record["stages_treated"] or []
                }
        except Exception as e:
            logger.error(f"Failed to get treatment statistics: {e}")
            return {"treatment_type": treatment_type, "error": str(e)}

    def close(self):
        if self.driver:
            self.driver.close()


class Neo4jWriteTools:
    """
    WRITE-ONLY Neo4j operations.
    CRITICAL: ONLY PersistenceAgent can use these methods.
    Complete audit trail for all write operations.
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            self._available = True
            logger.info(f"✓ Neo4j WRITE connection established: {self.uri}")
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}")
            self._available = False
            self.driver = None

    @property
    def is_available(self) -> bool:
        return self._available

    def save_patient_facts(self, patient_fact: PatientFact, embedding: list = None) -> str:
        """
        Save patient data to Neo4j with optional embedding for semantic search.
        WRITE: Creates/updates Patient node with all LUCADA properties.
        
        Args:
            patient_fact: Patient data to save
            embedding: Optional embedding vector for semantic similarity search
        
        Returns:
            Patient ID if successful, empty string otherwise
        """
        if not self._available:
            logger.warning("Neo4j not available - skipping patient save")
            return ""

        # Build query with optional embedding
        set_embedding = ", p.embedding = $embedding" if embedding else ""
        
        query = f"""
        MERGE (p:Patient {{patient_id: $patient_id}})
        SET p.name = $name,
            p.sex = $sex,
            p.age_at_diagnosis = $age_at_diagnosis,
            p.tnm_stage = $tnm_stage,
            p.histology_type = $histology_type,
            p.performance_status = $performance_status,
            p.laterality = $laterality,
            p.fev1_percent = $fev1_percent,
            p.diagnosis = $diagnosis,
            p.comorbidities = $comorbidities,
            p.notes = $notes,
            p.created_at = coalesce(p.created_at, datetime()),
            p.updated_at = datetime(){set_embedding}
        
        MERGE (cf:ClinicalFinding {{id: $patient_id + '_finding'}})
        SET cf.diagnosis = $diagnosis,
            cf.tnm_stage = $tnm_stage,
            cf.basis_of_diagnosis = $basis_of_diagnosis
        
        MERGE (h:Histology {{type: $histology_type}})
        
        MERGE (p)-[:HAS_CLINICAL_FINDING]->(cf)
        MERGE (cf)-[:HAS_HISTOLOGY]->(h)
        
        RETURN p.patient_id as patient_id
        """
        
        try:
            params = dict(
                patient_id=patient_fact.patient_id,
                name=patient_fact.name,
                sex=patient_fact.sex,
                age_at_diagnosis=patient_fact.age_at_diagnosis,
                tnm_stage=patient_fact.tnm_stage,
                histology_type=patient_fact.histology_type,
                performance_status=patient_fact.performance_status,
                laterality=patient_fact.laterality,
                fev1_percent=patient_fact.fev1_percent,
                diagnosis=patient_fact.diagnosis,
                comorbidities=patient_fact.comorbidities,
                notes=patient_fact.notes,
                basis_of_diagnosis=patient_fact.basis_of_diagnosis
            )
            if embedding:
                params["embedding"] = embedding
                
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **params)
                record = result.single()
                
                log_msg = f"✓ Saved patient facts: {patient_fact.patient_id}"
                if embedding:
                    log_msg += f" (with {len(embedding)}-dim embedding)"
                logger.info(log_msg)
                return patient_fact.patient_id
        except Exception as e:
            logger.error(f"Failed to save patient facts: {e}")
            return ""

    def save_inference_result(self, inference: InferenceRecord) -> str:
        """
        Save inference result to Neo4j with full audit trail.
        WRITE: Creates Inference node linked to Patient.
        
        Returns:
            Inference ID if successful, empty string otherwise
        """
        if not self._available:
            logger.warning("Neo4j not available - skipping inference save")
            return ""

        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        
        CREATE (i:Inference {
            inference_id: $inference_id,
            status: $status,
            created_at: datetime(),
            agent_version: $agent_version,
            workflow_duration_ms: $workflow_duration_ms,
            ingestion_result: $ingestion_result,
            classification_result: $classification_result,
            explanation_result: $explanation_result
        })
        
        CREATE (p)-[:HAS_INFERENCE]->(i)
        
        RETURN i.inference_id as inference_id
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    patient_id=inference.patient_id,
                    inference_id=inference.inference_id,
                    status=inference.status,
                    agent_version=inference.agent_version,
                    workflow_duration_ms=inference.workflow_duration_ms,
                    ingestion_result=str(inference.ingestion_result) if inference.ingestion_result else None,
                    classification_result=str(inference.classification_result) if inference.classification_result else None,
                    explanation_result=str(inference.explanation_result) if inference.explanation_result else None
                )
                record = result.single()
                
                logger.info(f"✓ Saved inference result: {inference.inference_id}")
                return inference.inference_id
        except Exception as e:
            logger.error(f"Failed to save inference result: {e}")
            return ""

    def mark_inference_obsolete(self, patient_id: str, current_inference_id: str) -> int:
        """
        Mark previous inferences as obsolete (superseded by new inference).
        WRITE: Updates status of existing Inference nodes.
        
        Returns:
            Number of inferences marked obsolete
        """
        if not self._available:
            logger.warning("Neo4j not available - skipping mark obsolete")
            return 0

        query = """
        MATCH (p:Patient {patient_id: $patient_id})-[:HAS_INFERENCE]->(i:Inference)
        WHERE i.inference_id <> $current_inference_id AND i.status <> 'obsolete'
        SET i.status = 'obsolete',
            i.obsolete_reason = 'Superseded by new inference',
            i.superseded_by = $current_inference_id,
            i.obsolete_at = datetime()
        RETURN count(i) as updated_count
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, patient_id=patient_id, current_inference_id=current_inference_id)
                record = result.single()
                count = record["updated_count"] if record else 0
                
                logger.info(f"✓ Marked {count} inferences obsolete for patient {patient_id}")
                return count
        except Exception as e:
            logger.error(f"Failed to mark inferences obsolete: {e}")
            return 0

    def update_inference_status(self, inference_id: str, status: InferenceStatus) -> WriteReceipt:
        """
        Update inference status.
        WRITE: Updates status field on Inference node.
        """
        if not self._available:
            return WriteReceipt(
                operation="update_inference_status",
                node_type="Inference",
                node_id=inference_id,
                success=False,
                message="Neo4j not available"
            )

        query = """
        MATCH (i:Inference {inference_id: $inference_id})
        SET i.status = $status,
            i.updated_at = datetime()
        RETURN i.inference_id as inference_id
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    inference_id=inference_id,
                    status=status.value if hasattr(status, 'value') else status
                )
                record = result.single()
                
                if record:
                    logger.info(f"✓ Updated inference status: {inference_id} -> {status}")
                    return WriteReceipt(
                        operation="update_inference_status",
                        node_type="Inference",
                        node_id=inference_id,
                        success=True,
                        message=f"Status updated to {status}",
                        properties_written=2
                    )
                else:
                    return WriteReceipt(
                        operation="update_inference_status",
                        node_type="Inference",
                        node_id=inference_id,
                        success=False,
                        message="Inference not found"
                    )
        except Exception as e:
            logger.error(f"Failed to update inference status: {e}")
            return WriteReceipt(
                operation="update_inference_status",
                node_type="Inference",
                node_id=inference_id,
                success=False,
                message=str(e)
            )

    def save_treatment_recommendation(self, patient_id: str, inference_id: str,
                                       recommendation: Dict[str, Any]) -> str:
        """
        Save treatment recommendation to Neo4j.
        WRITE: Creates TreatmentPlan and links to Inference and Patient.
        
        Args:
            patient_id: Patient identifier
            inference_id: Inference identifier
            recommendation: Dict with 'treatment', 'rank', 'evidence_level', etc.
            
        Returns:
            Recommendation node ID
        """
        if not self._available:
            logger.warning("Neo4j not available - skipping recommendation save")
            return ""

        # Extract fields from recommendation dict
        treatment = recommendation.get('treatment', 'Unknown')
        rank = recommendation.get('rank', 0)
        evidence_level = recommendation.get('evidence_level', 'Unknown')
        guideline_ref = recommendation.get('guideline_reference', '')
        rationale = recommendation.get('rationale', '')

        query = """
        MATCH (p:Patient {patient_id: $patient_id})
        MATCH (i:Inference {inference_id: $inference_id})
        
        CREATE (rec:Recommendation {
            recommendation_id: randomUUID(),
            treatment: $treatment,
            rank: $rank,
            evidence_level: $evidence_level,
            guideline_reference: $guideline_ref,
            rationale: $rationale,
            timestamp: datetime()
        })
        
        CREATE (i)-[:HAS_RECOMMENDATION]->(rec)
        CREATE (rec)-[:FOR_PATIENT]->(p)
        
        RETURN rec.recommendation_id as rec_id
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    patient_id=patient_id,
                    inference_id=inference_id,
                    treatment=treatment,
                    rank=rank,
                    evidence_level=evidence_level,
                    guideline_ref=guideline_ref,
                    rationale=rationale
                )
                record = result.single()
                rec_id = record["rec_id"] if record else ""
                logger.info(f"✓ Saved recommendation: {treatment}")
                return rec_id
        except Exception as e:
            logger.error(f"Failed to save recommendation: {e}")
            return ""

    def close(self):
        if self.driver:
            self.driver.close()


class Neo4jTools:
    """
    Combined Neo4j tools with explicit read/write separation.
    Provides both read and write capabilities with clear method naming.
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = None):
        self.read = Neo4jReadTools(uri, user, password, database)
        self.write = Neo4jWriteTools(uri, user, password, database)

    @property
    def is_available(self) -> bool:
        return self.read.is_available or self.write.is_available

    def close(self):
        self.read.close()
        self.write.close()


def setup_neo4j_schema():
    """Initialize Neo4j schema with all required constraints and indexes."""
    tools = Neo4jWriteTools()
    
    if not tools.is_available:
        logger.warning("Neo4j not available, skipping schema setup")
        return False

    schema_queries = [
        # Constraints
        "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE",
        "CREATE CONSTRAINT inference_id IF NOT EXISTS FOR (i:Inference) REQUIRE i.inference_id IS UNIQUE",
        "CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (r:GuidelineRule) REQUIRE r.rule_id IS UNIQUE",
        
        # Indexes
        "CREATE INDEX patient_stage IF NOT EXISTS FOR (p:Patient) ON (p.tnm_stage)",
        "CREATE INDEX patient_histology IF NOT EXISTS FOR (p:Patient) ON (p.histology_type)",
        "CREATE INDEX inference_status IF NOT EXISTS FOR (i:Inference) ON (i.status)",
        
        # Full-text search
        "CREATE FULLTEXT INDEX patient_search IF NOT EXISTS FOR (p:Patient) ON EACH [p.name, p.notes]"
    ]
    
    try:
        with tools.driver.session(database=tools.database) as session:
            for query in schema_queries:
                try:
                    session.run(query)
                    logger.debug(f"Executed: {query[:50]}...")
                except Exception as e:
                    logger.debug(f"Schema query skipped: {e}")
        
        logger.info("✓ Neo4j schema initialized")
        tools.close()
        return True
    except Exception as e:
        logger.error(f"Failed to setup schema: {e}")
        tools.close()
        return False
